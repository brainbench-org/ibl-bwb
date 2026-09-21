import multiprocessing as mp
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.util import get_num_params, get_num_trainable_params, human_readable
from ibl_bwb_eval.entity_ids import encode_entity_ids
from ibl_bwb_eval.multi_unit import multi_unit_prediction
from ibl_bwb_eval.predictions import PredictionsWriter
from ibl_bwb_eval.tasks import get_ts3_readout_spec, task_id

from .cache import update_lolcat_cache
from .dataset import LOLCATDataset
from .lolcat import LOLCAT
from .sampler import LossFeedbackSampler


class LOLCATTrainer(BaseTrainer):
    """Supervised training of LOLCAT, the one TS3 model fit on the region labels."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")

        self.train_losses = None
        self.train_labels = None

        self.setup_model(ckpt)
        # one writer, and every rank waits before the loaders read the cache
        if self.rank == 0:
            update_lolcat_cache(
                self.cfg.data_root,
                self.cfg.model.n_bins,
                self.cfg.t_min,
                self.cfg.t_max,
                self.cfg.cache_path,
                log_bins=self.cfg.log_bins,
            )
        self.barrier()
        self.setup_train_loader()
        self.setup_val_loader()
        self.setup_test_loader()
        self.model.link_datasets(self.train_dataset, self.val_dataset, self.test_dataset)

        self.model.to(self.device)
        self.model = self.make_ddp(self.model)

        self.setup_optimizers(ckpt)

    def train_epoch(self):
        self.model.train()
        train_losses, train_labels = [], []
        for x, batch_idx, labels, _ in self.logger.get_pbar(self.train_loader, prefix="train"):
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x, batch_idx)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            per_sample_loss.mean().backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.logger.log("train/loss", per_sample_loss.mean().item(), pbar=True)
            train_losses.append(per_sample_loss.detach().cpu())
            train_labels.append(labels.cpu())
            self.train_step += 1
        self.push_logs()
        self.train_losses = torch.cat(train_losses)
        self.train_labels = torch.cat(train_labels)

    @torch.inference_mode()
    def _evaluate(self, loader: DataLoader, split: str) -> tuple:
        """One pass: mean loss, per-sample losses, targets, predictions, probabilities, uids."""
        self.model.eval()
        all_preds, all_proba, all_labels, all_losses, all_uids = [], [], [], [], []
        loss_sum, loss_count = 0.0, 0

        for x, batch_idx, labels, uids in loader:
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(x, batch_idx)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            loss_sum += float(per_sample_loss.sum().item())
            loss_count += len(labels)

            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_proba.append(F.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_losses.append(per_sample_loss.cpu())
            all_uids.extend(uids)

        preds = np.concatenate(all_preds)
        proba = np.concatenate(all_proba)
        targets = np.concatenate(all_labels)
        avg_loss = float("nan") if loss_count == 0 else loss_sum / loss_count
        self.logger.log(f"{split}/loss", avg_loss)
        return avg_loss, torch.cat(all_losses), torch.from_numpy(targets), preds, proba, all_uids

    def val_epoch(self) -> bool:
        avg_loss, val_losses, val_labels, preds, _, _ = self._evaluate(self.val_loader, "val")

        # a selection number over the held-out subjects, not the suite's reported metric:
        # what test() logs is scored against the task's full vocabulary instead
        targets = val_labels.numpy()
        per_class_f1 = f1_score(targets, preds, labels=self._labels, average=None)
        metrics = {
            "val/loss": avg_loss,
            "val/bacc": balanced_accuracy_score(targets, preds),
            "val/f1": f1_score(targets, preds, average="macro"),
            **{
                f"val/f1/{self._idx_to_label[i]}": float(per_class_f1[i])
                for i in range(len(per_class_f1))
            },
            "val/epoch": self.epoch,
        }
        self.logger.log_dict(metrics)
        self.return_value = metrics

        if self.train_losses is None:
            self.logger.info("Sampler: not stepped, no training losses yet")
        else:
            self.sampler.step(
                train_loss=self.train_losses,
                train_labels=self.train_labels,
                val_loss=val_losses,
                val_labels=val_labels,
            )
            self.logger.log_dict(
                {
                    f"sampler/factor/{self._idx_to_label[i]}": float(factor)
                    for i, factor in enumerate(self.sampler.factors)
                }
            )

        improved = self.save_if_best(metrics[self.cfg.val.metric])
        early_stop = self.step_patience(improved)
        self.push_logs()
        return early_stop

    def test(self):
        if self.rank != 0:
            return

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
            self.logger.info(f"Loaded best model from epoch {self.best_epoch}")

        _, _, _, preds, proba, uids = self._evaluate(self.test_loader, "test")
        spec = get_ts3_readout_spec(self.cfg.task)

        # the pooling and the targets below are indexed by dataset order, so the loader
        # must not have reordered anything
        assert np.array_equal(np.asarray(uids), self.test_dataset.uids), (
            "test loader returned units out of dataset order"
        )

        averaged_proba = multi_unit_prediction(
            proba, self.test_dataset.depths, self.test_dataset.probe_ids
        )
        pred_multi = np.argmax(averaged_proba, axis=1)
        targets = self.test_dataset.labels
        names = np.array(spec.label_names)

        # the scorer a submission is scored by offline, so a run's numbers and the
        # leaderboard's cannot be computed differently
        scored = {}
        for scope, values in (("single", preds), ("multi", pred_multi)):
            scored[scope] = spec.score(names[targets], names[values])
            self.logger.log_dict(
                {
                    f"test/{self.cfg.task}/{scope}/{name}": value
                    for name, value in scored[scope].items()
                }
            )
        self.push_logs()

        self._save_preds(uids, proba, averaged_proba)

        target_names = [self._idx_to_label[i] for i in range(len(self._idx_to_label))]
        result_strs = []
        for scope, values in (("single", preds), ("multi", pred_multi)):
            result_strs.append(f"{'-' * 20} {scope.capitalize()} Unit {'-' * 20}")
            result_strs.append(
                classification_report(
                    targets, values, labels=self._labels, target_names=target_names, digits=5
                )
            )
        self.logger.info("\n" + "\n".join(result_strs))

        for scope, metrics in scored.items():
            f1s = [f"{metrics[f'{region}/f1-score']:.5f}" for region in spec.label_names]
            self.logger.info(f"\n=== {scope.capitalize()} Unit F1 ===\n" + "\n".join(f1s))

    def _save_preds(self, uids, proba: np.ndarray, averaged_proba: np.ndarray):
        """Write the two scored variants as submission files, if save_preds is enabled."""
        base_label = OmegaConf.select(self.cfg, "save_preds.label", default=None)
        writer_kwargs = {
            "enable": OmegaConf.select(self.cfg, "save_preds.enable", default=False),
            "base_path": Path(OmegaConf.select(self.cfg, "save_preds.path", default="")),
            "task": task_id("ts3", self.cfg.task),
            "seed": self.cfg.seed,
            "rank": self.rank,
            "metadata": {
                "label_names": ",".join(get_ts3_readout_spec(self.cfg.task).label_names),
                "unit_filtering": self.test_dataset.unit_filtering.label,
                "dataset_version": self.test_dataset.dataset_version,
            },
        }
        entity_ids = encode_entity_ids(np.asarray(uids))
        for variant, values in (("single", proba), ("multi", averaged_proba)):
            writer = PredictionsWriter(
                label=None if base_label is None else f"{base_label}_{variant}",
                **writer_kwargs,
            )
            writer.set(entity_ids=entity_ids, pred_proba=torch.from_numpy(values).float())
            writer.save(logger=self.logger)

    def _make_loader(
        self, dataset: LOLCATDataset, *, shuffle: bool, drop_last: bool, sampler=None
    ) -> DataLoader:
        has_workers = self.cfg.num_workers > 0
        return DataLoader(
            dataset=dataset,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=LOLCAT.collate,
            batch_size=self.cfg.batch_size // self.world_size,
            drop_last=drop_last,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

    def setup_train_loader(self):
        self.train_dataset = LOLCATDataset(
            data_root=self.cfg.data_root,
            regime="pretrain",
            split="train",
            val_fraction=self.cfg.val_fraction,
            task=self.cfg.task,
            cache_path=self.cfg.cache_path,
        )
        self._label_map = self.train_dataset.label_map
        self._idx_to_label = {v: k for k, v in self._label_map.items()}
        self._labels = list(range(len(self._idx_to_label)))

        labels = torch.from_numpy(self.train_dataset.labels)
        self.sampler = LossFeedbackSampler(
            labels=labels,
            factor=self.cfg.oversampling_factor,
            grow_divisor=self.cfg.grow_divisor,
            shrink_divisor=self.cfg.shrink_divisor,
            overfit_margin=self.cfg.overfit_margin,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        self.train_loader = self._make_loader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            sampler=DistributedSamplerWrapper(self.sampler),
        )
        self.logger.info(
            f"Train dataset: {len(self.train_dataset)} units, "
            f"{self.train_dataset.n_classes} classes, "
            f"{len(self.train_loader)} batches"
        )

    def setup_val_loader(self):
        self.val_dataset = LOLCATDataset(
            data_root=self.cfg.data_root,
            regime="pretrain",
            split="val",
            val_fraction=self.cfg.val_fraction,
            task=self.cfg.task,
            label_map=self._label_map,
            cache_path=self.cfg.cache_path,
        )
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False, drop_last=False)
        self.logger.info(f"Val dataset: {len(self.val_dataset)} units")

    def setup_test_loader(self):
        self.test_dataset = LOLCATDataset(
            data_root=self.cfg.data_root,
            regime="eval",
            split="test",
            val_fraction=self.cfg.val_fraction,
            task=self.cfg.task,
            label_map=self._label_map,
            cache_path=self.cfg.cache_path,
        )
        self.test_loader = self._make_loader(self.test_dataset, shuffle=False, drop_last=False)
        self.logger.info(f"Test dataset: {len(self.test_dataset)} units")

    def setup_model(self, ckpt: dict | None = None):
        # the head width is task-determined, so it comes from the spec, not the config
        n_classes = get_ts3_readout_spec(self.cfg.task).dim
        self.model = hydra.utils.instantiate(self.cfg.model, n_classes=n_classes)
        if not isinstance(self.model, LOLCAT):
            raise TypeError(
                f"{type(self).__name__} requires a LOLCAT model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)
        self.logger.info(f"Model: {self.model.__class__}, n_classes={n_classes}")

        num_params, skipped_params = get_num_params(self.model)
        num_params_encoder, _ = get_num_params(self.model.encoder)
        num_params_pool, _ = get_num_params(self.model.pool)
        num_params_classifier, _ = get_num_params(self.model.classifier)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of encoder params: {human_readable(num_params_encoder)} ({num_params_encoder:,})"
        )
        self.logger.info(
            f"Number of pool params: {human_readable(num_params_pool)} ({num_params_pool:,})"
        )
        self.logger.info(
            f"Number of classifier params: {human_readable(num_params_classifier)} ({num_params_classifier:,})"
        )
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info("Model: loaded state from checkpoint")

    def setup_optimizers(self, ckpt: dict | None = None):
        if self.model is None:
            raise ValueError("Trying to set up optimizers before the model is setup")
        if self.train_loader is None:
            raise ValueError("Trying to set up optimizers before the train data loader is setup")

        grouped_parameters = self.get_param_groups()

        self.optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=self.cfg.base_lr,
        )

        steps_per_epoch = len(self.train_loader)
        warmup_steps = 1 * steps_per_epoch
        total_steps = steps_per_epoch * self.cfg.num_epochs
        s1 = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.cfg.base_lr * self.cfg.lr_decay,
        )
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[s1, s2],
            milestones=[warmup_steps],
        )
        self.logger.info(
            f"LR Scheduler: steps_per_epoch={steps_per_epoch}, "
            f"warmup_steps={warmup_steps}, total_steps={total_steps}"
        )
        self.logger.info(f"LR Scheduler: {self.lr_scheduler}")

        self.add_checkpoint_items(
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
