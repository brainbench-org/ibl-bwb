import torch
from torch.utils.data import DataLoader

from core.dataset import IBLBrainWideBench2026
from core.trainer import BaseTrainer

from .my_model import MyModel


class MyModelPretrain(BaseTrainer):
    def setup(self, ckpt):
        # Direction of `save_if_best`, since the default is maximizing. Drop this if
        # your val_epoch selects on a score rather than a loss.
        self.reset_best_tracking(minimize=True)
        self.setup_train_loader()
        self.setup_val_loader()
        self.setup_model(ckpt)
        self.model.to(self.device)
        self.model = self.make_ddp(self.model)
        self.setup_optimizers(ckpt)

    def setup_train_loader(self):
        # TODO: replace with your dataset class and sampling strategy
        self.train_dataset = IBLBrainWideBench2026(
            root=self.cfg.data_root,
            recording_ids=self.cfg.recording_ids,
            split="train",
            regime="pretrain",
        )
        self.train_loader = DataLoader(self.train_dataset)

    def setup_val_loader(self):
        # TODO: replace with your dataset class and sampling strategy
        self.val_dataset = IBLBrainWideBench2026(
            root=self.cfg.data_root,
            recording_ids=self.cfg.recording_ids,
            split="val",
            regime="pretrain",
        )
        self.val_loader = DataLoader(self.val_dataset)

    def setup_model(self, ckpt):
        self.model = MyModel(dim=self.cfg.model.dim)
        self.model.link_datasets(self.train_loader.dataset, self.val_loader.dataset)
        self.add_checkpoint_items(model=self.model)

    def setup_optimizers(self, ckpt):
        # `get_param_groups` keeps weight decay off biases, norms and lookup tables.
        self.optimizer = torch.optim.AdamW(
            self.get_param_groups(),
            lr=self.cfg.base_lr,
        )
        self.add_checkpoint_items(optimizer=self.optimizer)

    def train_epoch(self):
        self.model.train()
        for _batch in self.train_loader:
            self.optimizer.zero_grad()
            # TODO: compute loss and call loss.backward()
            raise NotImplementedError
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            # yours to increment: the base only checkpoints and restores it
            self.train_step += 1

    def val_epoch(self):
        self.model.eval()
        # TODO: compute and log validation metrics
        metrics = {}
        raise NotImplementedError
        # `save_if_best` keeps the best weights, `step_patience` turns that into the
        # early-stop verdict `train` breaks on.
        improved = self.save_if_best(metrics["loss"], metrics=metrics)
        # `best_metrics` carries `best/val/avg`, the key tuning and sweeps select on.
        self.return_value = metrics | self.best_metrics
        return self.step_patience(improved)

    def log_training_results(self):
        pass
