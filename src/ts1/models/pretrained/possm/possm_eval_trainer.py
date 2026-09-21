import hydra
import torch
from torch_brain.transforms.container import Compose

from core.utils.util import get_num_params, get_num_trainable_params, human_readable
from ibl_bwb_eval.tasks import TargetResolution
from pretrain.models.possm import POSSM
from ts1.ts1_eval_trainer import TS1EvalTrainer


class POSSMEvalTrainer(TS1EvalTrainer):
    # Output query keys built by the multi-task input_fn; the eval path
    # rebuilds them from the supervised target so it works at test time
    # (where the value, but not the timestamp, is stripped from data).
    _INPUT_FN_OUTPUT_KEYS = (
        "output_timestamps",
        "output_decoder_index",
        "output_bin_index",
        "output_session_index",
    )

    def setup_model(self, ckpt: dict | None):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, POSSM):
            raise TypeError(
                f"{type(self).__name__} requires a POSSM model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def predict(self, X, target_timestamps=None, target_mask=None):
        model_inputs = {
            k: v for k, v in X["model_inputs"].items() if k not in self._INPUT_FN_OUTPUT_KEYS
        }

        session_index = X["session_index"].to(torch.long)
        B = session_index.shape[0]
        device = session_index.device
        task_idx = self.model.task_name_to_index[self.readout_spec.id]

        if self.readout_spec.target_resolution == TargetResolution.TIMESTEP:
            assert target_timestamps is not None, (
                "target_timestamps required for timestep-level eval"
            )
            out_ts = target_timestamps.to(torch.float32)
            if out_ts.ndim == 1:
                out_ts = out_ts.unsqueeze(0)
        else:
            # Sequence-level: single query at end of context window per sample
            out_ts = torch.full(
                (B, 1),
                float(self.model.context_duration),
                dtype=torch.float32,
                device=device,
            )

        T = out_ts.shape[-1]
        out_decoder_idx = torch.full((B, T), task_idx, dtype=torch.long, device=device)
        out_session_idx = session_index.view(B, 1).expand(B, T).contiguous()
        out_bin_idx = self.model.compute_bin_index(out_ts)

        output = self.model(
            **model_inputs,
            output_timestamps=out_ts,
            output_decoder_index=out_decoder_idx,
            output_bin_index=out_bin_idx,
            output_session_index=out_session_idx,
        )
        if self.readout_spec.target_resolution == TargetResolution.SEQUENCE:
            # sequence-level: one query per sample, (B, D) -> (B, 1, D), to make it compatible
            # with TS1TestMixin and TS1EvalTrainer.loss().
            output = output.unsqueeze(1)
        return output

    def link_model(self, model: POSSM, ckpt: dict | None):
        # attach model transforms to end of dataset transform pipelines
        for split in ["train", "val", "test"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        # link datasets to model
        model.link_datasets(
            self.train_loader.dataset,
            self.val_loader.dataset,
            self.test_loader.dataset,
        )
        model.configure_readout(self.readout_spec)
        if self.cfg.finetuning.enable:
            model.load_ckpt(ckpt)

        num_params, skipped_params = get_num_params(self.model)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        num_embedding_params = get_num_params(
            self.model.unit_emb, return_skipped=False
        ) + get_num_params(self.model.session_emb, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        self.logger.info(
            f"Number of embedding parameters: {human_readable(num_embedding_params)} ({num_embedding_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")
