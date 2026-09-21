# ----------------------------------------------------------------------------
# POYO+ for IBL benchmark.
# Implements multitask query tokens and multitask readout while keeping the
# original spike-token / variable-length batching path used by the benchmark.
# ----------------------------------------------------------------------------

import logging

import numpy as np
import torch
import torch.nn as nn
from torch_brain.batching import chain, track_batch
from torch_brain.data import Data
from torch_brain.models.poyo import (
    FeedForward,
    create_linspace_latent_tokens,
    create_start_end_unit_tokens,
)
from torch_brain.nn import (
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding, MultitaskReadout
from core.nn.varlen_attention import (
    AttnImpl,
    cross_attn,
    self_attn,
    uses_xformers,
    validate_attn_impl,
)
from core.utils.checkpoint import log_incompatible_keys
from core.utils.util import get_cli_logger
from ibl_bwb_eval.tasks import (
    TARGET_WINDOW,
    ReadoutSpec,
    TargetResolution,
    TS1ReadoutSpec,
    get_ts1_supported_tasks,
)


class POYOPlus(BaseModel):
    """POYO+ over chained spike tokens :cite:`poyo_plus`.

    ``attn_impl`` picks the kernel behind every attention layer. The default ``nested``
    runs on stock torch; ``xformers`` needs the optional ``xformers`` extra
    (``uv pip install -e ".[train,xformers]"``) and is worth installing for anything
    long-running. Its main win is memory -- a substantially lower peak, which is often what
    decides whether a batch size fits on one GPU -- with a smaller speedup on top. The two
    compute the same attention, so checkpoints are interchangeable: switching backends needs
    no retraining and no state-dict surgery.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.
    """

    def __init__(
        self,
        *,
        latent_step: float,
        num_latents_per_step: int = 64,
        dim: int = 512,
        depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
        task_vocab: list[str] | None = None,
        attn_impl: AttnImpl = "nested",
    ):
        super().__init__()

        self.logger = get_cli_logger()
        self.attn_impl = validate_attn_impl(attn_impl)
        use_xformers = uses_xformers(self.attn_impl)
        self.dim = dim
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.task_vocab = list(task_vocab) if task_vocab is not None else get_ts1_supported_tasks()
        # Reserve index 0 as padding/default.
        self.task_name_to_index = {task_name: i + 1 for i, task_name in enumerate(self.task_vocab)}

        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(len(self.task_vocab) + 1, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents_per_step, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            use_xformers=use_xformers,
        )
        self.enc_ffn = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))

        self.proc_layers = nn.ModuleList([])
        for _ in range(depth):
            self.proc_layers.append(
                nn.Sequential(
                    RotarySelfAttention(
                        dim=dim,
                        heads=self_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                        rotate_value=True,
                        use_xformers=use_xformers,
                    ),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, dropout=ffn_dropout),
                    ),
                )
            )

        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
            use_xformers=use_xformers,
        )
        self.dec_ffn = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))

        self.readout_specs: dict[str, TS1ReadoutSpec] = {}
        self.readout: MultitaskReadout | None = None
        self.context_duration: float | None = None

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
        finetune_enable: bool = False,
    ):
        self.context_duration = train_dataset.context_window
        self._validate_params(self.context_duration, self.latent_step)

        self.unit_ids = list(
            set(train_dataset.get_unit_ids())
            | set(val_dataset.get_unit_ids())
            | (set(test_dataset.get_unit_ids()) if test_dataset is not None else set())
        )
        self.session_ids = list(
            set(train_dataset.get_session_ids())
            | set(val_dataset.get_session_ids())
            | (set(test_dataset.get_session_ids()) if test_dataset is not None else set())
        )

        if not finetune_enable:
            if self.unit_emb.is_lazy():
                self.unit_emb.initialize_vocab(self.unit_ids)
            else:
                self.unit_emb.extend_vocab(self.unit_ids, exist_ok=True)
            if self.session_emb.is_lazy():
                self.session_emb.initialize_vocab(self.session_ids)
            else:
                self.session_emb.extend_vocab(self.session_ids, exist_ok=True)
        # finetune_enable: keep InfiniteVocabEmbedding lazy until load_ckpt(), then
        # extend_vocab / subset_vocab (same contract as POYO + POYOEvalTrainer).

    def load_ckpt(self, ckpt: dict):
        state_dict = ckpt["model_state_dict"]

        result = self.load_state_dict(state_dict, strict=False)
        self.logger.info("Loaded pretrained model from checkpoint")
        log_incompatible_keys(result, self.logger)

        self.unit_emb.extend_vocab(self.unit_ids, exist_ok=True)
        self.session_emb.extend_vocab(self.session_ids, exist_ok=True)

        self.unit_emb.subset_vocab(self.unit_ids)
        self.session_emb.subset_vocab(self.session_ids)

        self.logger.info("Initialized unit and session vocabularies")
        self.logger.info(f"Number of units: {len(self.unit_emb.vocab) - 1}")
        self.logger.info(f"Number of sessions: {len(self.session_emb.vocab) - 1}")

    def configure_multitask_readout(self, readout_specs: dict[str, TS1ReadoutSpec]):
        unknown = set(readout_specs) - set(self.task_name_to_index)
        if unknown:
            raise ValueError(f"Unknown tasks for POYOPlus: {sorted(unknown)}")

        self.readout_specs = dict(readout_specs)
        self.readout = MultitaskReadout(
            dim=self.dim,
            readout_specs=self.readout_specs,
            task_index=self.task_name_to_index,
        )

    def configure_readout(self, readout_spec: ReadoutSpec):
        self.configure_multitask_readout({readout_spec.id: readout_spec})

    @staticmethod
    def _extract_task_target(
        data: Data, readout_spec: TS1ReadoutSpec
    ) -> dict[str, np.ndarray] | None:
        try:
            values = data.get_nested_attribute(readout_spec.value_key)
        except AttributeError:
            return None

        if values is None:
            return None

        values = np.asarray(values)
        if values.size == 0:
            return None

        out: dict[str, np.ndarray] = {}

        if readout_spec.target_resolution == TargetResolution.SEQUENCE:
            interval_parent_key = ".".join(readout_spec.value_key.split(".")[:-1])
            try:
                interval = data.get_nested_attribute(interval_parent_key)
            except AttributeError:
                return None

            if not (hasattr(interval, "start") and hasattr(interval, "end")):
                return None

            start = np.asarray(interval.start)
            end = np.asarray(interval.end)
            num_values = np.atleast_1d(values).shape[0]
            if (
                start.size == 0
                or end.size == 0
                or start.size != end.size
                or start.size != num_values
            ):
                return None
            timestamps = ((start + end) / 2.0).astype(np.float32)

            if values.ndim == 0:
                values = values.reshape(1, 1)
            elif values.ndim == 1:
                values = values.reshape(-1, 1)

            out["timestamps"] = timestamps
            out["values"] = values
        else:
            if readout_spec.timestamp_key is None:
                raise ValueError(f"Timestep-level task {readout_spec.id} is missing timestamp_key")
            try:
                timestamps = np.asarray(data.get_nested_attribute(readout_spec.timestamp_key))
            except AttributeError:
                return None

            if timestamps.size == 0:
                return None

            if values.ndim == 0:
                values = values.reshape(1, 1)
            elif values.ndim == 1:
                values = values.reshape(-1, 1)

            n = min(len(timestamps), len(values))
            timestamps = timestamps[:n].astype(np.float32)
            values = values[:n]

            out["timestamps"] = timestamps
            out["values"] = values

            if readout_spec.mask_key is not None:
                try:
                    mask = np.asarray(data.get_nested_attribute(readout_spec.mask_key)).astype(bool)
                except AttributeError:
                    mask = None
                if mask is not None:
                    out["mask"] = mask[:n]

        if out["values"].dtype == np.float64:
            out["values"] = out["values"].astype(np.float32)

        return out

    def input_fn(self, data: Data) -> dict:
        if self.readout is None or not self.readout_specs:
            raise ValueError(
                "Readout has not been configured. Call configure_readout() or "
                "configure_multitask_readout() before building model inputs."
            )
        if self.context_duration is None:
            raise ValueError("Datasets have not been linked. Call link_datasets() first.")

        start, end = 0.0, self.context_duration

        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        se_token_type_index, se_unit_index, se_timestamps = create_start_end_unit_tokens(
            unit_ids, start, end
        )
        spike_token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(spike_unit_index)]
        )
        spike_unit_index = np.concatenate([se_unit_index, spike_unit_index])
        spike_timestamps = np.concatenate([se_timestamps, spike_timestamps])

        local_to_global_map = np.array(self.unit_emb.tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        raw_session_index = self.session_emb.tokenizer(data.session.id)
        session_index = int(np.asarray(raw_session_index).reshape(-1)[0])
        output_session_index = []
        output_timestamps = []
        output_decoder_index = []
        target_values = {}
        target_masks = {}

        for task_name, spec in self.readout_specs.items():
            target = self._extract_task_target(data, spec)
            if target is None:
                continue

            timestamps = np.asarray(target["timestamps"], dtype=np.float32)
            if timestamps.size == 0:
                continue

            task_index = self.task_name_to_index[task_name]
            output_session_index.append(np.full((len(timestamps),), session_index, dtype=np.int64))
            output_timestamps.append(timestamps)
            output_decoder_index.append(np.full((len(timestamps),), task_index, dtype=np.int64))
            target_values[task_name] = chain(np.asarray(target["values"]))
            if "mask" in target:
                target_masks[task_name] = chain(np.asarray(target["mask"], dtype=np.bool_))

        if output_timestamps:
            output_session_index = np.concatenate(output_session_index)
            output_timestamps = np.concatenate(output_timestamps)
            output_decoder_index = np.concatenate(output_decoder_index)
        else:
            output_session_index = np.empty((0,), dtype=np.int64)
            output_timestamps = np.empty((0,), dtype=np.float32)
            output_decoder_index = np.empty((0,), dtype=np.int64)

        # Windows with no targets are still batched alongside windows that have them,
        # and `collate` picks how to combine a batch from the type of its first element
        output_batch_index = track_batch(output_timestamps)

        batch = {
            "model_inputs": {
                "input_unit_index": chain(spike_unit_index),
                "input_timestamps": chain(spike_timestamps),
                "input_token_type": chain(spike_token_type_index),
                "input_seqlen": len(spike_unit_index),
                "latent_index": chain(latent_index),
                "latent_timestamps": chain(latent_timestamps),
                "latent_seqlen": len(latent_index),
                "output_session_index": chain(output_session_index),
                "output_timestamps": chain(output_timestamps),
                "output_decoder_index": chain(output_decoder_index),
                "output_batch_index": chain(output_batch_index),
            },
            # Compatibility with single-task eval trainer path.
            "session_index": raw_session_index,
            "target_values": chain(target_values, allow_missing_keys=True),
            "target_masks": chain(target_masks, allow_missing_keys=True),
        }
        return batch

    def forward(
        self,
        *,
        input_unit_index: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_token_type: torch.Tensor,
        input_seqlen: torch.Tensor | int,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
        latent_seqlen: torch.Tensor | int,
        output_session_index: torch.Tensor | None = None,
        output_timestamps: torch.Tensor | None = None,
        output_decoder_index: torch.Tensor | None = None,
        output_batch_index: torch.Tensor | None = None,
        return_dict: bool = True,
        unpack_output: bool = False,
        unflatten_output: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.readout is None or not self.readout_specs:
            raise ValueError("Readout has not been configured.")
        if self.unit_emb.is_lazy() or self.session_emb.is_lazy():
            raise ValueError("Vocabularies have not been initialized, call link_datasets().")

        if isinstance(input_seqlen, int):
            input_seqlen = torch.tensor([input_seqlen], device=input_unit_index.device)
        if isinstance(latent_seqlen, int):
            latent_seqlen = torch.tensor([latent_seqlen], device=input_unit_index.device)

        batch_size = int(input_seqlen.shape[0])

        if output_session_index is None:
            raise ValueError("output_session_index is required.")

        if output_timestamps is not None and output_timestamps.ndim == 2:
            # Eval path: (B, T) timestamps from target collation.
            B, T = output_timestamps.shape
            output_timestamps = output_timestamps.reshape(-1)
            if output_session_index.ndim == 1 and output_session_index.numel() == B:
                output_session_index = output_session_index.repeat_interleave(T)
            output_batch_index = torch.arange(B, device=output_timestamps.device).repeat_interleave(
                T
            )

        if output_timestamps is None:
            # Sequence-level fallback query: 1 output per sample at mid-window.
            ts = self.context_duration - TARGET_WINDOW / 2
            output_timestamps = torch.full(
                (batch_size,),
                ts,
                dtype=torch.float32,
                device=input_unit_index.device,
            )
            output_batch_index = torch.arange(batch_size, device=input_unit_index.device)

        if output_batch_index is None:
            # Pretrain path should provide this; fallback assumes one sample.
            output_batch_index = torch.zeros(
                output_timestamps.numel(),
                dtype=torch.long,
                device=output_timestamps.device,
            )

        if output_decoder_index is None or len(output_decoder_index) == 0:
            if len(self.readout_specs) != 1:
                raise ValueError(
                    "output_decoder_index is required when multiple readouts are active."
                )
            task_name = next(iter(self.readout_specs.keys()))
            task_index = self.task_name_to_index[task_name]
            output_decoder_index = torch.full(
                (output_timestamps.numel(),),
                task_index,
                dtype=torch.long,
                device=output_timestamps.device,
            )

        output_seqlen = torch.bincount(
            output_batch_index.to(dtype=torch.long), minlength=batch_size
        ).to(dtype=torch.int32)

        if output_timestamps.numel() == 0:
            if return_dict:
                return {}
            if len(self.readout_specs) != 1:
                raise ValueError("Cannot return a single tensor with multiple active tasks.")
            spec = next(iter(self.readout_specs.values()))
            return torch.empty(
                (batch_size, 0, spec.dim),
                device=input_unit_index.device,
                dtype=torch.float32,
            )

        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        output_queries = self.session_emb(output_session_index) + self.task_emb(
            output_decoder_index
        )
        output_queries_base = output_queries

        output_timestamp_emb = self.rotary_emb(output_timestamps)

        latents = latents + cross_attn(
            self.enc_atn,
            x_query=latents,
            x_context=inputs,
            query_pos_emb=latent_timestamp_emb,
            context_pos_emb=input_timestamp_emb,
            query_seqlen=latent_seqlen,
            context_seqlen=input_seqlen,
            impl=self.attn_impl,
        )
        latents = latents + self.enc_ffn(latents)

        for proc_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn(
                    proc_attn,
                    x=latents,
                    rotary_time_emb=latent_timestamp_emb,
                    x_seqlen=latent_seqlen,
                    impl=self.attn_impl,
                )
            )
            latents = latents + self.dropout(self_ff(latents))

        output_queries = (
            cross_attn(
                self.dec_atn,
                x_query=output_queries_base,
                x_context=latents,
                query_pos_emb=output_timestamp_emb,
                context_pos_emb=latent_timestamp_emb,
                query_seqlen=output_seqlen,
                context_seqlen=latent_seqlen,
                impl=self.attn_impl,
            )
            + output_queries_base
        )
        output_latents = output_queries + self.dec_ffn(output_queries)

        output = self.readout.forward_varlen(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
            output_batch_index=output_batch_index,
            unpack_output=unpack_output,
        )

        if return_dict:
            return output

        if len(self.readout_specs) != 1:
            raise ValueError(
                "return_dict=False is only supported when exactly one readout is active."
            )

        task_name = next(iter(self.readout_specs.keys()))
        task_output = output.get(task_name)
        if task_output is None:
            spec = self.readout_specs[task_name]
            return torch.empty(
                (batch_size, 0, spec.dim),
                device=input_unit_index.device,
                dtype=torch.float32,
            )

        if unflatten_output:
            task_index = self.task_name_to_index[task_name]
            task_mask = output_decoder_index == task_index
            task_counts = torch.bincount(output_batch_index[task_mask], minlength=batch_size).to(
                dtype=torch.int32
            )
            if task_counts.numel() > 0 and not torch.all(task_counts == task_counts[0]):
                raise ValueError(
                    "Cannot unflatten POYOPlus output because per-sample query counts differ."
                )
            if task_counts.numel() > 0:
                task_output = task_output.view(batch_size, int(task_counts[0].item()), -1)

        return task_output

    def _validate_params(self, context_duration, latent_step):
        if not isinstance(context_duration, float):
            raise ValueError("context_duration must be a float")
        if not context_duration > 0:
            raise ValueError("context_duration must be greater than 0")

        if not isinstance(latent_step, float):
            raise ValueError("latent_step must be a float")
        if not latent_step > 0:
            raise ValueError("latent_step must be greater than 0")

        if abs(context_duration % latent_step) > 1e-10:
            logging.warning(
                f"context_duration ({context_duration}) is not a multiple of latent_step "
                f"({latent_step}). This is allowed but may change the number of latent tokens."
            )
