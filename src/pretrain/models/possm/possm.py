import contextlib
from typing import ClassVar

import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.batching import chain, pad, pad2d, track_mask2d
from torch_brain.data import Data
from torch_brain.models.poyo import FeedForward, create_linspace_latent_tokens
from torch_brain.nn import (
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding, MultitaskReadout
from core.utils.checkpoint import log_incompatible_keys
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import (
    ReadoutSpec,
    TargetResolution,
    TS1ReadoutSpec,
    get_ts1_supported_tasks,
)


class GRU(nn.Module):
    """Thin wrapper around nn.GRU that discards the hidden state.

    Bidirectional by default; ``output_dim`` reports the per-step hidden size
    seen downstream (``hidden_size * 2`` when bidirectional).
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0, bidirectional=True):
        super().__init__()
        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        out, _ = self.net(x)
        return out


class POSSM(BaseModel):
    """POSSM (POYO + State Space Model) for the IBL benchmark :cite:`possm`.

    Combines a perceiver-based per-bin encoder with a sequential backbone for
    processing neural spike data in temporal bins. Uses
    :class:`core.nn.MultitaskReadout` for output projection so that the
    same model can be used for single-task eval or multi-task pretraining.

    This differs from the paper in one respect: only the GRU backbone is
    implemented here, not the S4D or Mamba variants.

    Architecture:
        1. Input spikes are binned into temporal intervals.
        2. Per-bin perceiver encoder compresses spikes into latent tokens via
           cross-attention, optionally refined by self-attention processing layers.
        3. A sequential backbone processes the per-bin latent representations.
        4. Decoder cross-attention with bounded causal context produces output
           query embeddings.
        5. ``MultitaskReadout`` projects to task-specific output dimensions.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.

    Args:
        backbone: Sequential backbone type; ``"gru"`` is the only one implemented.
        bin_width: Width of each temporal bin (seconds).
        bin_step: Step between consecutive bins (seconds).
        num_latents: Number of latent tokens per bin.
        dim: Hidden dimension of all embeddings.
        depth: Number of self-attention processing layers in the per-bin encoder.
        dim_head: Dimension of each attention head.
        cross_heads: Number of cross-attention heads.
        self_heads: Number of self-attention heads.
        ffn_dropout: Dropout rate for feed-forward networks.
        lin_dropout: Dropout rate for linear layers.
        atn_dropout: Dropout rate for attention.
        rnn_dim: Backbone hidden dimension.
        num_rnn_layers: Number of backbone layers.
        rnn_dropout: Backbone dropout rate.
        bidirectional: If True (default), use a bidirectional GRU. Note this
            makes the per-bin representation non-causal, adjust
            ``output_ca_ctx_lim`` if real-time decoding is required.
        output_ca_ctx_lim: Number of past bins visible to decoder cross-attention.
        emb_init_scale: Embedding initialization scale.
        t_min: Min time period for encoder rotary embeddings.
        t_max: Max time period for encoder rotary embeddings.
        dec_t_min: Min time period for decoder rotary embeddings.
        dec_t_max: Max time period for decoder rotary embeddings.
    """

    BACKBONES: ClassVar[dict] = {"gru": GRU}

    def __init__(
        self,
        *,
        backbone: str = "gru",
        bin_width: float = 0.05,
        bin_step: float = 0.05,
        num_latents: int = 1,
        dim: int = 256,
        depth: int = 0,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        rnn_dim: int = 512,
        num_rnn_layers: int = 4,
        rnn_dropout: float = 0.2,
        bidirectional: bool = True,
        output_ca_ctx_lim: int = 3,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 10.0,
        dec_t_min: float = 0.01,
        dec_t_max: float = 10.0,
        task_vocab: list[str] | None = None,
        finetune_enable: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.bin_width = bin_width
        self.bin_step = bin_step
        self.num_latents = num_latents
        self.output_ca_ctx_lim = output_ca_ctx_lim

        # Stable task vocabulary: index 0 reserved for padding, 1..N for tasks.
        # Reserving the full benchmark vocab up front keeps task_emb rows
        # aligned across pretrain (multi-task) and finetune (single-task).
        self.task_vocab = list(task_vocab) if task_vocab is not None else get_ts1_supported_tasks()
        self.task_name_to_index = {task_name: i + 1 for i, task_name in enumerate(self.task_vocab)}

        # Precompute latent tokens (constant per bin)
        latent_step = bin_width / num_latents
        self._latent_index, self._latent_timestamps = create_linspace_latent_tokens(
            0,
            bin_width,
            step=latent_step,
            num_latents_per_step=num_latents,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        # Task embedding: index 0 = padding, 1..N = tasks (stable across regimes)
        self.task_emb = Embedding(len(self.task_vocab) + 1, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )
        self.dec_rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=dec_t_min,
            t_max=dec_t_max,
        )

        # encoder layer
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            use_xformers=False,
        )
        self.enc_ffn = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))

        # processor layers
        self.proc_layers = nn.ModuleList()
        for _ in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                            use_xformers=False,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        # sequential backbone
        if backbone not in self.BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose from: {list(self.BACKBONES.keys())}"
            )
        backbone_kwargs = {
            "input_size": num_latents * dim,
            "hidden_size": rnn_dim,
            "num_layers": num_rnn_layers,
            "dropout": rnn_dropout,
        }
        if backbone == "gru":
            backbone_kwargs["bidirectional"] = bidirectional
        self.backbone = self.BACKBONES[backbone](**backbone_kwargs)
        backbone_output_dim = getattr(self.backbone, "output_dim", rnn_dim)

        # decoder layer
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=backbone_output_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            use_xformers=False,
        )
        self.dec_ffn = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))

        # Set later by configure_readout / configure_multitask_readout / link_datasets
        self.readout_specs: dict[str, TS1ReadoutSpec] = {}
        self.readout: MultitaskReadout | None = None
        self.readout_spec: TS1ReadoutSpec | None = None
        self.context_duration: float | None = None

        self.finetune_enable = finetune_enable

        self.logger = get_cli_logger()

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        self.context_duration = train_dataset.context_window
        self._validate_params()

        # initialize vocab
        self.unit_ids = set(train_dataset.get_unit_ids()) | set(val_dataset.get_unit_ids())
        if test_dataset is not None:
            self.unit_ids |= set(test_dataset.get_unit_ids())
        self.unit_ids = list(self.unit_ids)

        self.session_ids = set(train_dataset.get_session_ids()) | set(val_dataset.get_session_ids())
        if test_dataset is not None:
            self.session_ids |= set(test_dataset.get_session_ids())
        self.session_ids = list(self.session_ids)

        if not self.finetune_enable:
            # InfiniteVocabEmbedding requires vocabulary to be un-initialized
            # if loading from a checkpoint later
            self.unit_emb.initialize_vocab(self.unit_ids)
            self.session_emb.initialize_vocab(self.session_ids)

    def configure_multitask_readout(self, readout_specs: dict[str, TS1ReadoutSpec]):
        """Configure :class:`MultitaskReadout` over a set of tasks.

        Each task in ``readout_specs`` must be present in ``self.task_vocab``
        so that its ``task_emb`` row and ``MultitaskReadout`` head id are
        stable across regimes (multi-task pretrain -> single-task finetune).

        Args:
            readout_specs: mapping ``task_name -> TS1ReadoutSpec``.
        """
        unknown = set(readout_specs) - set(self.task_name_to_index)
        if unknown:
            raise ValueError(
                f"Unknown tasks for POSSM: {sorted(unknown)}. Add them to task_vocab on the model."
            )

        self.readout_specs = dict(readout_specs)
        # Singular spec: convenience for single-task eval / loss code paths.
        self.readout_spec = (
            next(iter(self.readout_specs.values())) if len(self.readout_specs) == 1 else None
        )

        self.readout = MultitaskReadout(
            dim=self.dim,
            readout_specs=self.readout_specs,
            task_index=self.task_name_to_index,
        )

    def configure_readout(
        self,
        readout_spec: ReadoutSpec | list[ReadoutSpec] | dict[str, ReadoutSpec],
    ):
        """Convenience wrapper around :meth:`configure_multitask_readout`.

        Accepts a single :class:`ReadoutSpec` (single-task), a list of them, or a
        ``{task_name: ReadoutSpec}`` dict. The container types are tested first, so the
        spec itself is matched structurally and never by its suite's class.
        """
        if isinstance(readout_spec, list):
            specs = {spec.id: spec for spec in readout_spec}
        elif isinstance(readout_spec, dict):
            specs = readout_spec
        elif hasattr(readout_spec, "id"):
            specs = {readout_spec.id: readout_spec}
        else:
            raise TypeError(f"Unsupported readout_spec type: {type(readout_spec).__name__}")
        self.configure_multitask_readout(specs)

    def load_ckpt(self, ckpt: dict):
        state_dict = ckpt["model_state_dict"]

        result = self.load_state_dict(state_dict, strict=False)
        self.logger.info("Loaded pretrained model from checkpoint")
        # A one-task model builds the same `readout.projections.<task>` key the multitask
        # checkpoint stores, so the head above loaded with it; the other tasks' heads are
        # the unused keys just logged.
        log_incompatible_keys(result, self.logger)

        self.unit_emb.extend_vocab(self.unit_ids, exist_ok=True)
        self.session_emb.extend_vocab(self.session_ids, exist_ok=True)

        self.unit_emb.subset_vocab(self.unit_ids)
        self.session_emb.subset_vocab(self.session_ids)

        self.logger.info("Initialized unit and session vocabularies")
        self.logger.info(f"Number of units: {len(self.unit_emb.vocab) - 1}")
        self.logger.info(f"Number of sessions: {len(self.session_emb.vocab) - 1}")

    def input_fn(self, data: Data) -> dict:
        r"""Input function used to convert Data into model inputs for the POSSM model.

        This input function can be called as a transform. If you are applying multiple
        transforms, make sure to apply this one last.

        This code runs on CPU. Do not access GPU tensors inside this function.

        Prepares per-bin spike inputs and multitask output queries for all
        configured readout specs. Each output query carries a decoder index
        identifying which readout head it belongs to.
        """

        end = self.context_duration

        # Compute bin structure
        end_rounded = round(end, 5)
        n_intervals = int(np.floor(np.round((end_rounded - self.bin_width) / self.bin_step, 5)) + 1)
        interval_end = np.round(np.arange(self.bin_width, end_rounded + 1e-6, self.bin_step), 5)
        interval_start = np.round(interval_end - self.bin_width, 5)
        interval_end[-1] = end_rounded + 1e-6

        ### prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # unit_index is relative to the recording, so we want to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_emb.tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        # Bin spikes
        (
            spike_unit_index_2d,
            spike_timestamps_2d,
            spike_type_2d,
            input_mask_2d,
        ) = self._bin_spikes(
            spike_unit_index,
            spike_timestamps,
            n_intervals,
            interval_start,
            interval_end,
        )

        # Prepare multitask output queries (and gather targets per task)
        all_timestamps = []
        all_decoder_index = []
        target_values: dict[str, np.ndarray] = {}
        target_masks: dict[str, np.ndarray] = {}
        for task_name, spec in self.readout_specs.items():
            target = self._extract_task_target(data, spec, end)
            if target is None:
                continue
            ts = target["timestamps"]
            all_timestamps.append(ts)
            all_decoder_index.append(
                np.full(len(ts), self.task_name_to_index[task_name], dtype=np.int64)
            )
            target_values[task_name] = np.asarray(target["values"], dtype=np.float32)
            if "mask" in target:
                target_masks[task_name] = np.asarray(target["mask"], dtype=bool)

        if all_timestamps:
            output_timestamps = np.concatenate(all_timestamps)
            output_decoder_index = np.concatenate(all_decoder_index)
        else:
            output_timestamps = np.empty((0,), dtype=np.float32)
            output_decoder_index = np.empty((0,), dtype=np.int64)

        # Compute bin index for each output query (1-based)
        output_bin_index = np.searchsorted(interval_end, output_timestamps, side="right") + 1

        # Session index repeated for all output queries
        session_index = self.session_emb.tokenizer(data.session.id)
        output_session_index = np.full(len(output_timestamps), session_index, dtype=np.int64)

        batch = {
            "model_inputs": {
                # binned input sequence
                "spike_unit_index": pad2d(spike_unit_index_2d),
                "spike_timestamps": pad2d(spike_timestamps_2d),
                "spike_type": pad2d(spike_type_2d),
                "input_mask": track_mask2d(input_mask_2d),
                "n_intervals": n_intervals,
                # latent sequence (constant per bin)
                "latent_index": self._latent_index,
                "latent_timestamps": self._latent_timestamps,
                # output queries (multitask)
                "output_timestamps": pad(output_timestamps),
                "output_decoder_index": pad(output_decoder_index),
                "output_bin_index": pad(output_bin_index),
                "output_session_index": pad(output_session_index),
            },
            "session_index": session_index,
            # multi-task targets (used by POSSMMultitaskPretrain; ignored by
            # eval / single-task trainers which read targets via the dataset)
            "target_values": chain(target_values, allow_missing_keys=True),
            "target_masks": chain(target_masks, allow_missing_keys=True),
        }

        return batch

    def forward(
        self,
        *,
        # binned input sequence
        spike_unit_index,  # (B, n_intervals, n_in)
        spike_timestamps,  # (B, n_intervals, n_in)
        spike_type,  # (B, n_intervals, n_in)
        input_mask,  # (B, n_intervals, n_in)
        n_intervals,  # (B,) or int
        # latent sequence
        latent_index,  # (B, n_latent)
        latent_timestamps,  # (B, n_latent)
        # output queries
        output_timestamps,  # (B, n_out)
        output_decoder_index,  # (B, n_out)
        output_bin_index,  # (B, n_out)
        output_session_index,  # (B, n_out)
        # output options
        unflatten_output: bool = True,
        return_dict: bool = False,
    ):
        """Forward pass of the POSSM model.

        Args:
            spike_unit_index: per-bin spike unit indices, (B, n_intervals, n_in)
            spike_timestamps: per-bin spike timestamps (bin-relative)
            spike_type: per-bin spike token types
            input_mask: per-bin attention mask (True = valid token)
            n_intervals: number of bins per sample (or int when constant)
            latent_index: latent token indices, (B, n_latent)
            latent_timestamps: latent token timestamps (bin-relative)
            output_timestamps: output query timestamps, (B, n_out)
            output_decoder_index: per-query readout id, (B, n_out)
            output_bin_index: per-query 1-based bin index, (B, n_out)
            output_session_index: per-query session index, (B, n_out)
            unflatten_output: if True and the (single) eval task is timestep-level,
                returns ``(B, n_out, dim_out)``; sequence-level tasks return
                ``(B, dim_out)``. Used for compatibility with ``TS1EvalTrainer``.
            return_dict: if True, returns the raw multitask dict
                ``{task_id: tensor}`` from :class:`MultitaskReadout`. Used by
                multi-task training.
        """
        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary has not been initialized, please use "
                "`model.unit_emb.initialize_vocab(unit_ids)`"
            )

        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        B = spike_unit_index.shape[0]
        num_intervals = spike_unit_index.shape[1]
        N_out = output_timestamps.shape[1]
        latents = self.latent_emb(latent_index)

        # ---- Encoder: batch across all bins ----
        spike_unit_flat = spike_unit_index.reshape(B * num_intervals, -1)
        spike_ts_flat = spike_timestamps.reshape(B * num_intervals, -1)
        spike_type_flat = spike_type.reshape(B * num_intervals, -1)
        input_mask_flat = input_mask.reshape(B * num_intervals, -1)

        inputs = self.unit_emb(spike_unit_flat) + self.token_type_emb(spike_type_flat)
        input_ts_emb = self.rotary_emb(spike_ts_flat)
        latent_ts_emb = self.rotary_emb(latent_timestamps[:1].expand(B * num_intervals, -1))

        latents_rep = latents.repeat(num_intervals, 1, 1)

        # encoder cross-attention
        ca_output = latents_rep + self.enc_atn(
            latents_rep,
            inputs,
            latent_ts_emb,
            input_ts_emb,
            input_mask_flat,
        )

        # processor layers
        for self_attn, self_ff in self.proc_layers:
            ca_output = ca_output + self.dropout(self_attn(ca_output, latent_ts_emb))
            ca_output = ca_output + self.dropout(self_ff(ca_output))

        # encoder FFN
        ca_output = ca_output + self.enc_ffn(ca_output)

        # flatten latents -> (B, n_intervals, num_latents*dim)
        ca_output = torch.flatten(ca_output, start_dim=1)
        outputs = ca_output.reshape(B, num_intervals, -1)

        # ---- Sequential backbone ----
        outputs = self.backbone(outputs)  # (B, n_intervals, rnn_dim)

        # ---- Decoder ----
        # Build output queries from task + session embeddings
        output_queries = self.task_emb(output_decoder_index) + self.session_emb(
            output_session_index
        )

        # Gather per-query context from backbone outputs
        ctx_lim = self.output_ca_ctx_lim
        rnn_dim = outputs.shape[-1]
        device = outputs.device

        bin_idx = output_bin_index.clamp(min=1)  # (B, N_out), 1-based
        offsets = torch.arange(-ctx_lim, 0, device=device)
        ctx_pos = bin_idx.unsqueeze(-1) + offsets  # (B, N_out, ctx_lim)
        valid = ctx_pos >= 0
        ctx_pos = ctx_pos.clamp(min=0)

        ctx = outputs.gather(1, ctx_pos.reshape(B, -1, 1).expand(-1, -1, rnn_dim)).reshape(
            B, N_out, ctx_lim, rnn_dim
        )

        # Relative timestamps for decoder rotary embeddings
        ctx_rel_ts = (
            torch.arange(-(ctx_lim - 1), 1, dtype=torch.float32, device=device) * self.bin_width
        )
        query_rel_ts = output_timestamps - bin_idx.float() * self.bin_width

        # Decoder cross-attention (fold each query into its own batch element)
        BN = B * N_out
        q_emb = self.dec_rotary_emb(query_rel_ts.reshape(BN, 1))
        c_emb = self.dec_rotary_emb(ctx_rel_ts.unsqueeze(0).expand(BN, -1))

        latent = output_queries.reshape(BN, 1, -1) + self.dec_atn(
            output_queries.reshape(BN, 1, -1),
            ctx.reshape(BN, ctx_lim, -1),
            q_emb,
            c_emb,
            context_mask=valid.reshape(BN, ctx_lim),
        )
        output_latents = (latent + self.dec_ffn(latent)).reshape(B, N_out, -1)

        # ---- MultitaskReadout ----
        readout_output = self.readout(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
        )

        if return_dict:
            return readout_output

        # Single-task tensor return path for TS1EvalTrainer compatibility.
        if self.readout_spec is None:
            raise ValueError(
                "return_dict=False requires exactly one configured readout. "
                "Use return_dict=True for multi-task forward."
            )
        task_key = self.readout_spec.id
        task_output = readout_output[task_key]  # (total_queries, dim_out)
        output = task_output.view(B, N_out, -1)

        if unflatten_output and self.readout_spec.target_resolution == TargetResolution.TIMESTEP:
            pass  # (B, N_out, D) already correct
        else:
            output = output.squeeze(1)  # (B, 1, D) -> (B, D)

        return output

    def compute_bin_index(self, timestamps: torch.Tensor) -> torch.Tensor:
        r"""Map query timestamps to 1-based bin indices using the model's bin
        structure (matches the boundary convention in :meth:`input_fn`).

        Args:
            timestamps: float tensor of any shape; values in [0, context_duration].

        Returns:
            Long tensor of the same shape with 1-based bin indices.
        """
        end_rounded = round(self.context_duration, 5)
        interval_end = torch.arange(
            self.bin_width,
            end_rounded + 1e-6,
            self.bin_step,
            device=timestamps.device,
            dtype=torch.float32,
        )
        interval_end = torch.round(interval_end * 1e5) / 1e5
        interval_end[-1] = end_rounded + 1e-6
        return torch.searchsorted(interval_end, timestamps.to(torch.float32), right=True) + 1

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model - binning
        trial.suggest_categorical("model.bin_width", [0.025, 0.05, 0.1])
        trial.suggest_categorical("model.bin_step", [0.025, 0.05, 0.1])
        trial.suggest_int("num_latents_log2", 0, 2, step=1)  # 1, 2, 4

        # model - perceiver
        trial.suggest_int("model.depth", 0, 2, step=1)
        trial.suggest_float("dropout", 0.0, 0.6, step=0.2)

        dim_log2 = trial.suggest_int("dim_log2", 6, 8, step=1)  # 64, 128, 256
        if dim_log2 > 5:  # only allow dim_head up to min(64, dim)
            trial.suggest_categorical("model.dim_head", [32, 64])

        # model - backbone
        trial.suggest_int("rnn_dim_log2", 7, 9, step=1)  # 128, 256, 512
        trial.suggest_int("model.num_rnn_layers", 1, 4, step=1)
        trial.suggest_float("model.rnn_dropout", 0.0, 0.4, step=0.2)
        trial.suggest_categorical("model.output_ca_ctx_lim", [1, 3, 5])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 7, step=1)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
        trial.suggest_float("pct_start", 0.1, 0.9)
        trial.suggest_float("div_factor", 1, 5, log=True)

        # unit dropout
        trial.suggest_int("ud_min_units", 100, 400, step=100)
        trial.suggest_int("ud_max_units", 600, 1_200, step=200)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        num_latents_log2 = tune_params.pop("num_latents_log2")
        tune_params["model.num_latents"] = 2**num_latents_log2

        dim_log2 = tune_params.pop("dim_log2")
        dim = 2**dim_log2
        tune_params["model.dim"] = dim

        rnn_dim_log2 = tune_params.pop("rnn_dim_log2")
        tune_params["model.rnn_dim"] = 2**rnn_dim_log2

        dropout = tune_params.pop("dropout")
        tune_params["model.ffn_dropout"] = dropout
        tune_params["model.lin_dropout"] = dropout
        tune_params["model.atn_dropout"] = dropout

        if "model.dim_head" not in tune_params:
            tune_params["model.dim_head"] = tune_params["model.dim"]
        num_heads = tune_params["model.dim"] // tune_params["model.dim_head"]
        tune_params["model.cross_heads"] = num_heads
        tune_params["model.self_heads"] = num_heads

        batch_size_log2 = tune_params.pop("batch_size_log2")
        tune_params["batch_size"] = 2**batch_size_log2

        max_units = tune_params.pop("ud_max_units")
        min_units = tune_params.pop("ud_min_units")
        mode_units = int(min_units + (max_units - min_units) / 3)
        tune_params["train_transforms.0.max_units"] = max_units
        tune_params["train_transforms.0.min_units"] = min_units
        tune_params["train_transforms.0.mode_units"] = mode_units

        return tune_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_params(self):
        if not isinstance(self.context_duration, float):
            raise ValueError("context_duration must be a float")
        if not self.context_duration > 0:
            raise ValueError("context_duration must be greater than 0")

        if not isinstance(self.bin_width, float):
            raise ValueError("bin_width must be a float")
        if not self.bin_width > 0:
            raise ValueError("bin_width must be greater than 0")

        if self.bin_width > self.context_duration:
            raise ValueError(
                f"bin_width ({self.bin_width}) must not exceed "
                f"context_duration ({self.context_duration})"
            )

    @staticmethod
    def _extract_task_target(
        data: Data, spec: TS1ReadoutSpec, context_end: float
    ) -> dict[str, np.ndarray] | None:
        """Extract values, timestamps, and mask for one task from a Data slice.

        Returns ``None`` if the task is not present in this sample (e.g.
        recording missing the task's value/timestamp attributes).
        """
        try:
            values = data.get_nested_attribute(spec.value_key)
        except AttributeError:
            return None
        if values is None:
            return None

        values = np.asarray(values)
        if values.size == 0:
            return None
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values.reshape(-1, 1)

        if spec.target_resolution == TargetResolution.TIMESTEP:
            if spec.timestamp_key is None:
                return None
            try:
                timestamps = np.asarray(data.get_nested_attribute(spec.timestamp_key)).astype(
                    np.float32
                )
            except AttributeError:
                return None
        else:
            # Sequence-level: single query at end of context window
            timestamps = np.array([context_end], dtype=np.float32)

        target: dict[str, np.ndarray] = {
            "values": values.astype(np.float32, copy=False),
            "timestamps": timestamps,
        }
        if spec.mask_key is not None:
            with contextlib.suppress(AttributeError):
                target["mask"] = np.asarray(data.get_nested_attribute(spec.mask_key))
        return target

    @staticmethod
    def _bin_spikes(spike_unit_index, spike_timestamps, n_intervals, interval_start, interval_end):
        """Assign spikes to temporal bins using binary search and vectorized scatter.

        Returns ``[n_intervals, max_spikes]`` tensors for ``unit_index``,
        ``timestamps`` (bin-relative), ``type`` (zeros), and attention mask.
        """
        first_bin = np.searchsorted(interval_end, spike_timestamps, side="right")
        last_bin = np.searchsorted(interval_start, spike_timestamps, side="right") - 1
        bins_per_spike = np.maximum(last_bin - first_bin + 1, 0)
        total = int(bins_per_spike.sum())

        if total == 0:
            return (
                torch.zeros(n_intervals, 1, dtype=torch.long),
                torch.zeros(n_intervals, 1),
                torch.zeros(n_intervals, 1, dtype=torch.long),
                torch.ones(n_intervals, 1, dtype=torch.bool),
            )

        spike_sel = np.repeat(np.arange(len(spike_timestamps)), bins_per_spike)
        cum = np.cumsum(bins_per_spike)
        bin_idx = np.repeat(first_bin, bins_per_spike) + (
            np.arange(total) - np.repeat(cum - bins_per_spike, bins_per_spike)
        )

        order = np.argsort(bin_idx, kind="stable")
        bin_sorted = bin_idx[order]
        counts = np.bincount(bin_idx, minlength=n_intervals)
        cum_counts = np.empty(n_intervals + 1, dtype=np.intp)
        cum_counts[0] = 0
        np.cumsum(counts, out=cum_counts[1:])
        pos = np.arange(total) - cum_counts[bin_sorted]

        max_spikes = max(int(counts.max()), 1)
        b = torch.from_numpy(bin_sorted.astype(np.int64))
        p = torch.from_numpy(pos.astype(np.int64))

        unit_2d = torch.zeros(n_intervals, max_spikes, dtype=torch.long)
        ts_2d = torch.zeros(n_intervals, max_spikes)
        type_2d = torch.zeros(n_intervals, max_spikes, dtype=torch.long)
        mask_2d = torch.zeros(n_intervals, max_spikes, dtype=torch.bool)

        unit_2d[b, p] = torch.from_numpy(spike_unit_index[spike_sel[order]].astype(np.int64))
        ts_2d[b, p] = torch.from_numpy(
            (spike_timestamps[spike_sel[order]] - interval_start[bin_sorted]).astype(np.float32)
        )
        mask_2d[b, p] = True
        mask_2d[counts == 0, 0] = True  # blank token for empty bins

        return unit_2d, ts_2d, type_2d, mask_2d
