import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.batching import chain
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
from core.nn import Embedding
from core.utils.checkpoint import log_incompatible_keys
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import TARGET_WINDOW, ReadoutSpec


class POYO(BaseModel):
    """Transformer-based model for neural decoding from spike trains :cite:`poyo`.

    Adapted from `torch_brain
    <https://github.com/neuro-galaxy/torch_brain/blob/main/torch_brain/models/poyo.py>`_.

    1. Input tokens are constructed by combining unit embeddings, token type embeddings,
        and time embeddings for each spike in the sequence.
    2. The input sequence is compressed using cross-attention, where learnable latent
        tokens (each with an associated timestamp) attend to the input tokens.
    3. The compressed latent token representations undergo further refinement through
        multiple self-attention processing layers.
    4. Query tokens are constructed for the desired outputs by combining session
        embeddings, and output timestamps.
    5. These query tokens attend to the processed latent representations through
        cross-attention, producing outputs in the model's dimensional space (dim).
    6. Finally, a task-specific linear layer maps the outputs from the model dimension
        to the appropriate output dimension.

    Args:
        context_duration: Maximum duration of the input spike sequence (in seconds)
        modality_spec: A :class:`torch_brain.registry.ModalitySpec` specifying readout properties
        latent_step: Timestep of the latent grid (in seconds)
        num_latents_per_step: Number of unique latent tokens (repeated at every latent step)
        dim: Hidden dimension of the model
        depth: Number of processing layers (self-attentions in the latent space)
        dim_head: Dimension of each attention head
        cross_heads: Number of attention heads used in a cross-attention layer
        self_heads: Number of attention heads used in a self-attention layer
        ffn_dropout: Dropout rate for feed-forward networks
        lin_dropout: Dropout rate for linear layers
        atn_dropout: Dropout rate for attention
        emb_init_scale: Scale for embedding initialization
        t_min: Minimum timestamp resolution for rotary embeddings
        t_max: Maximum timestamp resolution for rotary embeddings

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
        finetune_enable: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents_per_step, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        # encoder layer
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))

        # process layers
        self.proc_layers = nn.ModuleList([])
        for _i in range(depth):
            self.proc_layers.append(
                nn.Sequential(
                    RotarySelfAttention(
                        dim=dim,
                        heads=self_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                        rotate_value=True,
                    ),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, dropout=ffn_dropout),
                    ),
                )
            )

        # decoder layer
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.dec_ffn = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))

        self.finetune_enable = finetune_enable

        self.logger = get_cli_logger()

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        self.context_duration = train_dataset.context_window
        self._validate_params(self.context_duration, self.latent_step)

        # The latent grid spans the context window and carries no per-sample
        # information, so it is built once here and tiled to the batch in forward.
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            0,
            self.context_duration,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )
        self.register_buffer("latent_index", torch.as_tensor(latent_index), persistent=False)
        self.register_buffer(
            "latent_timestamps", torch.as_tensor(latent_timestamps), persistent=False
        )

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
            # InfiniteVocabEmbedding requires vocabulary to be un-initalized if loading from a checkpoint later
            self.unit_emb.initialize_vocab(self.unit_ids)
            self.session_emb.initialize_vocab(self.session_ids)

    def configure_readout(self, readout_spec: ReadoutSpec):
        # Output projections
        self.readout_spec = readout_spec
        self.readout = nn.Linear(self.dim, readout_spec.dim)

    def load_ckpt(self, ckpt: dict):
        state_dict = ckpt["model_state_dict"]

        result = self.load_state_dict(state_dict, strict=False)
        self.logger.info("Loaded pretrained model from checkpoint")

        grafted = ()
        if "readout.weight" in result.missing_keys:
            task_id = self.readout_spec.id
            weight_key = f"readout.projections.{task_id}.weight"
            bias_key = f"readout.projections.{task_id}.bias"
            if weight_key in state_dict and bias_key in state_dict:
                self.readout.load_state_dict(
                    {
                        "weight": state_dict[weight_key],
                        "bias": state_dict[bias_key],
                    }
                )
                self.logger.info(f"Loaded readout weights from '{task_id}' task head")
                grafted = ("readout.",)
            else:
                self.logger.warning(
                    f"No '{task_id}' task head found in checkpoint, using random weights!"
                )
        log_incompatible_keys(result, self.logger, ignore=grafted)

        self.unit_emb.extend_vocab(self.unit_ids, exist_ok=True)
        self.session_emb.extend_vocab(self.session_ids, exist_ok=True)

        self.unit_emb.subset_vocab(self.unit_ids)
        self.session_emb.subset_vocab(self.session_ids)

        self.logger.info("Initialized unit and session vocabularies")
        self.logger.info(f"Number of units: {len(self.unit_emb.vocab) - 1}")
        self.logger.info(f"Number of sessions: {len(self.session_emb.vocab) - 1}")

    def input_fn(self, data: Data) -> dict:
        r"""Input function used to convert Data into model inputs for the POYO model.

        This input function can be called as a transform. If you are applying multiple
        transforms, make sure to apply this one last.

        This code runs on CPU. Do not access GPU tensors inside this function.
        """

        # context window
        start, end = 0, self.context_duration

        ### prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        spike_token_type_index = np.concatenate(
            [
                se_token_type_index,
                np.zeros_like(spike_unit_index),
            ]
        )
        spike_unit_index = np.concatenate([se_unit_index, spike_unit_index])
        spike_timestamps = np.concatenate([se_timestamps, spike_timestamps])

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_emb.tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        # create session index for output
        output_session_index = self.session_emb.tokenizer(data.session.id)

        batch = {
            "model_inputs": {
                # input sequence (keys/values for the encoder)
                "input_unit_index": chain(spike_unit_index),
                "input_timestamps": chain(spike_timestamps),
                "input_token_type": chain(spike_token_type_index),
                "input_seqlen": len(spike_unit_index),
            },
            "session_index": output_session_index,
        }

        return batch

    def forward(
        self,
        *,
        # input sequence
        input_unit_index: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_token_type: torch.Tensor,
        input_seqlen: torch.Tensor,
        # output queries
        output_session_index: torch.Tensor,
        output_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the POYO model.

        The model processes input spike sequences through its encoder-processor-decoder
        architecture to generate task-specific predictions.

        Args:
            input_unit_index: :math:`(S_{in},)` int, unit index per input token.
            input_timestamps: :math:`(S_{in},)` float, spike timestamp per input token.
            input_token_type: :math:`(S_{in},)` int, token type id per input token.
            input_seqlen: :math:`(B,)` int, number of input tokens per sample.
            output_session_index: :math:`(S_{out},)` int, session index per output query.
            output_timestamps: :math:`(S_{out},)` float, timestamp per output query, or ``None``.

        Returns:
            :math:`(S_{out}, D_{out})` float. Callers that need a batch axis reshape it
            against the target they already hold.
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

        # input
        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # latents, tiled from the fixed grid to this batch
        batch_size = input_seqlen.shape[0]
        latent_seqlen = torch.full_like(input_seqlen, self.latent_index.shape[0])
        latents = self.latent_emb(self.latent_index.repeat(batch_size))
        latent_timestamp_emb = self.rotary_emb(self.latent_timestamps.repeat(batch_size))

        # outputs
        if output_timestamps is not None:
            # timestep-level: timestamps are provided
            B, T = output_timestamps.shape
            output_seqlen = torch.full(
                (B,), T, dtype=torch.int32, device=output_timestamps.device
            )  # (B,)
            output_queries = self.session_emb(
                output_session_index.repeat_interleave(output_seqlen)
            )  # (B*T, D)
            output_timestamp_emb = self.rotary_emb(output_timestamps.flatten())  # (B*T, D)
        else:
            # sequence-level: timestamps not provided, so we use middle of the target window
            B = output_session_index.shape[0]
            ts = self.context_duration - TARGET_WINDOW / 2
            output_seqlen = torch.full(
                (B,), 1, dtype=torch.int32, device=output_session_index.device
            )  # (B,)
            output_queries = self.session_emb(output_session_index)  # (B, D)
            output_timestamps = torch.full(
                (B,), ts, dtype=torch.float32, device=output_session_index.device
            )  # (B,)
            output_timestamp_emb = self.rotary_emb(output_timestamps)  # (B, D)

        # encode
        latents = latents + self.enc_atn.forward_varlen(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            latent_seqlen,
            input_seqlen,
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn.forward_varlen(
                    latents,
                    latent_timestamp_emb,
                    latent_seqlen,
                )
            )
            latents = latents + self.dropout(self_ff(latents))

        # decode
        output_queries = output_queries + self.dec_atn.forward_varlen(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
            output_seqlen,
            latent_seqlen,
        )
        output_latents = output_queries + self.dec_ffn(output_queries)

        return self.readout(output_latents)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_float("model.latent_step", 0.0625, 0.125, step=0.0625)
        trial.suggest_int("num_latents_per_step_log2", 3, 5, step=1)
        trial.suggest_int("model.depth", 2, 10, step=2)
        trial.suggest_float("dropout", 0.0, 0.6, step=0.2)

        dim_log2 = trial.suggest_int("dim_log2", 5, 8, step=1)
        if dim_log2 > 5:  # only allow dim_head up min(64, dim)
            trial.suggest_categorical("model.dim_head", [32, 64])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 6, step=1)
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
        num_latents_per_step_log2 = tune_params.pop("num_latents_per_step_log2")
        num_latents_per_step = 2**num_latents_per_step_log2
        tune_params["model.num_latents_per_step"] = num_latents_per_step

        dim_log2 = tune_params.pop("dim_log2")
        dim = 2**dim_log2
        tune_params["model.dim"] = dim

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
        batch_size = 2**batch_size_log2
        tune_params["batch_size"] = batch_size

        max_units = tune_params.pop("ud_max_units")
        min_units = tune_params.pop("ud_min_units")
        mode_units = int(min_units + (max_units - min_units) / 3)
        tune_params["train_transforms.0.max_units"] = max_units
        tune_params["train_transforms.0.min_units"] = min_units
        tune_params["train_transforms.0.mode_units"] = mode_units

        return tune_params

    def _validate_params(self, context_duration, latent_step):
        r"""Ensure: context_duration, and latent_step are floating point numbers greater
        than zero. And context_duration is a multiple of latent_step.
        """

        if not isinstance(context_duration, float):
            raise ValueError("context_duration must be a float")
        if not context_duration > 0:
            raise ValueError("context_duration must be greater than 0")

        if not isinstance(latent_step, float):
            raise ValueError("latent_step must be a float")
        if not latent_step > 0:
            raise ValueError("latent_step must be greater than 0")

        # check if context_duration is a multiple of latent_step
        if abs(context_duration % latent_step) > 1e-10:
            self.logger.warning(
                f"context_duration ({context_duration}) is not a multiple of latent_step "
                f"({latent_step}). This is a simple warning, and this behavior is allowed."
            )
