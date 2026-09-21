import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.data import Data

from core.model import BaseModel
from ibl_bwb_eval.tasks import ReadoutSpec, TargetResolution


class CEBRA(BaseModel):
    """CEBRA-based neural decoding model on single-session :cite:`cebra`.

    .. note::
        Only single-session inference is supported.

    Notation: :math:`B` = batch size, :math:`T_{in}` = input time bins, :math:`N` = units,
    :math:`D_{emb}` = CEBRA embedding dimension (``output_dimension``),
    :math:`D_{out}` = task output dim, :math:`D` = MLP hidden dim
    (``num_hidden_units``, shared with the CEBRA encoder).

    1. :meth:`fit`: train the CEBRA encoder on the eval training split.
       The trainer then calls ``collect_embeddings`` to run
       ``cebra_model.transform`` on all splits and cache the results in
       ``model.embeddings``.
    2. :meth:`configure_readout`: build the two-layer MLP head and fix
       :math:`D_{out}`.
    3. :meth:`input_fn`: look up cached CEBRA embeddings to get
       :math:`(T_{in}, D_{emb})` or :math:`(1, D_{emb})` for sequence-level tasks.
    4. :meth:`forward`: pass embeddings through the MLP readout to produce
       :math:`(B, T_{out}, D_{out})`.

    Args:
        output_dimension: CEBRA embedding dimension :math:`D_{emb}`.
        bin_size: Width of each time bin in seconds.
        mode: Contrastive objective, ``"time"`` uses temporal labels only;
            ``"behavior"`` conditions on behavioral labels (and enables hybrid
            mode for continuous labels).
        max_iterations: Number of CEBRA training iterations.
        batch_size: Mini-batch size for CEBRA training.
        learning_rate: CEBRA optimiser learning rate.
        temperature: Contrastive loss temperature (used when
            ``temperature_mode="constant"``).
        model_architecture: CEBRA model architecture string (e.g.
            ``"offset10-model"``).
        verbose: Whether to print CEBRA training progress.
        num_hidden_units: MLP hidden dimension :math:`D`.
        time_offsets: Number of time offsets for the temporal objective.
        temperature_mode: How to set the contrastive temperature
            (``"auto"`` or ``"constant"``).
    """

    def __init__(
        self,
        output_dimension: int = 8,
        bin_size: float = 0.02,
        mode: str = "time",
        max_iterations: int = 1_000,
        batch_size: int = 512,
        learning_rate: float = 3e-4,
        temperature: float = 1.0,
        model_architecture: str = "offset10-model",
        verbose: bool = True,
        num_hidden_units: int = 1_024,
        time_offsets: int = 10,
        temperature_mode: str = "auto",
    ):
        try:
            import cebra as cebra_lib
        except ImportError as exc:
            raise ImportError(
                "The CEBRA baseline needs the optional cebra extra: "
                'uv pip install -e ".[train,cebra]".'
            ) from exc

        super().__init__()
        self.output_dimension = output_dimension
        self.bin_size = bin_size
        self.mode = mode
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.model_architecture = model_architecture
        self._cebra_lib = cebra_lib
        self.cebra_model = None
        self.verbose = verbose
        self.num_hidden_units = num_hidden_units
        self.time_offsets = time_offsets
        self.temperature_mode = temperature_mode

    def fit(
        self,
        session: np.ndarray,
        behavior_labels: np.ndarray | None = None,
    ):
        """Train the CEBRA encoder on a single session.

        In ``"time"`` mode the model uses a temporal contrastive objective
        with synthetic time labels. In ``"behavior"`` mode it conditions on
        ``behavior_labels`` and additionally enables the hybrid (time +
        behavior) objective for continuous labels. After this call the trainer
        runs ``collect_embeddings`` to cache ``cebra_model.transform`` outputs
        in ``model.embeddings`` before :meth:`configure_readout` is invoked.

        Args:
            session: :math:`(T_{in}, N)` float spike-rate array for the session.
            behavior_labels: :math:`(T_{in}, ...)` behavioral labels required
                when ``mode="behavior"``. Pass continuous floats to enable hybrid
                mode; integer labels disable it.
        """
        # hybrid=True combines time + behavior contrastive objectives (time objective is
        # handled internally by CEBRA when conditional="time_delta"; no extra labels needed).
        # hybrid requires continuous (float) labels and is not supported for discrete labels.
        # hybrid is irrelevant in time mode since there are no behavior labels.
        is_discrete = behavior_labels is not None and behavior_labels.dtype.kind == "i"
        hybrid = self.mode == "behavior" and not is_discrete
        self.cebra_model = self._cebra_lib.CEBRA(
            model_architecture=self.model_architecture,
            output_dimension=self.output_dimension,
            max_iterations=self.max_iterations,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            temperature=self.temperature,
            num_hidden_units=self.num_hidden_units,
            time_offsets=self.time_offsets,
            conditional="time" if self.mode == "time" else "time_delta",
            device="cuda_if_available",
            temperature_mode=self.temperature_mode,
            hybrid=hybrid,
        )

        if self.mode == "behavior":
            if behavior_labels is None:
                raise ValueError("behavior_labels are required when mode='behavior'.")
            self.cebra_model.fit(session, behavior_labels)
        else:
            time_labels = np.arange(session.shape[0], dtype=np.float32).reshape(-1, 1)
            self.cebra_model.fit(session, time_labels)

    def configure_readout(self, readout_spec: ReadoutSpec):
        r"""Fix :math:`D_{out}` and build the two-layer MLP readout head.

        The output shape depends on the target resolution:

        - **Sequence-level** (:math:`T_{out}=1`): :meth:`input_fn` mean-pools
          embeddings to :math:`(1, D_{emb})` before the MLP.
        - **Timestep-level**: :meth:`input_fn` returns :math:`(T_{in}, D_{emb})`;
          the MLP is applied at each step to yield
          :math:`(B, T_{in}, D_{out})`.

        The readout is a two-layer MLP:
        :math:`D_{emb} \to D \xrightarrow{\text{GELU}} \text{Dropout}(0.2) \to D_{out}`.

        Args:
            readout_spec: Task specification carrying :math:`D_{out}` and the target resolution.
        """
        self.readout_spec = readout_spec
        # Two-layer MLP readout; hidden size is num_hidden_units (shared with the CEBRA encoder).
        hidden_dim = self.num_hidden_units
        self.readout = nn.Sequential(
            nn.Linear(self.output_dimension, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, readout_spec.dim),
        )

    def input_fn(self, data: Data) -> dict[str, torch.Tensor]:
        """Look up pre-computed CEBRA embeddings for a single trial.

        Args:
            data: Trial data; must expose the interval key specified in
                :meth:`configure_readout` and a single trial index into
                ``self.embeddings``.

        Returns:
            Dict with:

            - ``model_inputs.embeddings``: :math:`(T_{in}, D_{emb})` float
              tensor for timestep-level tasks, or :math:`(1, D_{emb})` after
              mean-pooling for sequence-level tasks.
        """
        task_aligned_trial_idx = data.get_nested_attribute(
            self.readout_spec.interval_key
        ).trial_index
        assert len(task_aligned_trial_idx) == 1, (
            f"Expected 1 trial, got {len(task_aligned_trial_idx)}"
        )
        trial_idx = task_aligned_trial_idx[0]
        embeddings_np = self.embeddings[trial_idx]
        embeddings = torch.from_numpy(embeddings_np).float()  # (T, D)

        if self.readout_spec.target_resolution == TargetResolution.SEQUENCE:
            embeddings = embeddings.mean(dim=0, keepdim=True)  # (1, D)

        return {"model_inputs": {"embeddings": embeddings}}

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map CEBRA embeddings to task predictions via the MLP readout.

        Args:
            embeddings: :math:`(B, T_{out}, D_{emb})` float tensor of CEBRA
                embeddings, where :math:`T_{out}=1` for sequence-level tasks
                or :math:`T_{out}=T_{in}` for timestep-level tasks.

        Returns:
            :math:`(B, T_{out}, D_{out})` task predictions.
        """
        return self.readout(embeddings)  # (B, T_out, D_out)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_int("model.output_dimension_log2", 3, 8)
        trial.suggest_int("model.time_offsets", 1, 5)
        trial.suggest_float("model.learning_rate", 1e-4, 1e-3, log=True)
        trial.suggest_float("model.temperature", 0.5, 2.0, step=0.1)
        trial.suggest_categorical("model.mode", ["time", "behavior"])
        trial.suggest_int("model.max_iterations", 5_000, 9_000, step=2_000)

        # training
        trial.suggest_int("num_epochs", 100, 500, step=100)
        trial.suggest_int("batch_size_log2", 4, 6)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
        trial.suggest_float("pct_start", 0.1, 0.9)
        trial.suggest_float("div_factor", 1, 5, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        tune_params["model.output_dimension"] = 2 ** tune_params["model.output_dimension_log2"]
        tune_params["model.num_hidden_units"] = tune_params["model.output_dimension"] * 2
        tune_params["batch_size"] = 2 ** tune_params["batch_size_log2"]

        tune_params.pop("model.output_dimension_log2")
        tune_params.pop("batch_size_log2")

        return tune_params
