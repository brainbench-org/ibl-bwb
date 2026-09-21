"""Writing a submission to disk."""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from ibl_bwb_eval._version import __version__

__api_ref__ = {
    "description": None,
    "sections": [
        {"title": "Submission format", "autosummary": ["PredictionsWriter", "version_of"]}
    ],
}


def version_of(metadata: dict[str, str]) -> str | None:
    """The ``ibl_bwb_eval_version`` a prediction file was written under.

    ``None`` for a file written before the stamp existed, which is not malformed: no
    release predates the stamp, so there is no version to attribute it to.

    Not yet gated on by any scorer: today every on-disk shape is read the same way
    regardless of version. When a future release changes that shape, branch the
    reader on ``version_of(metadata)`` and freeze a new fixture under
    ``src/tests/fixtures/legacy/v<N>_*`` (see ``make_v01_fixtures.py``) rather than
    updating the v01 ones in place.
    """
    return metadata.get("ibl_bwb_eval_version")


class PredictionsWriter:
    """Buffer tensors from an eval loop and save them as a ``.safetensors`` file.

    The output path is structured as::

        base_path / label / task / [path_fn(metadata)] / seed_{seed}.safetensors

    ``path_fn`` is a callable that receives the accumulated path and ``metadata``,
    and returns a middle segment of the path. This lets each task suite define its
    own sub-hierarchy (e.g. a recording id). ``metadata`` is also embedded in the file
    header alongside the common fields (``label``, ``task``, ``seed``) and
    ``ibl_bwb_eval_version``, :data:`ibl_bwb_eval.__version__` at write time, which
    traces a submission back to the writer that produced it. Read that one back with
    :func:`version_of`.

    Fields are recorded in one of two ways:

    - :meth:`add` appends a per-batch chunk; repeated calls for the same key are
      concatenated along dim 0 at save time.
    - :meth:`set` stores a single value once, for a field that is constant across
      batches or already aggregated.

    All values must be :class:`torch.Tensor`. They are detached and moved to CPU
    on ingest. Floating-point tensors are cast to ``float32`` by default; pass
    ``dtype=`` to :meth:`add`/:meth:`set` to use a lower-precision dtype instead
    (e.g. ``torch.float16`` for bulky prediction tensors), avoid this for fields
    used for exact alignment (e.g. timestamps), since low precision can collapse
    distinct values. :meth:`save` writes on rank 0 only and is a no-op if nothing
    was recorded.
    """

    @staticmethod
    def _enable_only(method):
        """Only allow method execution if write is enabled."""

        def wrapper(self, *args, **kwargs):
            if not getattr(self, "enable", False):
                return
            return method(self, *args, **kwargs)

        return wrapper

    def __init__(
        self,
        enable: bool,
        base_path: Path | str,
        task: str,
        seed: int,
        label: str | None = None,
        rank: int = 0,
        metadata: dict[str, Any] | None = None,
        path_fn: Callable[[Path, dict[str, Any]], Path] | None = None,
    ) -> None:
        if label is None and enable:
            raise ValueError("label is required for saving predictions")

        if metadata is None:
            metadata = {}

        self.enable = enable
        self.base_path = Path(base_path)
        self.task = task
        self.seed = seed
        self.rank = rank
        self.label = label
        self.metadata = metadata
        self.path_fn = path_fn

        self._streamed: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._constant: dict[str, torch.Tensor] = {}

    def _output_path(self) -> Path:
        base = self.base_path / self.label / self.task
        if self.path_fn is not None:
            base = self.path_fn(base, self.metadata)
        return base / f"seed_{self.seed}.safetensors"

    @_enable_only
    def add(self, dtype: torch.dtype | None = None, **fields: torch.Tensor) -> None:
        for key, value in fields.items():
            if key in self._constant:
                raise ValueError(f"{key!r} was already recorded via set()")
            self._streamed[key].append(self._require_tensor(key, value, dtype))

    @_enable_only
    def set(self, dtype: torch.dtype | None = None, **fields: torch.Tensor) -> None:
        for key, value in fields.items():
            if key in self._constant or key in self._streamed:
                raise ValueError(f"{key!r} was already recorded")
            self._constant[key] = self._require_tensor(key, value, dtype)

    def is_empty(self) -> bool:
        return not self._streamed and not self._constant

    @_enable_only
    def save(self, logger=None) -> None:
        if self.rank != 0 or self.is_empty():
            return
        out: dict[str, torch.Tensor] = {
            key: torch.cat(chunks, dim=0) for key, chunks in self._streamed.items()
        }
        out.update(self._constant)

        file_metadata = {
            "label": self.label,
            "task": self.task,
            "seed": str(self.seed),
            "ibl_bwb_eval_version": __version__,
            **{k: str(v) for k, v in self.metadata.items()},
        }
        path = self._output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(out, path, metadata=file_metadata)

        if logger is not None:
            n = len(out.get("predictions", next(iter(out.values()))))
            logger.info(f"Saved {n} predictions -> {path}")

    def _require_tensor(self, name: str, value: object, dtype: torch.dtype | None) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"PredictionsWriter field {name!r} expects torch.Tensor, got {type(value).__name__}"
            )
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype) if dtype is not None else tensor.float()
        return tensor
