from collections.abc import Mapping

import torch
import torch.nn as nn

from ibl_bwb_eval.tasks import ReadoutSpec


class MultitaskReadout(nn.Module):
    """Linear readout with one head per task, routed by integer index.

    Args:
        dim: Dimensionality of the incoming output embeddings.
        readout_specs: Task name to :class:`ibl_bwb_eval.tasks.ReadoutSpec`. Keys name the
            projections and the entries of the returned dicts.
        task_index: Task name to the integer written into ``output_readout_index``.
            Routing uses these, not ``ReadoutSpec.id``, which is a string.
    """

    def __init__(
        self,
        dim: int,
        readout_specs: Mapping[str, ReadoutSpec],
        task_index: Mapping[str, int],
    ):
        super().__init__()

        missing = set(readout_specs) - set(task_index)
        if missing:
            raise ValueError(f"No routing index for readouts: {sorted(missing)}")

        self.readout_specs = readout_specs

        # ``projections`` is load-bearing: checkpoints store these as
        # ``readout.projections.<task_name>.{weight,bias}`` and single-task models
        # graft a head out of them by that key.
        self.projections = nn.ModuleDict({})
        self._readout_id_to_name = {}
        for readout_name, readout_spec in self.readout_specs.items():
            self.projections[readout_name] = nn.Linear(dim, readout_spec.dim)
            self._readout_id_to_name[task_index[readout_name]] = readout_name

    def forward(
        self,
        output_embs: torch.Tensor,
        output_readout_index: torch.Tensor,
        unpack_output: bool = False,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Project padded output embeddings through their per-task heads.

        Args:
            output_embs: Output embeddings, ``(batch, n_out, dim)``.
            output_readout_index: Routing index per output token, ``(batch, n_out)``.
            unpack_output: Return one dict per batch sample instead of one dict with
                every sample's queries concatenated.

        Returns:
            ``{task_name: (total_queries, n_channels)}``, or a list of such dicts with
            ``(n_queries, n_channels)`` entries when ``unpack_output`` is set.
        """
        outputs = [{} for _ in range(output_embs.shape[0])] if unpack_output else {}

        for readout_id in output_readout_index.unique().tolist():
            readout_name = self._readout_id_to_name.get(readout_id, None)

            # ids with no head are padding, or tasks excluded from this readout
            if readout_name is None:
                continue

            mask = output_readout_index == readout_id
            task_output = self.projections[readout_name](output_embs[mask])

            if unpack_output:
                # scatter this task's rows back to the samples they came from
                batch_index = torch.where(mask)[0]
                targeted, batch_index = torch.unique(batch_index, return_inverse=True)
                for i in range(len(targeted)):
                    outputs[targeted[i]][readout_name] = task_output[batch_index == i]
            else:
                outputs[readout_name] = task_output

        return outputs

    def forward_varlen(
        self,
        output_embs: torch.Tensor,
        output_readout_index: torch.Tensor,
        output_batch_index: torch.Tensor,
        unpack_output: bool = False,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Project chained output embeddings through their per-task heads.

        As :meth:`forward`, but for sequences chained along a single batch dimension
        rather than padded, which avoids the padding memory.

        Args:
            output_embs: Output embeddings, ``(total_ntokens, dim)``.
            output_readout_index: Routing index per output token, ``(total_ntokens,)``.
            output_batch_index: Batch index per output token, ``(total_ntokens,)``.
            unpack_output: Return one dict per batch sample. See :meth:`forward`.

        Returns:
            See :meth:`forward`.
        """
        n_samples = output_batch_index.max().item() + 1
        outputs = [{} for _ in range(n_samples)] if unpack_output else {}

        for readout_id in output_readout_index.unique().tolist():
            readout_name = self._readout_id_to_name.get(readout_id, None)

            # ids with no head are padding, or tasks excluded from this readout
            if readout_name is None:
                continue

            mask = output_readout_index == readout_id
            task_output = self.projections[readout_name](output_embs[mask])

            if unpack_output:
                # sequences were chained, so batch membership comes from the index
                batch_index = output_batch_index[mask]
                targeted, batch_index = torch.unique(batch_index, return_inverse=True)
                for i in range(len(targeted)):
                    outputs[targeted[i]][readout_name] = task_output[batch_index == i]
            else:
                outputs[readout_name] = task_output

        return outputs
