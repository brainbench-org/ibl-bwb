import itertools

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with optional batchnorm and dropout.

    Args:
        hidden_layers: list of dims from input to output. Use -1 for lazy first layer.
        batchnorm: insert BatchNorm1d after each linear layer.
        drop_last_nonlin: remove the ReLU/batchnorm/dropout of the last block.
        dropout: dropout probability applied after the ReLU in each block.
    """

    def __init__(
        self,
        hidden_layers: list[int],
        *,
        bias: bool,
        batchnorm: bool,
        dropout: float,
        drop_last_nonlin: bool = True,
    ):
        super().__init__()
        layers = []
        for in_dim, out_dim in itertools.pairwise(hidden_layers):
            if in_dim == -1:
                layers.append(nn.LazyLinear(out_dim, bias=bias))
            else:
                layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        if drop_last_nonlin:
            layers = layers[: -(1 + int(batchnorm) + int(dropout > 0.0))]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadGlobalAttention(nn.Module):
    """Multi-Head Global pooling layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        *,
        bias: bool,
        batchnorm: bool,
        dropout: float,
    ):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels

        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, heads * in_channels, bias=False),
            nn.PReLU(),
            nn.Linear(heads * in_channels, heads, bias=False),
        )
        self.proj = MLP(
            [in_channels, out_channels * heads, out_channels * heads],
            bias=bias,
            batchnorm=batchnorm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        n_units = int(batch.max().item()) + 1

        gate = self.gate_nn(x)  # (N, heads)
        score = self._batched_softmax(gate, batch, n_units)  # (N, heads)

        proj = self.proj(x).view(x.size(0), self.heads, self.out_channels)  # (N, heads, C)
        score = score.unsqueeze(-1)  # (N, heads, 1)
        weighted = score * proj  # (N, heads, C)

        out = torch.zeros(n_units, self.heads, self.out_channels, dtype=x.dtype, device=x.device)
        idx = batch.view(-1, 1, 1).expand_as(weighted)
        out.scatter_add_(0, idx, weighted)  # (B, heads, C)
        return out.view(n_units, -1)  # (B, heads * C)

    @staticmethod
    def _batched_softmax(gate: torch.Tensor, batch: torch.Tensor, n_units: int) -> torch.Tensor:
        """Softmax over nodes within each batch item (each unit)."""
        idx = batch.unsqueeze(1).expand_as(gate)
        src_max = torch.full(
            (n_units, gate.size(1)), float("-inf"), dtype=gate.dtype, device=gate.device
        )
        src_max.scatter_reduce_(0, idx, gate.detach(), reduce="amax", include_self=True)
        out = (gate - src_max[batch]).exp()
        out_sum = torch.zeros(n_units, gate.size(1), dtype=gate.dtype, device=gate.device)
        out_sum.scatter_add_(0, idx, out)
        return out / (out_sum[batch] + 1e-16)


class LOLCAT(nn.Module):
    """ISI histogram encoder, attention pooling and a brain region classifier :cite:`lolcat`.

    Reference implementation: `nerdslab/lolcat <https://github.com/nerdslab/lolcat>`_.
    """

    def __init__(
        self,
        n_classes: int,
        encoder_hidden: list[int],
        n_heads: int,
        attn_hidden: int,
        dropout: float,
        batchnorm: bool,
        bias: bool,
        n_bins: int,
    ):
        super().__init__()
        enc_out = encoder_hidden[-1]
        self.encoder = MLP(
            [n_bins, *encoder_hidden], bias=bias, batchnorm=batchnorm, dropout=dropout
        )
        self.pool = MultiHeadGlobalAttention(
            enc_out,
            attn_hidden,
            heads=n_heads,
            bias=bias,
            batchnorm=batchnorm,
            dropout=dropout,
        )
        self.classifier = MLP(
            [attn_hidden * n_heads, n_classes],
            bias=True,
            batchnorm=False,
            dropout=0.0,
        )

    @staticmethod
    def collate(batch: list[dict]) -> tuple:
        x = torch.cat([item["isi_hists"] for item in batch], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((item["isi_hists"].size(0),), i, dtype=torch.long)
                for i, item in enumerate(batch)
            ]
        )
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        uids = [item["uid"] for item in batch]
        return x, batch_idx, labels, uids

    def link_datasets(self, train_ds, val_ds, test_ds) -> None:
        assert train_ds.label_map == val_ds.label_map == test_ds.label_map, (
            "All splits must share the same label_map"
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        global_emb = self.pool(emb, batch)
        return self.classifier(global_emb)
