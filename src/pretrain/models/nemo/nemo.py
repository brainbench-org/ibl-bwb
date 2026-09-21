import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        x = self.drop(F.gelu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class BasicBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        x = self.drop(F.gelu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


def _make_layer_1d(
    in_ch: int, out_ch: int, stride: int, dropout: float, n_blocks: int = 2
) -> nn.Sequential:
    layers = [BasicBlock1D(in_ch, out_ch, dropout=dropout, stride=stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock1D(out_ch, out_ch, dropout=dropout))
    return nn.Sequential(*layers)


def _make_layer_2d(
    in_ch: int, out_ch: int, stride: int, dropout: float, n_blocks: int = 2
) -> nn.Sequential:
    layers = [BasicBlock2D(in_ch, out_ch, dropout=dropout, stride=stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock2D(out_ch, out_ch, dropout=dropout))
    return nn.Sequential(*layers)


class WVFEncoder(nn.Module):
    def __init__(self, base_channels: int, dropout: float, dim_rep: int):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer_1d(c, c, stride=1, dropout=dropout, n_blocks=2)
        self.layer2 = _make_layer_1d(c, 2 * c, stride=2, dropout=dropout, n_blocks=3)
        self.layer3 = _make_layer_1d(2 * c, 4 * c, stride=2, dropout=dropout, n_blocks=3)
        self.layer4 = _make_layer_1d(4 * c, 8 * c, stride=2, dropout=dropout, n_blocks=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8 * c, dim_rep)
        self.dim_rep = dim_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x.unsqueeze(1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class ACGEncoder(nn.Module):
    def __init__(self, base_channels: int, dropout: float, dim_rep: int):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(3, 7), stride=2, padding=(1, 3), bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer_2d(c, c, stride=1, dropout=dropout, n_blocks=2)
        self.layer2 = _make_layer_2d(c, 2 * c, stride=2, dropout=dropout, n_blocks=2)
        self.layer3 = _make_layer_2d(2 * c, 4 * c, stride=2, dropout=dropout, n_blocks=2)
        self.layer4 = _make_layer_2d(4 * c, 8 * c, stride=2, dropout=dropout, n_blocks=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * c, dim_rep)
        self.dim_rep = dim_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)  # (N, dim_rep)


class LinearProjector(nn.Module):
    """Linear -> LayerNorm projection head (from NEMO LinearProjector, layer_norm=True)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x))


class NEMO(nn.Module):
    """NEMO bimodal contrastive learning model for IBL brain region pretraining :cite:`nemo`.

    Reference implementation: `Haansololfp/NEMO <https://github.com/Haansololfp/NEMO>`_.

    Interfaces:
        forward(wf, acg) -> (z_wf, z_acg)  L2-normalised projections
        representation(wf, acg) -> (wf_rep, acg_rep)  pre-projection
    """

    def __init__(
        self,
        wvf_base_channels: int,
        wvf_dropout: float,
        wvf_dim_rep: int,
        acg_base_channels: int,
        acg_dropout: float,
        acg_dim_rep: int,
        dim_embed: int,
    ):
        super().__init__()
        self.wvf_encoder = WVFEncoder(wvf_base_channels, wvf_dropout, wvf_dim_rep)
        self.acg_encoder = ACGEncoder(acg_base_channels, acg_dropout, acg_dim_rep)
        self.wvf_projector = LinearProjector(self.wvf_encoder.dim_rep, dim_embed)
        self.acg_projector = LinearProjector(self.acg_encoder.dim_rep, dim_embed)

    def representation(
        self, wf: torch.Tensor, acg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-projection representations (used for linear probe)."""
        return self.wvf_encoder(wf), self.acg_encoder(acg)

    def forward(self, wf: torch.Tensor, acg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L2-normalised post-projection embeddings."""
        wf_rep, acg_rep = self.representation(wf, acg)
        z_wf = F.normalize(self.wvf_projector(wf_rep), dim=1)
        z_acg = F.normalize(self.acg_projector(acg_rep), dim=1)
        return z_wf, z_acg
