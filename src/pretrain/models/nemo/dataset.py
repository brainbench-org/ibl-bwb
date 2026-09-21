import torch
from torch.utils.data import TensorDataset


class NEMOAugmentedTensorDataset(TensorDataset):
    """TensorDataset that applies augmentation transforms in __getitem__ (runs in DataLoader workers)."""

    def __init__(
        self,
        waveforms: torch.Tensor,
        acgs: torch.Tensor,
        wvf_transform=None,
        acg_transform=None,
    ):
        super().__init__(waveforms, acgs)
        self.wvf_transform = wvf_transform
        self.acg_transform = acg_transform

    def __getitem__(self, index):
        wf, acg = self.tensors[0][index], self.tensors[1][index]
        if self.wvf_transform is not None:
            wf = torch.from_numpy(self.wvf_transform(wf.numpy()))
        if self.acg_transform is not None:
            acg = torch.from_numpy(self.acg_transform(acg.numpy()))
        return wf, acg
