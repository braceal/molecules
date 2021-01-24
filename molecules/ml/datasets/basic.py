from pathlib import Path
from typing import Union
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

PathLike = Union[str, Path]


class BasicDataset(Dataset):
    """
    PyTorch Dataset class to load vector or scalar data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """

    def __init__(
        self,
        path: PathLike,
        dataset_name: str,
        split_ptc: float = 0.8,
        split: str = "train",
        seed: int = 333,
    ):
        """
        Parameters
        ----------
        path : PathLike
            Path to h5 file containing contact matrices.

        dataset_name : str
            Path to contact maps in HDF5 file.

        split_ptc : float
            Percentage of total data to be used as training set.

        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.

        seed : int
            Seed for the RNG for the splitting. Make sure it is the same for
            all workers reading from the same file.
        """
        if split not in ("train", "valid"):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError("Parameter split_ptc must satisfy 0 <= split_ptc <= 1.")

        self._file_path = Path(path)
        self._dataset_name = dataset_name
        self._initialized = False

        # get lengths and paths
        with self._open_h5_file() as f:
            datalen = len(f[self._dataset_name])

        split_ind = int(split_ptc * datalen)
        split_rng = np.random.default_rng(seed)
        self.indices = split_rng.permutation(np.arange(datalen))
        if split == "train":
            self.indices = sorted(self.indices[:split_ind])
        else:
            self.indices = sorted(self.indices[split_ind:])

    def _open_h5_file(self):
        return h5py.File(self._file_path, "r", libver="latest", swmr=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self.h5_file = self._open_h5_file()
            self.dset = self.h5_file[self._dataset_name]
            self._initialized = True

        # Get real index
        index = self.indices[idx]

        # Select data format and return data at idx
        data = torch.Tensor(self.dset[index, ...])

        return data, index
