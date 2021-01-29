import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from molecules.utils import open_h5


class ContactMapDataset(Dataset):
    """
    PyTorch Dataset class to load contact matrix data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """

    def __init__(
        self,
        path: str,
        dataset_name: str,
        scalar_dset_names: List[str],
        shape: Tuple[int],
        split_ptc: float = 0.8,
        split: str = "train",
        seed: int = 333,
        cm_format: str = "sparse-concat",
        scalar_requires_grad: bool = False,
    ):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.

        dataset_name : str
            Path to contact maps in HDF5 file.

        scalar_dset_names : List[str]
            List of scalar dataset names inside HDF5 file to be passed
            to training logs.

        shape : tuple
            Shape of contact matrices (H, W), may be (1, H, W).

        split_ptc : float
            Percentage of total data to be used as training set.

        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.

        seed : int
            Seed for the RNG for the splitting. Make sure it is the same for all workers reading
            from the same file.

        cm_format : str
            If 'sparse-concat', process data as concatenated row,col indicies.
            If 'sparse-rowcol', process data as sparse row/col COO format.
            If 'full', process data is normal torch tensors (matrices).
            If none of the above, raise a ValueError.

        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by `scalar_dset_names`.
            Set to True, to use scalars for multi-task learning. If scalars are only required for
            plotting, then set it as False.
        """
        if split not in ("train", "valid"):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError("Parameter split_ptc must satisfy 0 <= split_ptc <= 1.")
        if cm_format not in ("sparse-concat", "sparse-rowcol", "full"):
            raise ValueError(
                f"Invalid cm_format {cm_format}. Should be one of "
                "[sparse-rowcol, sparse-concat, full]."
            )

        # HDF5 data params
        self.file_path = path
        self.dataset_name = dataset_name
        self.scalar_dset_names = scalar_dset_names
        self.cm_format = cm_format
        self.shape = shape
        self._scalar_requires_grad = scalar_requires_grad
        self.load_values = False

        # get lengths and paths
        with open_h5(self.file_path, "r", libver="latest", swmr=False) as f:
            if self.cm_format == "sparse-rowcol":
                self.len = len(f[self.dataset_name]["row"])
            elif self.cm_format == "sparse-concat":
                self.len = len(f[self.dataset_name])
            elif self.cm_format == "full":
                self.len = len(f[self.dataset_name])

            # check if we need to load values
            if self.dataset_name + "_values" in f:
                self.load_values = True

        # do splitting
        self.split_ind = int(split_ptc * self.len)
        self.split = split
        split_rng = np.random.default_rng(seed)
        self.indices = split_rng.permutation(list(range(self.len)))
        if self.split == "train":
            self.indices = sorted(self.indices[: self.split_ind])
        else:
            self.indices = sorted(self.indices[self.split_ind :])

        # inited:
        self._initialized = False

    def _init_file(self):
        self.h5_file = open_h5(self.file_path, "r", libver="latest", swmr=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self._init_file()
            if self.cm_format == "sparse-rowcol":
                self.row_dset = self.h5_file[self.dataset_name]["row"]
                self.col_dset = self.h5_file[self.dataset_name]["col"]
                if self.load_values:
                    self.val_dset = self.h5_file[self.dataset_name + "_values"]
            elif self.cm_format == "sparse-concat":
                self.dset = self.h5_file[self.dataset_name]
                if self.load_values:
                    self.val_dset = self.h5_file[self.dataset_name + "_values"]
            else:
                self.dset = self.h5_file[self.dataset_name]
            # Load scalar dsets
            self.scalar_dsets = {
                name: self.h5_file[name] for name in self.scalar_dset_names
            }
            self._initialized = True

        # get real index
        index = self.indices[idx]

        # Select data format and return data at idx
        if self.cm_format == "full":
            data = torch.Tensor(self.dset[index, ...])
        else:
            if self.cm_format == "sparse-concat":
                ind = self.dset[index, ...].reshape(2, -1)
                indices = torch.from_numpy(ind).to(torch.long)
            else:  # sparse-rowcol
                rowind = self.row_dset[index, ...]
                colind = self.col_dset[index, ...]
                indices = torch.from_numpy(np.vstack((rowind, colind))).to(torch.long)

            # Create array of 1s, all values in the contact map are 1
            if self.load_values:
                values = torch.from_numpy(self.val_dset[index, ...]).to(torch.float32)
            else:
                values = torch.ones(indices.shape[1], dtype=torch.float32)
            # Set shape to the last 2 elements of self.shape.
            # Handles (1, W, H) and (W, H)
            data = torch.sparse.FloatTensor(indices, values, self.shape[-2:]).to_dense()
            data = data.view(self.shape)

        sample = {"X": data}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(index, requires_grad=False)
        # Add scalars for logging
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[index], requires_grad=self._scalar_requires_grad
            )

        return sample
