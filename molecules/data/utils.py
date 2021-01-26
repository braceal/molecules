import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

PathLike = Union[Path, str]


def concatenate_virtual_h5(
    input_file_names: List[str], output_name: str, fields: Optional[List[str]] = None
):
    r"""Concatenate HDF5 files into a virtual HDF5 file.
    Concatenates a list `input_file_names` of HDF5 files containing
    the same format into a single virtual dataset.
    Parameters
    ----------
    input_file_names : List[str]
        List of HDF5 file names to concatenate.
    output_name : str
        Name of output virtual HDF5 file.
    fields : Optional[List[str]]
        Which dataset fields to concatenate. Will concatenate all fields by default.
    """

    # Open first file to get dataset shape and dtype
    # Assumes uniform number of data points per file
    h5_file = h5py.File(input_file_names[0], "r")

    if not fields:
        fields = list(h5_file.keys())

    # Helper function to output concatenated shape
    def concat_shape(shape: Tuple[int]) -> Tuple[int]:
        return (len(input_file_names) * shape[0], *shape[1:])

    # Create a virtual layout for each input field
    layouts = {
        field: h5py.VirtualLayout(
            shape=concat_shape(h5_file[field].shape),
            dtype=h5_file[field].dtype,
        )
        for field in fields
    }

    with h5py.File(output_name, "w", libver="latest") as f:
        for field in fields:
            for i, filename in enumerate(input_file_names):
                shape = h5_file[field].shape
                vsource = h5py.VirtualSource(filename, field, shape=shape)
                layouts[field][i * shape[0] : (i + 1) * shape[0], ...] = vsource

            f.create_virtual_dataset(field, layouts[field])

    h5_file.close()


def concatenate_h5(
    input_file_names: List[str], output_name: str, fields: Optional[List[str]] = None
):

    if not fields:
        # Peak into first file and collect all the field names
        with h5py.File(input_file_names[0], "r") as h5_file:
            fields = list(h5_file.keys())

    # Initialize data buffers
    data = {x: [] for x in fields}

    for in_file in input_file_names:
        with h5py.File(in_file, "r", libver="latest") as fin:
            for field in fields:
                data[field].append(fin[field][...])

    # Concatenate data
    for field in data:
        data[field] = np.concatenate(data[field])

    # Create new dsets from concatenated dataset
    fout = h5py.File(output_name, "w", libver="latest")
    for field, concat_dset in data.items():

        shape = concat_dset.shape
        chunkshape = (1,) + shape[1:]
        # Create dataset
        if concat_dset.dtype != np.object:
            if np.any(np.isnan(concat_dset)):
                raise ValueError("NaN detected in concat_dset.")
            dset = fout.create_dataset(
                field, shape, chunks=chunkshape, dtype=concat_dset.dtype
            )
        else:
            dset = fout.create_dataset(
                field, shape, chunks=chunkshape, dtype=h5py.vlen_dtype(np.int16)
            )
        # write data
        dset[...] = concat_dset[...]

    # Clean up
    fout.flush()
    fout.close()


def parse_h5(path: PathLike, fields: List[str]) -> Dict[str, np.ndarray]:
    r"""Helper function for accessing data fields in H5 file.

    Parameters
    ----------
    path : Union[Path, str]
        Path to HDF5 file.
    fields : List[str]
        List of dataset field names inside of the HDF5 file.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary maping each field name in `fields` to a numpy
        array containing the data from the associated HDF5 dataset.
    """
    data = {}
    with h5py.File(path, "r") as f:
        for field in fields:
            data[field] = f[field][...]
    return data


def unimodal_subsample(
    data: np.ndarray,
    peak_left_bound: float,
    peak_right_bound: float,
    peak_samples: int,
) -> np.ndarray:
    r"""Return indices into `data` with a portion of the peak and both tails.

    Assuming `data` is unimodal, sample at random `peak_samples` from the peak
    region bound by the interval [`peak_left_bound`, `peak_right_bound`] and
    all the data from both left and right tails.

    Parameters
    ----------
    data : np.ndarray
        Real valued data distribution to sample from.
    peak_left_bound : float
        Value of `data` defining the end of the left tail.
    peak_right_bound : float
        Value of `data` defining the start of the right tail.
    peak_samples : int
        The number of samples to take from the peak.

    Returns
    -------
    np.ndarray
        The indices into `data` of the subsample.
    """
    # Define boolean index arrays for each part of the distribution
    left_tail = data < peak_left_bound
    right_tail = data > peak_right_bound
    peak = ~(left_tail | right_tail)

    inds = np.arange(len(data))
    return np.concatenate(
        (
            np.random.choice(inds[peak], size=peak_samples),
            inds[left_tail],
            inds[right_tail],
        )
    )
