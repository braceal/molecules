import time
import h5py
import numpy as np
from pathlib import Path
from typing import List
from .callback import Callback


class SaveEmbeddingsCallback(Callback):

    """
    Saves embeddings
    """

    def __init__(
        self,
        out_dir: str,
        interval: int = 1,
        sample_interval: int = 20,
        embeddings_dset_name: str = "embeddings",
        scalar_dset_names: List[str] = ["rmsd", "fnc"],
        mpi_comm=None,
    ):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output embedding files.
        interval : int
            Plots every interval epochs, default is once per epoch.
        sample_interval : int
            Plots every sample_interval'th point in the data set
        embeddings_dset_name: str
            Name of the embeddings dataset in the HDF5 file.
        scalar_dset_names : List[str]
            List of scalar dataset names inside HDF5 file.
        mpi_comm : mpi communicator for distributed training
        """
        super().__init__(interval, mpi_comm)

        self.out_dir = Path(out_dir)
        self.sample_interval = sample_interval
        self.embeddings_dset_name = embeddings_dset_name
        self.scalar_dset_names = scalar_dset_names

        if self.is_eval_node:
            self.out_dir.mkdir(exist_ok=True)

    def on_validation_begin(self, epoch, logs):
        self.sample_counter = 0
        self.embeddings = []
        self.scalars = {name: [] for name in self.scalar_dset_names}

    def on_validation_batch_end(self, batch, epoch, logs, **kwargs):
        if self.sample_interval == 0:
            return
        if epoch % self.interval != 0:
            return
        mu, scalars, index = logs.get("mu"), logs.get("scalars"), logs.get("index")
        if (mu is None) or (index is None):
            return

        # decide what to store
        for idx in range(len(mu)):
            if (self.sample_counter + idx) % self.sample_interval == 0:
                # use a singleton slice to keep dimensions intact
                self.embeddings.append(mu[idx : idx + 1].detach().cpu().numpy())
                for name, data in scalars:
                    self.scalars[name].append(
                        data[idx : idx + 1].detach().cpu().numpy()
                    )

        # increase sample counter
        self.sample_counter += len(mu)

    def on_validation_end(self, epoch, logs):
        if epoch % self.interval != 0:
            return
        # if the sample interval was too large, we should warn here and return
        if not self.embeddings:
            print(
                "Warning, not enough samples collected for tSNE, \
                  try to reduce sampling interval"
            )
            return

        # prepare data
        embeddings = np.concatenate(self.embeddings, axis=0).astype(np.float32)
        scalars = {
            name: np.concatenate(dset, axis=0).astype(np.float32)
            for name, dset in self.scalars.items()
        }

        # communicate if necessary
        if self.comm is not None:
            # gather data
            embeddings_gather = self.comm.gather(embeddings, root=0)
            scalars_gather = {
                name: self.comm.gather(scalar, root=0)
                for scalar, name in scalars.items()
            }
            # concat
            if self.is_eval_node:
                embeddings = np.concatenate(embeddings_gather, axis=0)
                scalars = {
                    name: np.concatenate(scalar, axis=0)
                    for name, scalar in scalars_gather.items()
                }

        # Save embeddings to disk
        if self.is_eval_node and (self.sample_interval > 0):
            self.save_embeddings(epoch, embeddings, scalars, logs)

        # All other nodes wait for node 0 to save
        if self.comm is not None:
            self.comm.barrier()

    def save_embeddings(self, epoch, embeddings, scalars, logs):
        # Create embedding file path and store in logs for downstream callbacks
        time_stamp = time.strftime(f"embeddings-epoch-{epoch}-%Y%m%d-%H%M%S.h5")
        embeddings_path = self.out_dir.joinpath(time_stamp).as_posix()
        logs["embeddings_path"] = embeddings_path

        # Write embedding data to disk
        with h5py.File(embeddings_path, "w", libver="latest", swmr=False) as f:
            f[self.embeddings_dset_name] = embeddings[...]
            for name, dset in scalars.items():
                f[name] = dset[...]
