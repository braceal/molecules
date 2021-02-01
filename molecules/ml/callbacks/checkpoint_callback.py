import os
import time
import torch
import warnings
from pathlib import Path
from typing import Union, Optional
from .callback import Callback

PathLike = Union[str, Path]


class CheckpointCallback(Callback):
    def __init__(
        self,
        interval: int = 1,
        out_dir: Optional[PathLike] = None,
        mpi_comm=None,
    ):
        """
        Checkpoint interface for saving dictionary objects to disk
        during training. Typically used to save model state_dict
        and optimizer state_dict in order to resume training and
        record model weight history.

        Parameters
        ----------
        interval : int
            Plots every interval epochs, default is once per epoch.
        out_dir : Optional[PathLike]
            Directory to store checkpoint files.
            Files are named 'epoch-{e}-%Y%m%d-%H%M%S.pt'
        """
        super().__init__(interval, mpi_comm)

        if out_dir is None:
            self.out_dir = Path(".").joinpath("checkpoints")
        else:
            self.out_dir = Path(out_dir)

        if self.is_eval_node:
            self.out_dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs):
        if self.is_eval_node and epoch % self.interval == 0:
            self._new_save(epoch, logs)

    def _new_save(self, epoch, logs):
        """Saves arbitrary checkpoint dictionary."""
        checkpoint = logs.get("checkpoint", {})
        if not checkpoint:
            warnings.warn(
                "CheckpointCallback is defined but checkpoint logs dict is empty."
            )
            return
        checkpoint["epoch"] = epoch

        time_stamp = time.strftime(f"epoch-{epoch}-%Y%m%d-%H%M%S.pt")
        path = self.out_dir.joinpath(time_stamp)
        torch.save(checkpoint, path)

    def _save(self, epoch, logs):
        """Saves optimizer state and encoder/decoder weights."""

        # create new dictionary
        checkpoint = {"epoch": epoch}

        # optimizer
        if "optimizer" in logs:
            checkpoint["optimizer_state_dict"] = logs["optimizer"].state_dict()

        if "optimizer_d" in logs:
            checkpoint["optimizer_d_state_dict"] = logs["optimizer_d"].state_dict()

        if "optimizer_eg" in logs:
            checkpoint["optimizer_eg_state_dict"] = logs["optimizer_eg"].state_dict()

        # model parameter
        handle = logs["model"]
        # just to be safe here
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module

        if hasattr(handle, "encoder"):
            checkpoint["encoder_state_dict"] = handle.encoder.state_dict()

        if hasattr(handle, "decoder"):
            checkpoint["decoder_state_dict"] = handle.decoder.state_dict()

        if hasattr(handle, "generator"):
            checkpoint["generator_state_dict"] = handle.generator.state_dict()

        if hasattr(handle, "discriminator"):
            checkpoint["discriminator_state_dict"] = handle.discriminator.state_dict()

        time_stamp = time.strftime(f"epoch-{epoch}-%Y%m%d-%H%M%S.pt")
        path = os.path.join(self.out_dir, time_stamp)
        torch.save(checkpoint, path)
