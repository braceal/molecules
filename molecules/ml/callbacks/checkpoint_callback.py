import os
import time
import torch
from .callback import Callback


class CheckpointCallback(Callback):
    def __init__(
        self, interval=1, out_dir=os.path.join(".", "checkpoints"), mpi_comm=None
    ):
        """
        Checkpoint interface for saving dictionary objects to disk
        during training. Typically used to save model state_dict
        and optimizer state_dict in order to resume training and
        record model weight history.

        Parameters
        ----------
        out_dir : str
            Directory to store checkpoint files.
            Files are named 'epoch-{e}-%Y%m%d-%H%M%S.pt'
        interval : int
            Plots every interval epochs, default is once per epoch.
        """
        super().__init__(interval, mpi_comm)

        if self.is_eval_node:
            os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs):
        if self.is_eval_node and epoch % self.interval == 0:
            self._save(epoch, logs)

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
