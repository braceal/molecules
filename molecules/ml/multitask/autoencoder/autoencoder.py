import time
import torch
from torch import nn
import torch.optim
from typing import List, Callable, Tuple
from molecules.data.utils import dict_to_device
from molecules.ml.unsupervised.autoencoder.autoencoder import AutoEncoder


class MultiTaskModelHead(nn.Module):
    def __init__(self, criterion: Callable, scalar_name: str):
        super().__init__()

        self._criterion = criterion
        self._scalar_name = scalar_name

    def criterion(self, x_pred, sample):
        x_true = sample[self._scalar_name]
        return self._criterion(x_pred, x_true)

    def forward(self, x):
        pass


class RMSDNet(MultiTaskModelHead):
    def __init__(self, criterion: Callable, scalar_name: str, input_dim: int):
        super().__init__(criterion, scalar_name)

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=input_dim // 2, out_features=1),
        )

    def forward(self, z):
        return self.model(z)


class MultiTaskModel(nn.Module):
    def __init__(self, autoencoder_model: AutoEncoder, heads: List[MultiTaskModelHead]):
        super().__init__()
        self.autoencoder_model = autoencoder_model
        self.heads = nn.ModuleList(heads)

        # To implement the loss weighting for different tasks as in
        # `https://arxiv.org/pdf/1705.07115.pdf`
        self.multitask_log_vars = nn.Parameter(torch.zeros((len(heads))))
        self.recon_log_var = nn.Parameter(torch.zeros((1,)))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z, recon_x = self.autoencoder_model(x)
        model_preds = []
        for model in self.heads:
            model_preds.append(model(z))
        return z, recon_x, model_preds


def _compute_weighted_loss(loss, log_var):
    precision = torch.exp(-log_var)
    # Add + log_var to regularize
    weighted_loss = torch.sum(precision * loss + log_var, -1)
    return weighted_loss


def _compute_loss(monolith, sample, recon_criterion, recon_x, model_preds):
    # Compute loss
    recon_loss = recon_criterion(recon_x, sample["X"])
    loss = _compute_weighted_loss(recon_loss, monolith.recon_log_var)

    # multi-task loss
    for pred, head, log_var in zip(
        model_preds, monolith.heads, monolith.multitask_log_vars
    ):
        task_loss = head.criterion(pred, sample)
        loss += _compute_weighted_loss(task_loss, log_var)

    return loss


def _train(
    monolith: MultiTaskModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader,
    recon_criterion,
    epoch,
    callbacks,
    logs,
    verbose: bool = True,
):
    """
    Train for 1 epoch

    Parameters
    ----------
    train_loader : torch.utils.data.dataloader.DataLoader
        Contains training data

    epoch : int
        Current epoch of training

    callbacks : list
        Contains molecules.utils.callback.Callback objects
        which are called during training.

    logs : dict
        Filled with data for callbacks
    """

    monolith.train()
    train_loss = 0.0
    for batch_idx, sample in enumerate(train_loader):

        sample = dict_to_device(sample, device)

        if verbose:
            start = time.time()

        if callbacks:
            logs["sample"] = sample

        for callback in callbacks:
            callback.on_batch_begin(batch_idx, epoch, logs)

        # forward
        _, recon_x, model_preds = monolith(sample["X"])

        # Compute loss
        loss = _compute_loss(monolith, sample, recon_criterion, recon_x, model_preds)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update loss
        train_loss += loss.item()

        if callbacks:
            logs["train_loss"] = loss.item()
            logs["global_step"] = (epoch - 1) * len(train_loader) + batch_idx

        if verbose:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}".format(
                    epoch,
                    (batch_idx + 1) * len(sample["X"]),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - start,
                )
            )

        for callback in callbacks:
            callback.on_batch_end(batch_idx, epoch, logs)

    train_loss_ave = train_loss / float(batch_idx + 1)

    if callbacks:
        logs["train_loss_average"] = train_loss_ave

    if verbose:
        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss_ave))


def _validate(
    monolith, device, valid_loader, recon_criterion, epoch, callbacks, logs, verbose
):
    """
    Test model on validation set.

    Parameters
    ----------
    valid_loader : torch.utils.data.dataloader.DataLoader
        Contains validation data

    callbacks : list
        Contains molecules.utils.callback.Callback objects
        which are called during training.

    logs : dict
        Filled with data for callbacks
    """
    monolith.eval()
    valid_loss = 0.0
    for callback in callbacks:
        callback.on_validation_begin(epoch, logs)

    with torch.no_grad():
        for batch_idx, sample in enumerate(valid_loader):
            sample = dict_to_device(sample, device)

            if callbacks:
                logs["sample"] = sample

            z, recon_x, model_preds = monolith(sample["X"])

            # Compute loss
            loss = _compute_loss(
                monolith, sample, recon_criterion, recon_x, model_preds
            )

            valid_loss += loss.item()

            if callbacks:
                logs["embeddings"] = z.detach()

            for callback in callbacks:
                callback.on_validation_batch_end(
                    epoch,
                    batch_idx,
                    logs,
                )

    valid_loss /= float(batch_idx + 1)

    if callbacks:
        logs["valid_loss"] = valid_loss

    for callback in callbacks:
        callback.on_validation_end(epoch, logs)

    if verbose:
        print("====> Validation loss: {:.4f}".format(valid_loss))


def train(
    model: MultiTaskModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader,
    valid_loader,
    criterion,
    epochs: int = 1,
    checkpoint=None,
    callbacks=[],
    verbose: bool = True,
):
    """
    Train model

    Parameters
    ----------
    train_loader : torch.utils.data.dataloader.DataLoader
        Contains training data

    valid_loader : torch.utils.data.dataloader.DataLoader
        Contains validation data

    epochs : int
        Number of epochs to train for

    checkpoint : str, None
        Path to checkpoint file to load and resume training
        from the epoch when the checkpoint was saved.

    callbacks : list
        Contains molecules.utils.callback.Callback objects
        which are called during training.
    """

    if callbacks:
        logs = {"model": model, "optimizer": optimizer}
    else:
        logs = {}

    start_epoch = 1

    if checkpoint:
        # start_epoch += _load_checkpoint(checkpoint, model, optimizer)
        raise NotImplementedError("Loading checkpoints not supported.")

    for callback in callbacks:
        callback.on_train_begin(logs)

    for epoch in range(start_epoch, epochs + 1):

        for callback in callbacks:
            callback.on_epoch_begin(epoch, logs)

        _train(
            model,
            optimizer,
            device,
            train_loader,
            criterion,
            epoch,
            callbacks,
            logs,
            verbose,
        )
        _validate(
            model, device, valid_loader, criterion, epoch, callbacks, logs, verbose
        )

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

    for callback in callbacks:
        callback.on_train_end(logs)
