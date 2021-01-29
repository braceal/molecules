from typing import Optional, Tuple
import time
import torch
import torch.optim
from torch import nn
from math import isclose
from molecules.ml.unsupervised.utils import get_activation, _init_weights
from molecules.ml.unsupervised.autoencoder.hyperparams import AutoEncoderHyperparams


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int],
        hparams: AutoEncoderHyperparams,
        init_weights: Optional[str] = None,
    ):
        super().__init__()

        assert isinstance(hparams, AutoEncoderHyperparams)
        hparams.validate()

        self.hparams = hparams
        self.input_dim = input_dim[0]  # D dimensions

        self.encoder = nn.Sequential(
            *self._affine_layers(),
        )

        self.init_weights(init_weights)

    def init_weights(self, init_weights: Optional[str]):
        if init_weights is None:
            self.encoder.apply(_init_weights)
        # Loading checkpoint weights
        elif init_weights.endswith(".pt"):
            checkpoint = torch.load(init_weights, map_location="cpu")
            self.load_state_dict(checkpoint["encoder_state_dict"])

    def forward(self, x):
        x = self.encoder(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(x)

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path))

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        layers : list
            Linear layers
        """

        layers = []

        in_features = self.input_dim

        act = get_activation(self.hparams.activation)

        for width, dropout in zip(
            self.hparams.affine_widths, self.hparams.affine_dropouts
        ):

            layers.append(nn.Linear(in_features=in_features, out_features=width))
            layers.append(act)

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add latent layer
        layers.append(
            nn.Linear(in_features=in_features, out_features=self.hparams.latent_dim)
        )
        layers.append(act)

        return layers


def reversedzip(*iterables):
    """
    Yields the zip of iterables in reversed order.

    Example
    -------
    l1 = [1,2,3]
    l2 = ['a','b','c']
    l3 = [5,6,7]

    for tup in reversedzip(l1, l2, l3):
        print(tup)

    Outputs:
        (3, 'c', 7)
        (2, 'b', 6)
        (1, 'a', 5)

    """
    for tup in zip(*map(reversed, iterables)):
        yield tup


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: Tuple[int],
        hparams: AutoEncoderHyperparams,
        init_weights: Optional[str] = None,
    ):
        super().__init__()

        assert isinstance(hparams, AutoEncoderHyperparams)
        hparams.validate()

        self.output_dim = output_dim[0]
        self.hparams = hparams

        self.decoder = nn.Sequential(*self._affine_layers())

        self.init_weights(init_weights)

    def init_weights(self, init_weights: Optional[str]):
        if init_weights is None:
            self.decoder.apply(_init_weights)
        # Loading checkpoint weights
        elif init_weights.endswith(".pt"):
            checkpoint = torch.load(init_weights, map_location="cpu")
            self.load_state_dict(checkpoint["decoder_state_dict"])

    def forward(self, x):
        x = self.decoder(x)
        return x

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(embedding)

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path))

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        layers : list
            Linear layers
        """

        layers = []

        act = get_activation(self.hparams.activation)

        in_features = self.hparams.latent_dim

        for width, dropout in reversedzip(
            self.hparams.affine_widths, self.hparams.affine_dropouts
        ):

            layers.append(nn.Linear(in_features=in_features, out_features=width))

            layers.append(act)

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add last layer
        layers.append(
            nn.Linear(
                in_features=self.hparams.affine_widths[0],
                out_features=self.output_dim,
            )
        )
        layers.append(get_activation(self.hparams.output_activation))

        return layers


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        hparams: AutoEncoderHyperparams,
        init_weights: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = Encoder(input_shape, hparams, init_weights)
        self.decoder = Decoder(input_shape, hparams, init_weights)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x


def _load_checkpoint(path: str, model: AutoEncoder, optimizer: torch.optim.Optimizer):
    """
    Loads checkpoint file containing optimizer state and
    encoder/decoder weights.

    Parameters
    ----------
    path : str
        Path to checkpoint file

    Returns
    -------
    Epoch of training corresponding to the saved checkpoint.
    """

    # checkpoint
    cp = torch.load(path, map_location="cpu")

    # model
    model.encoder.load_state_dict(cp["encoder_state_dict"])
    model.decoder.load_state_dict(cp["decoder_state_dict"])

    # optimizer
    optimizer.load_state_dict(cp["optimizer_state_dict"])

    return cp["epoch"]


def _train(
    model,
    optimizer,
    device,
    train_loader,
    criterion,
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

    model.train()
    train_loss = 0.0
    for batch_idx, sample in enumerate(train_loader):

        data = sample["X"]
        data = data.to(device)

        if verbose:
            start = time.time()

        if callbacks:
            logs["sample"] = sample

        for callback in callbacks:
            callback.on_batch_begin(batch_idx, epoch, logs)

        # forward
        _, recon_batch = model(data)
        loss = criterion(recon_batch, data)

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
                    (batch_idx + 1) * len(data),
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


def _validate(model, device, valid_loader, criterion, epoch, callbacks, logs, verbose):
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
    model.eval()
    valid_loss = 0.0
    for callback in callbacks:
        callback.on_validation_begin(epoch, logs)

    with torch.no_grad():
        for batch_idx, sample in enumerate(valid_loader):
            data = sample["X"].to(device)

            if callbacks:
                logs["sample"] = sample

            z, recon_batch = model(data)
            loss = criterion(recon_batch, data)
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
    model: AutoEncoder,
    optimizer: torch.optim.Optimizer,
    device,
    train_loader,
    valid_loader,
    criterion,
    epochs=1,
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
        start_epoch += _load_checkpoint(checkpoint, model, optimizer)

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
