import time
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from collections import namedtuple
from .resnet import ResnetVAEHyperparams
from .symmetric import SymmetricVAEHyperparams
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer
import torch.cuda.amp as amp

__all__ = ["VAE"]

Device = namedtuple("Device", ["encoder", "decoder"])


class VAEModel(nn.Module):
    def __init__(self, input_shape, hparams, init_weights, device):
        super(VAEModel, self).__init__()

        # Select encoder/decoder models by the type of the hparams
        if isinstance(hparams, SymmetricVAEHyperparams):
            from .symmetric import SymmetricEncoderConv2d, SymmetricDecoderConv2d

            self.encoder = SymmetricEncoderConv2d(input_shape, hparams, init_weights)
            self.decoder = SymmetricDecoderConv2d(
                input_shape, hparams, self.encoder.shapes, init_weights
            )

        elif isinstance(hparams, ResnetVAEHyperparams):
            from .resnet import ResnetEncoder, ResnetDecoder

            self.encoder = ResnetEncoder(input_shape, hparams, init_weights)
            self.decoder = ResnetDecoder(
                self.encoder.match_shape, input_shape, hparams, init_weights
            )

        else:
            raise TypeError(f"Invalid hparams type: {type(hparams)}.")

        self.encoder.to(device.encoder)
        self.decoder.to(device.decoder)

        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x should be placed on encoder gpu in the dataset class
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar).to(self.device.decoder)
        x = self.decoder(z).to(self.device.encoder)
        return x, z, mu, logvar

    def encode(self, x):
        # mu layer
        return self.encoder.encode(x)

    def decode(self, embedding):
        return self.decoder.decode(embedding)

    def save_weights(self, enc_path, dec_path):
        self.encoder.save_weights(enc_path)
        self.decoder.save_weights(dec_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder.load_weights(enc_path)
        self.decoder.load_weights(dec_path)


def vae_loss(recon_x, x, mu, logvar, reduction="mean"):
    """
    Effects
    -------
    Reconstruction + KL divergence losses summed over all elements and batch

    See Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    """

    BCE = F.binary_cross_entropy(recon_x, x, reduction=reduction)

    # 0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def vae_logit_loss(logit_recon_x, x, mu, logvar, reduction="mean"):
    """
    As above, but works directly on logits
    """
    BCE = F.binary_cross_entropy_with_logits(logit_recon_x, x, reduction=reduction)

    # 0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def vae_logit_loss_outlier_helper(
    logit_recon_x, x, mu, logvar, lambda_rec=1.0, reduction="mean"
):
    """
    As above, but works directly on logits
    """
    BCE = F.binary_cross_entropy_with_logits(logit_recon_x, x, reduction=reduction)
    # 0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return lambda_rec * BCE, KLD


# TODO: set weight initialization hparams
class VAE:
    """
    Provides high level interface for training, testing and saving VAE
    models. Takes arbitrary encoder/decoder models specified by the choice
    of hyperparameters. Assumes the shape of the data is square.

    Attributes
    ----------
    model : torch.nn.Module (VAEModel)
        Underlying Pytorch model with encoder/decoder attributes.

    optimizer : torch.optim.Optimizer
        Pytorch optimizer used to train model.

    loss_func : function
        Loss function used to train model.

    Methods
    -------
    train(train_loader, valid_loader, epochs=1, checkpoint='', callbacks=[])
        Train model

    encode(x)
        Embed data into the latent space.

    decode(embedding)
        Generate matrices from embeddings.

    save_weights(enc_path, dec_path)
        Save encoder/decoder weights.

    load_weights(enc_path, dec_path)
        Load saved encoder/decoder weights.
    """

    def __init__(
        self,
        input_shape,
        hparams=SymmetricVAEHyperparams(),
        optimizer_hparams=OptimizerHyperparams(),
        loss=None,
        gpu=None,
        enable_amp=False,
        init_weights=None,
        verbose=True,
    ):
        """
        Parameters
        ----------
        input_shape : tuple
            shape of incomming data.
            Note: For use with SymmetricVAE use (1, num_residues, num_residues)
                  For use with ResnetVAE use (num_residues, num_residues)

        hparams : molecules.ml.hyperparams.Hyperparams
            Defines the model architecture hyperparameters. Currently implemented
            are SymmetricVAEHyperparams and ResnetVAEHyperparams.

        optimizer_hparams : molecules.ml.hyperparams.OptimizerHyperparams
            Defines the optimizer type and corresponding hyperparameters.

        loss: : function, optional
            Defines an optional loss function with inputs (recon_x, x, mu, logvar)
            and ouput torch loss.

        enable_amp: bool
            Set to true to enable automatic mixed precision.

        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively.

        init_weights : str, None
            If str and ends with .pt, init_weights is a model checkpoint from
            which pretrained weights can be loaded for the encoder and decoder.
            If None, model will start with default weight initialization.

        verbose : bool
            True prints training and validation loss to stdout.
        """

        hparams.validate()
        optimizer_hparams.validate()

        self.enable_amp = enable_amp
        self.verbose = verbose

        # Tuple of encoder, decoder device
        self.device = Device(*self._configure_device(gpu))

        self.model = VAEModel(input_shape, hparams, init_weights, self.device)

        # TODO: consider making optimizer_hparams a member variable
        # RMSprop with lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_hparams)

        # amp grad scaler
        self.gscaler = amp.GradScaler(enabled=self.enable_amp)

        self.loss_fnc = vae_logit_loss if loss is None else loss
        self.lambda_rec = hparams.lambda_rec

        # these are helpers for distributed computing
        self.comm_rank = 0
        self.comm_size = 1
        if dist.is_initialized():
            self.comm_rank = dist.get_rank()
            self.comm_size = dist.get_world_size()

    def _configure_device(self, gpu):
        """
        Configures GPU/CPU device for training VAE. Allows encoder
        and decoder to be trained on seperate devices.

        Parameters
        ----------
        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively or None
            option if tuple contains None.

        Returns
        -------
        2-tuple of encoder_device, decoder_device
        """

        if gpu is None or (isinstance(gpu, tuple) and None in gpu):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return device, device
        if not torch.cuda.is_available():
            raise ValueError("Specified GPU training but CUDA is not available.")
        if isinstance(gpu, int):
            device = torch.device(f"cuda:{gpu}")
            return device, device
        if isinstance(gpu, tuple) and len(gpu) == 2:
            return torch.device(f"cuda:{gpu[0]}"), torch.device(f"cuda:{gpu[1]}")
        raise ValueError(
            "Specified GPU device is invalid. Should be int, 2-tuple or None."
        )

    def __repr__(self):
        return str(self.model)

    def train(
        self, train_loader, valid_loader, epochs=1, checkpoint=None, callbacks=[]
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
            handle = self.model
            if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
                handle = handle.module
            logs = {"model": handle, "optimizer": self.optimizer}
            if dist.is_initialized():
                logs["comm_size"] = self.comm_size
        else:
            logs = {}

        start_epoch = 1

        if checkpoint:
            start_epoch += self._load_checkpoint(checkpoint)

        for callback in callbacks:
            callback.on_train_begin(logs)

        for epoch in range(start_epoch, epochs + 1):

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            self._train(train_loader, epoch, callbacks, logs)
            self._validate(valid_loader, epoch, callbacks, logs)

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end(logs)

    def _train(self, train_loader, epoch, callbacks, logs):
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

        self.model.train()
        train_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):

            data = sample["X"]
            data = data.to(self.device.encoder)

            if self.verbose:
                start = time.time()

            if callbacks:
                logs["sample"] = sample

            for callback in callbacks:
                callback.on_batch_begin(batch_idx, epoch, logs)

            # forward
            with amp.autocast(self.enable_amp):
                logit_recon_batch, codes, mu, logvar = self.model(data)
                loss_rec, loss_kld = self.loss_fnc(logit_recon_batch, data, mu, logvar)
                loss = self.lambda_rec * loss_rec + loss_kld

            # backward
            self.optimizer.zero_grad()
            self.gscaler.scale(loss).backward()
            self.gscaler.step(self.optimizer)
            self.gscaler.update()

            # update loss
            train_loss += loss.item()

            if callbacks:
                logs["train_loss"] = loss.item()
                logs["train_loss_rec"] = loss_rec.item()
                logs["train_loss_kld"] = loss_kld.item()
                logs["global_step"] = (epoch - 1) * len(train_loader) + batch_idx

            if self.verbose and (self.comm_rank == 0):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}".format(
                        epoch,
                        (batch_idx + 1) * self.comm_size * len(data),
                        self.comm_size * len(train_loader.dataset),
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

        if self.verbose and (self.comm_rank == 0):
            print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss_ave))

    def _validate(self, valid_loader, epoch, callbacks, logs):
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
        self.model.eval()
        valid_loss = 0
        for callback in callbacks:
            callback.on_validation_begin(epoch, logs)

        with torch.no_grad():
            for batch_idx, sample in enumerate(valid_loader):
                data = sample["X"].to(self.device.encoder)

                if callbacks:
                    logs["sample"] = sample

                with amp.autocast(self.enable_amp):
                    logit_recon_batch, codes, mu, logvar = self.model(data)
                    valid_loss_rec, valid_loss_kld = self.loss_fnc(
                        logit_recon_batch, data, mu, logvar
                    )
                    valid_loss += (
                        self.lambda_rec * valid_loss_rec + valid_loss_kld
                    ).item()

                if callbacks:
                    logs["mu"] = mu.detach()

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

        if self.verbose and (self.comm_rank == 0):
            print("====> Validation loss: {:.4f}".format(valid_loss))

    def compute_losses(self, data_loader, checkpoint):
        self._load_checkpoint(checkpoint)
        self.model.eval()
        bce_losses, kld_losses, indices = [], [], []
        with torch.no_grad():
            for batch_idx, sample in enumerate(data_loader):
                data = sample["X"].to(self.device.encoder)

                with amp.autocast(self.enable_amp):
                    logit_recon_batch, codes, mu, logvar = self.model(data)
                    bce_loss, kld_loss = vae_logit_loss_outlier_helper(
                        logit_recon_batch, data, mu, logvar, self.lambda_rec
                    )

                bce_losses.append(bce_loss.item())
                kld_losses.append(kld_loss.item())
                indices.append(sample["index"])

        return bce_losses, kld_losses, indices

    def _load_checkpoint(self, path):
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
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        handle.encoder.load_state_dict(cp["encoder_state_dict"])
        handle.decoder.load_state_dict(cp["decoder_state_dict"])

        # optimizer
        self.optimizer.load_state_dict(cp["optimizer_state_dict"])
        return cp["epoch"]

    def encode(self, x):
        """
        Embed data into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Data to encode, could be a batch of data with dimension
            (batch_size, input_shape)

        Returns
        -------
        torch.Tensor of embeddings of shape (batch-size, latent_dim)

        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        return handle.encode(x)

    def decode(self, embedding):
        """
        Generate matrices from embeddings.

        Parameters
        ----------
        embedding : torch.Tensor
            Embedding data, could be a batch of data with dimension
            (batch-size, latent_dim)

        Returns
        -------
        torch.Tensor of generated matrices of shape (batch-size, input_shape)
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        return handle.decode(embedding)

    def save_weights(self, enc_path, dec_path):
        """
        Save encoder/decoder weights.

        Parameters
        ----------
        enc_path : str
            Path to save the encoder weights.

        dec_path : str
            Path to save the decoder weights.
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        handle.save_weights(enc_path, dec_path)

    def load_weights(self, enc_path, dec_path):
        """
        Load saved encoder/decoder weights.

        Parameters
        ----------
        enc_path : str
            Path to save the encoder weights.

        dec_path : str
            Path to save the decoder weights.
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        handle.load_weights(enc_path, dec_path)
