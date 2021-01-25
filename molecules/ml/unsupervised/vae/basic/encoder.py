from typing import Optional, Tuple
import torch
from torch import nn
from math import isclose
from molecules.ml.unsupervised.utils import get_activation, _init_weights
from molecules.ml.unsupervised.vae.basic import BasicVAEHyperparams


class BasicEncoder(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int],
        hparams: BasicVAEHyperparams,
        init_weights: Optional[str] = None,
    ):
        super(BasicEncoder, self).__init__()

        assert isinstance(hparams, BasicVAEHyperparams)
        hparams.validate()

        self.hparams = hparams
        self.input_dim = input_dim[0]  # D dimensions

        self.encoder = nn.Sequential(*self._affine_layers())

        self.mu = self._embedding_layer()
        self.logvar = self._embedding_layer()

        self.init_weights(init_weights)

    def init_weights(self, init_weights: Optional[str]):
        if init_weights is None:
            self.encoder.apply(_init_weights)
            _init_weights(self.mu)
            _init_weights(self.logvar)
        # Loading checkpoint weights
        elif init_weights.endswith(".pt"):
            checkpoint = torch.load(init_weights, map_location="cpu")
            self.load_state_dict(checkpoint["encoder_state_dict"])

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(x)[0]

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

        for width, dropout in zip(
            self.hparams.affine_widths, self.hparams.affine_dropouts
        ):

            layers.append(nn.Linear(in_features=in_features, out_features=width))

            layers.append(get_activation(self.hparams.activation))

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        return layers

    def _embedding_layer(self):
        return nn.Linear(
            in_features=self.hparams.affine_widths[-1],
            out_features=self.hparams.latent_dim,
        )
