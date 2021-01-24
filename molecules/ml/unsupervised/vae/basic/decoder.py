from typing import Optional
import torch
from torch import nn
from math import isclose
from molecules.ml.unsupervised.utils import (
    get_activation,
    _init_weights,
)
from molecules.ml.unsupervised.vae.basic import BasicVAEHyperparams


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


class BasicDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hparams: BasicVAEHyperparams,
        init_weights: Optional[str] = None,
    ):
        super(BasicDecoder, self).__init__()

        assert isinstance(hparams, BasicVAEHyperparams)
        hparams.validate()

        self.output_dim = output_dim
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

        # Add last layer with dims to connect the last linear layer to
        # the first convolutional decoder layer
        layers.append(
            nn.Linear(
                in_features=self.hparams.affine_widths[0],
                out_features=self.output_dim,
            )
        )
        layers.append(act)

        return layers
