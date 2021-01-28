from typing import List
from molecules.ml.hyperparams import Hyperparams


class AutoEncoderHyperparams(Hyperparams):
    def __init__(
        self,
        latent_dim: int = 3,
        affine_widths: List[int] = [128],
        affine_dropouts: List[float] = [0.0],
        activation: str = "ReLU",
        output_activation: str = "Sigmoid",
    ):

        self.latent_dim = latent_dim
        self.affine_widths = affine_widths
        self.affine_dropouts = affine_dropouts
        self.activation = activation
        self.output_activation = output_activation

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        if len(self.affine_dropouts) != len(self.affine_widths):
            raise ValueError("Must have same number of dropout values as layers.")
        if any(p < 0 or p > 1 for p in self.affine_dropouts):
            raise ValueError("Dropout probabilities, p, must be 0 <= p <= 1.")
