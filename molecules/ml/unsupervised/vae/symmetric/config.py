from pathlib import Path
from typing import Optional, List
from molecules.config import ModelBaseConfig


class SymmetricVAEModelConfig(ModelBaseConfig):

    # File paths
    # Path to HDF5 training file
    input_path: Path = Path("TODO")
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path("TODO")

    # Name of the dataset in the HDF5 file.
    dataset_name: str = "contact_map"
    # Name of the RMSD data in the HDF5 file.
    rmsd_name: str = "rmsd"
    # Name of the fraction of contacts data in the HDF5 file.
    fnc_name: str = "fnc"
    # Format of contact maps
    cm_format: str = "sparse-concat"
    # Fraction of training data to use
    split_pct: float = 0.8
    # Random seed for shuffling train/validation data
    seed: int = 333
    # Whether or not to shuffle train/validation data
    shuffle: bool = True
    # First dimension of contact map
    dim1: int = 66
    # Second dimension of contact map
    dim2: int = 66
    # Number of epochs to train
    epochs: int = 10
    # Training batch size
    batch_size: int = 64
    # Whether to train with automatic mixed precision
    amp: bool = False
    # Pretrained model weights
    init_weights: Optional[str] = None

    # Optimizer params
    # PyTorch Optimizer name
    optimizer_name: str = "Adam"
    # Learning rate
    optimizer_lr: float = 0.0001

    # Model hyperparameters
    latent_dim: int = 64
    filters: List[int] = [100, 100, 100, 100]
    kernels: List[int] = [5, 5, 5, 5]
    strides: List[int] = [1, 2, 1, 1]
    latent_dim: int = 10
    affine_widths: List[int] = [64]
    affine_dropouts: List[float] = [0.0]
    activation: str = "ReLU"
    output_activation: str = "None"  # Identity function
    lambda_rec: float = 1.0

    # Training settings
    # Saves embeddings every embed_interval'th epoch
    embed_interval: int = 1
    # Saves tsne plots every tsne_interval'th epoch
    tsne_interval: int = 5
    # For saving and plotting embeddings. Saves len(validation_set) / sample_interval points.
    sample_interval: int = 20
    # Specify t-SNE plotting backend (plotly or mpl)
    plot_backend: str = "plotly"
    # Number of data loaders for training
    num_data_workers: int = 0


if __name__ == "__main__":
    SymmetricVAEModelConfig().dump_yaml("symmetric_vae_template.yaml")
