from pathlib import Path
from typing import Optional
from molecules.config import ModelBaseConfig


class ResnetVAEModelConfig(ModelBaseConfig):

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
    enc_kernel_size: int = 5
    latent_dim: int = 10
    activation: str = "ReLU"
    output_activation: str = "None"  # Identity function
    lambda_rec: float = 1.0
    enc_reslayers: Optional[int] = None
    scale_factor: int = 2
    dec_reslayers: int = 3
    dec_kernel_size: int = 5
    dec_filters: int = 66
    dec_filter_growth_rate: float = 1.0

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
    ResnetVAEModelConfig().dump_yaml("resnet_vae_template.yaml")
