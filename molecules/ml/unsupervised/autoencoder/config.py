from pathlib import Path
from typing import Optional, List, Dict, Any
from molecules.config import BaseSettings


class OptimizerConfig(BaseSettings):
    class Config:
        extra = "allow"

    name: str = "Adam"
    hparams: Dict[str, Any] = {}


class AutoEncoderModelConfig(BaseSettings):

    # File paths
    # Path to HDF5 training file
    input_path: Path = Path("TODO")
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path("TODO")

    # Name of the training dataset in the HDF5 file.
    dataset_name: str = "feature"
    # Name of the RMSD data in the HDF5 file.
    scalar_dset_names: List[str] = ["rmsd"]
    # Fraction of training data to use
    split_pct: float = 0.8
    # Random seed for shuffling train/validation data
    seed: int = 333
    # Whether or not to shuffle train/validation data
    shuffle: bool = True
    # Dimension of input training feature vector
    input_dim: int = 40
    # Number of epochs to train
    epochs: int = 10
    # Training batch size
    batch_size: int = 64
    # Pretrained model weights
    init_weights: Optional[str] = None

    # Optimizer params
    # PyTorch Optimizer name
    optimizer: OptimizerConfig = OptimizerConfig()

    # Model hyperparameters
    latent_dim: int = 64
    affine_widths: List[int] = [64]
    affine_dropouts: List[float] = [0.0]
    activation: str = "ReLU"
    output_activation: str = "Sigmoid"
    loss_function: str = "mse"

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
    # Project name for wandb logging
    wandb_project_name: Optional[str] = None
    # Team name for wandb logging
    wandb_entity_name: Optional[str] = None
    # Model tag for wandb labeling
    model_tag: Optional[str] = None


if __name__ == "__main__":
    AutoEncoderModelConfig().dump_yaml("autoencoder_template.yaml")
