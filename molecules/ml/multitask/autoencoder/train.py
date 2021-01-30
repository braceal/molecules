import shutil
import argparse

from torchsummary import summary
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

# molecules stuff
from molecules.utils import setup_wandb
from molecules.ml.datasets import BasicDataset
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer
from molecules.ml.callbacks import (
    LossCallback,
    CheckpointCallback,
    SaveEmbeddingsCallback,
    TSNEPlotCallback,
)
from molecules.ml.unsupervised.autoencoder.autoencoder import AutoEncoder
from molecules.ml.unsupervised.autoencoder.hyperparams import AutoEncoderHyperparams
from molecules.ml.multitask.autoencoder.autoencoder import (
    MultiTaskModel,
    train,
    RMSDNet,
)
from molecules.ml.multitask.autoencoder.config import MultiTaskAutoEncoderModelConfig


def select_loss_function(loss_function: str, reduction: str):
    if loss_function == "bce":

        def bce_loss_function(recon_x, x):
            BCE = F.binary_cross_entropy(recon_x, x, reduction=reduction)
            return BCE

        return bce_loss_function
    elif loss_function == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"loss function {loss_function} not supported")


def main(cfg: MultiTaskAutoEncoderModelConfig):

    # Create output directory
    cfg.output_path.mkdir()
    # Copy training data to output directory to not slow down other
    # training processes using the same data.
    cfg.input_path = shutil.copy(cfg.input_path, cfg.output_path)

    hparams = AutoEncoderHyperparams(
        latent_dim=cfg.latent_dim,
        affine_widths=cfg.affine_widths,
        affine_dropouts=cfg.affine_dropouts,
        activation=cfg.activation,
        output_activation=cfg.output_activation,
    )

    input_shape = (cfg.input_dim,)

    # Create autoencoder model
    autorencoder_model = AutoEncoder(
        input_shape=input_shape,
        hparams=hparams,
        init_weights=cfg.init_weights,
    )
    criterion = select_loss_function(cfg.loss_function, cfg.loss_reduction)

    # Create multitask model heads
    rmsd_criterion = select_loss_function(
        cfg.rmsd_loss_function, cfg.rmsd_loss_reduction
    )
    rmsd_model = RMSDNet(rmsd_criterion, scalar_name="rmsd", input_dim=cfg.latent_dim)

    multitask_model = MultiTaskModel(autorencoder_model, heads=[rmsd_model])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multitask_model.to(device)

    # Diplay model
    print(multitask_model)
    # Only print summary when encoder_gpu is None or 0
    summary(multitask_model, input_shape)

    optimizer_hparams = OptimizerHyperparams(
        name=cfg.optimizer.name,
        hparams=cfg.optimizer.hparams,
    )
    optimizer = get_optimizer(multitask_model.parameters(), optimizer_hparams)

    # Load training and validation data
    # training
    train_dataset = BasicDataset(
        path=cfg.input_path,
        dataset_name=cfg.dataset_name,
        scalar_dset_names=cfg.scalar_dset_names,
        split_ptc=cfg.split_pct,
        split="train",
        seed=cfg.seed,
        scalar_requires_grad=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        shuffle=cfg.shuffle,
        pin_memory=True,
        num_workers=cfg.num_data_workers,
    )

    # validation
    valid_dataset = BasicDataset(
        path=cfg.input_path,
        dataset_name=cfg.dataset_name,
        scalar_dset_names=cfg.scalar_dset_names,
        split_ptc=cfg.split_pct,
        split="valid",
        seed=cfg.seed,
        scalar_requires_grad=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        shuffle=cfg.shuffle,
        pin_memory=True,
        num_workers=cfg.num_data_workers,
    )

    wandb_config = setup_wandb(cfg, multitask_model, cfg.output_path)

    # Optional callbacks
    loss_callback = LossCallback(
        cfg.output_path.joinpath("loss.json"), wandb_config=wandb_config
    )

    checkpoint_callback = CheckpointCallback(
        out_dir=cfg.output_path.joinpath("checkpoint")
    )

    save_callback = SaveEmbeddingsCallback(
        out_dir=cfg.output_path.joinpath("embeddings"),
        interval=cfg.embed_interval,
        sample_interval=cfg.sample_interval,
        scalar_dset_names=cfg.scalar_dset_names,
    )

    # TSNEPlotCallback requires SaveEmbeddingsCallback to run first
    tsne_callback = TSNEPlotCallback(
        out_dir=cfg.output_path.joinpath("embeddings"),
        projection_type="3d",
        target_perplexity=100,
        colors=cfg.scalar_dset_names,
        tsne_is_blocking=True,
        interval=cfg.tsne_interval,
        plot_backend=cfg.plot_backend,
        wandb_config=wandb_config,
    )

    # Train model with callbacks
    callbacks = [loss_callback, checkpoint_callback, save_callback, tsne_callback]

    # Save hparams to disk before training
    hparams.save(cfg.output_path.joinpath("model-hparams.json"))
    optimizer_hparams.save(cfg.output_path.joinpath("optimizer-hparams.json"))

    # create model
    train(
        multitask_model,
        optimizer,
        device,
        train_loader,
        valid_loader,
        criterion,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=True,
    )

    # Save loss history to disk.
    loss_callback.save(cfg.output_path.joinpath("loss.json"))

    # Output directory structure
    #  out_path
    # ├── model_path
    # │   ├── checkpoint
    # │   │   ├── epoch-1-20200606-125334.pt
    # │   │   └── epoch-2-20200606-125338.pt
    # │   ├── decoder-weights.pt
    # │   ├── encoder-weights.pt
    # │   ├── loss.json
    # │   ├── model-hparams.json
    # │   └── optimizer-hparams.json
    # |   |__ wandb/


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = MultiTaskAutoEncoderModelConfig.from_yaml(args.config)
    main(cfg)
