import shutil
import argparse
from pathlib import Path
from typing import Optional

from torchsummary import summary
import wandb
import torch
import torch.nn
from torch.utils.data import DataLoader

# molecules stuff
from molecules.ml.datasets import BasicDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import (
    LossCallback,
    CheckpointCallback,
    SaveEmbeddingsCallback,
    TSNEPlotCallback,
)
from molecules.ml.unsupervised.vae import VAE, BasicVAEHyperparams

from molecules.ml.unsupervised.vae.basic.config import BasicVAEModelConfig


def setup_wandb(
    cfg: BasicVAEModelConfig,
    model: torch.nn.Module,
    model_path: Path,
) -> Optional[wandb.config]:
    wandb_config = None
    if cfg.wandb_project_name is not None:
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity_name,
            name=cfg.model_tag,
            id=cfg.model_tag,
            dir=str(model_path),
            config=cfg.dict(),
            resume=False,
        )
        wandb_config = wandb.config
        # watch model
        wandb.watch(model)

    return wandb_config


def main(cfg: BasicVAEModelConfig, encoder_gpu: int, decoder_gpu: int):

    # Create output directory
    cfg.output_path.mkdir()
    # Copy training data to output directory to not slow down other
    # training processes using the same data.
    cfg.input_path = shutil.copy(cfg.input_path, cfg.output_path)

    hparams = BasicVAEHyperparams(
        latent_dim=cfg.latent_dim,
        affine_widths=cfg.affine_widths,
        affine_dropouts=cfg.affine_dropouts,
        activation=cfg.activation,
        output_activation=cfg.output_activation,
        lambda_rec=cfg.lambda_rec,
    )

    optimizer_hparams = OptimizerHyperparams(
        name=cfg.optimizer_name, hparams={"lr": cfg.optimizer_lr}
    )

    input_shape = (cfg.input_dim,)

    # Create model
    vae = VAE(
        input_shape=input_shape,
        hparams=hparams,
        optimizer_hparams=optimizer_hparams,
        gpu=(encoder_gpu, decoder_gpu),
        enable_amp=cfg.amp,
        init_weights=cfg.init_weights,
    )

    # Diplay model
    print(vae)
    # Only print summary when encoder_gpu is None or 0
    summary(vae.model, input_shape)

    # Load training and validation data
    # training
    train_dataset = BasicDataset(
        path=cfg.input_path,
        dataset_name=cfg.dataset_name,
        scalar_dset_names=cfg.scalar_dset_names,
        split_ptc=cfg.split_pct,
        split="train",
        seed=cfg.seed,
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
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        shuffle=cfg.shuffle,
        pin_memory=True,
        num_workers=cfg.num_data_workers,
    )

    wandb_config = setup_wandb(cfg, vae.model, cfg.output_path)

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
    vae.train(train_loader, valid_loader, cfg.epochs, callbacks=callbacks)

    # Save loss history to disk.
    loss_callback.save(cfg.output_path.joinpath("loss.json"))

    # Save final model weights to disk
    vae.save_weights(
        cfg.output_path.joinpath("encoder-weights.pt"),
        cfg.output_path.joinpath("decoder-weights.pt"),
    )

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
    parser.add_argument(
        "-E", "--encoder_gpu", help="GPU to place encoder", type=int, default=0
    )
    parser.add_argument(
        "-D", "--decoder_gpu", help="GPU to place decoder", type=int, default=0
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = BasicVAEModelConfig.from_yaml(args.config)
    main(cfg, args.encoder_gpu, args.decoder_gpu)
