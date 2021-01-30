import shutil
import argparse

from torchsummary import summary
from torch.utils.data import DataLoader

# molecules stuff
from molecules.utils import setup_wandb
from molecules.ml.datasets import ContactMapDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import (
    LossCallback,
    CheckpointCallback,
    SaveEmbeddingsCallback,
    TSNEPlotCallback,
)
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams
from molecules.ml.unsupervised.vae.symmetric.config import SymmetricVAEModelConfig


def main(cfg: SymmetricVAEModelConfig, encoder_gpu: int, decoder_gpu: int):

    # Create output directory
    cfg.output_path.mkdir()
    # Copy training data to output directory to not slow down other
    # training processes using the same data.
    cfg.input_path = shutil.copy(cfg.input_path, cfg.output_path)

    hparams = SymmetricVAEHyperparams(
        filters=cfg.filters,
        kernels=cfg.kernels,
        strides=cfg.strides,
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

    input_shape = (1, cfg.dim1, cfg.dim2)

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
    train_dataset = ContactMapDataset(
        cfg.input_path,
        cfg.dataset_name,
        scalar_dset_names=[cfg.rmsd_name, cfg.fnc_name],
        shape=input_shape,
        seed=cfg.seed,
        split="train",
        split_ptc=cfg.split_pct,
        cm_format=cfg.cm_format,
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
    valid_dataset = ContactMapDataset(
        cfg.input_path,
        cfg.dataset_name,
        scalar_dset_names=[cfg.rmsd_name, cfg.fnc_name],
        shape=input_shape,
        seed=cfg.seed,
        split="valid",
        split_ptc=cfg.split_pct,
        cm_format=cfg.cm_format,
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
    )

    # TSNEPlotCallback requires SaveEmbeddingsCallback to run first
    tsne_callback = TSNEPlotCallback(
        out_dir=cfg.output_path.joinpath("embeddings"),
        projection_type="3d",
        target_perplexity=100,
        colors=["rmsd", "fnc"],
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
    cfg = SymmetricVAEModelConfig.from_yaml(args.config)
    main(cfg, args.encoder_gpu, args.decoder_gpu)
