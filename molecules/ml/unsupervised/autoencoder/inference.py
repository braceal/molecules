import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
from molecules.ml.datasets import BasicDataset
from molecules.ml.unsupervised.autoencoder.autoencoder import Encoder
from molecules.ml.unsupervised.autoencoder.hyperparams import AutoEncoderHyperparams


def generate_embeddings(
    hparams_path: str,
    checkpoint_path: str,
    input_path: str,
    input_shape: Tuple[int],
    device: str,
    batch_size: int,
    dataset_name: str,
    scalar_dset_names: List[str] = [],
):

    hparams = AutoEncoderHyperparams().load(hparams_path)
    encoder = Encoder(input_shape, hparams, checkpoint_path)

    dataset = BasicDataset(
        path=input_path,
        dataset_name=dataset_name,
        scalar_dset_names=scalar_dset_names,
        split_ptc=1,
        split="train",
        seed=333,
    )

    # Put encoder on specified CPU/GPU
    encoder.to(device)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # Collect embeddings and associated index into simulation trajectory
    print("Generating embeddings")

    embeddings = []
    for sample in tqdm(data_loader):
        data = sample["X"].to(device)
        embeddings.append(encoder.encode(data).cpu().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings
