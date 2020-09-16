import os
import re
import click
from os.path import join

# torch stuff
from torchsummary import summary
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

# mpi4py
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# molecules stuff
from molecules.ml.datasets import ContactMapDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import (LossCallback, CheckpointCallback,
                                    SaveEmbeddingsCallback, TSNEPlotCallback)
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams, ResnetVAEHyperparams

# hpo stuff
import ray
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLogger

def parse_dict(ctx, param, value):
    if value is not None:
        token = value.split(",")
        result = {}
        for item in token:
            k, v = item.split("=")
            result[k] = v
        return result

@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-dn', '--dataset_name', default='contact_map',
              help='Name of the dataset in the HDF5 file.')

@click.option('-rn', '--rmsd_name', default='rmsd',
              help='Name of the RMSD data in the HDF5 file.')

@click.option('-fn', '--fnc_name', default='fnc',
              help='Name of the fraction of contacts data in the HDF5 file.')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_prefix', required=True,
              help='Model prefix for file naming.')

@click.option('-h', '--dim1', required=True, type=int,
              help='H of (H,W) shaped contact matrix')

@click.option('-w', '--dim2', required=True, type=int,
              help='W of (H,W) shaped contact matrix')

@click.option('-c', '--checkpoint', default=None,
             type=click.Path(exists=True),
             help='Model checkpoint file to resume training. ' \
                  'Checkpoint files saved as .pt by CheckpointCallback.')

@click.option('-r', '--resume', is_flag=True,
              help='Resume from latest checkpoint')

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Format of contact map files. Options ' \
                   '[full, sparse-concat, sparse-rowcol]')

@click.option('-E', '--encoder_gpu', default=None, type=int,
              help='Encoder GPU id')

@click.option('-D', '--decoder_gpu', default=None, type=int,
              help='Decoder GPU id')

@click.option('-e', '--epochs', default=10, type=int,
              help='Number of epochs to train for')

@click.option('-lw', '--loss_weights', callback=parse_dict,
              help='Loss parameters')

@click.option('-t', '--model_type', default='resnet',
              help='Model architecture option: [resnet, symmetric]')

@click.option('-ei', '--embed_interval', default=1, type=int,
              help="Saves embedddings every interval'th point")

@click.option('-ti', '--tsne_interval', default=1, type=int,
              help='Saves model checkpoints, embedddings, tsne plots every ' \
                   "interval'th point")

@click.option('-S', '--sample_interval', default=20, type=int,
              help="For embedding plots. Plots every sample_interval'th point")

@click.option('-a', '--amp', is_flag=True,
              help='Specify if we want to enable automatic mixed precision (AMP)')

@click.option('--distributed', is_flag=True,
              help='Enable distributed training')

@click.option('--local_rank', default=None, type=int,
              help='Local rank on the machine, required for DDP')

@click.option('-wp', '--wandb_project_name', default=None,
              help='Project name for wandb logging')

@click.option('-wp', '--wandb_api_key', default=None,
              help='API key for wandb logging')

def main(input_path, dataset_name, rmsd_name, fnc_name, out_path, checkpoint, resume, model_prefix, dim1, dim2,
         cm_format, encoder_gpu, decoder_gpu, epochs, loss_weights, model_type,
         embed_interval, tsne_interval, sample_interval, amp, distributed, local_rank, wandb_project_name, wandb_api_key):
    
    # init raytune
    ray.init()

    tune_config = {
        'input_path': input_path,
        'dataset_name': dataset_name,
        'rmsd_name': rmsd_name,
        'fnc_name': fnc_name,
        'out_path': out_path,
        'checkpoint': checkpoint,
        'resume': resume,
        'model_prefix': model_prefix,
        'dim1': dim1,
        'dim2': dim2,
        'cm_format': cm_format,
        'encoder_gpu': encoder_gpu,
        'decoder_gpu': encoder_gpu,
        'epochs': epochs,
        'batch_size': hp.choice('batch_size', [4, 8, 16, 32]),
        'optimizer': {
            'name': 'Adam', 
            'lr': hp.loguniform('lr', 1e-5, 1e-1)
        },
        'latent_dim': hp.choice('latent_dim', [64, 128, 256]),
        'scale_factor': hp.choice('scale_factor', [2, 4]),
        'loss_weights': {key: float(val) for key, val in loss_weights.items()},
        'model_type': model_type,
        'embed_interval': embed_interval,
        'tsne_interval': tsne_interval,
        'sample_interval': sample_interval,
        'amp': amp,
        'distributed': distributed,
        'local_rank': local_rank,
        'wandb': {
            'project': wandb_project_name,
            'api_key': wandb_api_key
        }
    }

    tune_config_good = tune_config.copy()
    # we need to feed the indices here for choice args
    tune_config_good['batch_size'] = 3
    tune_config_good['optimizer']['lr'] = 1e-4
    tune_config_good['latent_dim'] = 2
    tune_config_good['scale_factor'] = 0

    hyperopt_search = HyperOptSearch(tune_config, points_to_evaluate = [tune_config_good],
                                     max_concurrent=1, metric='loss_eg', mode='min')

    analysis = tune.run(run_config, 
                        loggers=[WandbLogger], 
                        resources_per_trial={'gpu': len(set(encoder_gpu, decoder_gpu))}, 
                        num_samples=20, 
                        search_alg=hyperopt_search)

    # goodbye
    ray.shutdown()


# main function, called by raytune
@wandb_mixin
def run_config(config):
    """Example for training Fs-peptide with either Symmetric or Resnet VAE."""

    # get parameters from config
    input_path = config['input_path']
    dataset_name = config['dataset_name']
    rmsd_name = config['rmsd_name']
    fnc_name = config['fnc_name']
    out_path = config['out_path']
    checkpoint = config['checkpoint']
    resume = config['resume']
    model_prefix = config['model_prefix']
    dim1 = config['dim1']
    dim2 = config['dim2']
    cm_format = config['cm_format']
    encoder_gpu = config['encoder_gpu']
    decoder_gpu = config['decoder_gpu']
    epochs = config['epochs']
    batch_size = config['batch_size']
    optimizer = config['optimizer']
    latent_dim = config['latent_dim']
    loss_weights = config['loss_weights']
    model_type = config['model_type']
    embed_interval = config['embed_interval']
    tsne_interval = config['tsne_interval']
    sample_interval = config['sample_interval']
    amp = config['amp']
    distributed = config['distributed']
    local_rank = config['local_rank']
    wandb_project_name = config['wandb']['project']
    wandb_api_key = config['wandb']['api_key']
    
    # do some scaffolding for DDP
    comm_rank = 0
    comm_size = 1
    comm_local_rank = 0
    comm = None
    if distributed and dist.is_available():
        # init mpi4py:
        MPI.Init_thread()

        # get communicator: duplicate from comm world
        comm = MPI.COMM_WORLD.Dup()

        # now match ranks between the mpi comm and the nccl comm
        os.environ["WORLD_SIZE"] = str(comm.Get_size())
        os.environ["RANK"] = str(comm.Get_rank())

        # init torch distributed
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        comm_rank = dist.get_rank()
        comm_size = dist.get_world_size()
        if local_rank is not None:
            comm_local_rank = local_rank
        else:
            comm_local_rank = int(os.getenv("LOCAL_RANK", 0))


    assert model_type in ['symmetric', 'resnet']

    # Note: See SymmetricVAEHyperparams, ResnetVAEHyperparams class definitions
    #       for hyperparameter options. 

    if model_type == 'symmetric':
        # Optimal Fs-peptide params
        fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                             'kernels': [5, 5, 5, 5],
                             'strides': [1, 2, 1, 1],
                             'affine_widths': [64],
                             'affine_dropouts': [0],
                             'latent_dim': latent_dim,
                             'output_activation': 'None'}

        input_shape = (1, dim1, dim2)
        hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)

    elif model_type == 'resnet':
        resnet_hparams = {'max_len': dim1,
                          'nchars': dim2,
                          'latent_dim': latent_dim,
                          'dec_filters': dim1,
                          'scale_factor': scale_factor,
                          'lambda_rec': loss_weights['lambda_rec'],
                          'output_activation': 'None'}

        input_shape = (dim1, dim1)
        hparams = ResnetVAEHyperparams(**resnet_hparams)

    optimizer_hparams = OptimizerHyperparams(name=optimizer['name'], hparams={'lr': float(optimizer["lr"])})

    vae = VAE(input_shape, hparams, optimizer_hparams,
              gpu=(encoder_gpu, decoder_gpu), enable_amp=amp)

    enc_device = torch.device(f'cuda:{encoder_gpu}')
    dec_device = torch.device(f'cuda:{decoder_gpu}')
    if comm_size > 1:
        if (encoder_gpu == decoder_gpu):
            vae.model = DDP(vae.model, device_ids = [enc_device], output_device = enc_device)
        else:
            vae.model = DDP(vae.model, device_ids = None, output_device = None)

    # set global default device
    torch.cuda.set_device(enc_device.index)

    # Diplay model
    if comm_rank == 0:
        print(vae)
        # Only print summary when encoder_gpu is None or 0
        #summary(vae.model, input_shape)

    # Load training and validation data
    # training
    train_dataset = ContactMapDataset(input_path,
                                      dataset_name,
                                      rmsd_name, 
                                      fnc_name,
                                      input_shape,
                                      split='train',
                                      cm_format=cm_format)

    # split across nodes
    if comm_size > 1:
        chunksize = len(train_dataset) // comm_size
        train_dataset = Subset(train_dataset,
                               list(range(chunksize * comm_rank, chunksize * (comm_rank + 1))))
    
    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              drop_last = True,
                              shuffle = True,
                              pin_memory = True,
                              num_workers = 0)

    # validation
    valid_dataset = ContactMapDataset(input_path,
                                      dataset_name,
                                      rmsd_name,
                                      fnc_name,
                                      input_shape,
                                      split='valid',
                                      cm_format=cm_format)

    # split across nodes
    if comm_size > 1:
        chunksize = len(valid_dataset) // comm_size
        valid_dataset = Subset(valid_dataset,
                               list(range(chunksize * comm_rank, chunksize * (comm_rank + 1))))
    
    valid_loader = DataLoader(valid_dataset,
                              batch_size = batch_size,
                              drop_last = True,
                              shuffle = True,
                              pin_memory = True,
                              num_workers = 0)

    ## we call next once here to make sure the data is pinned to the right GPU
    #with torch.cuda.device(enc_device.index):
    #    _ = next(train_loader)
    #    _ = valid_loader.next()

    # For ease of training multiple models
    model_path = join(out_path, f'model-{model_prefix}')
    os.makedirs(model_path, exist_ok=True)

    # do we want wandb
    wandb_config = None
    if (comm_rank == 0) and (wandb_project_name is not None):
        import wandb
        wandb.init(project = wandb_project_name,
                   name = model_prefix,
                   id = model_prefix,
                   dir = model_path,
                   resume = False,
                   config=config)
        wandb_config = wandb.config
        
        # watch model
        wandb.watch(vae.model)
    
    # Optional callbacks
    loss_callback = LossCallback(join(model_path, 'loss.json'),
                                 wandb_config=wandb_config,
                                 mpi_comm=comm)
    
    checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'),
                                             mpi_comm=comm)

    save_callback = SaveEmbeddingsCallback(out_dir=join(model_path, 'embedddings'),
                                           interval=embed_interval,
                                           sample_interval=sample_interval,
                                           mpi_comm=comm)

    # TSNEPlotCallback requires SaveEmbeddingsCallback to run first
    tsne_callback = TSNEPlotCallback(out_dir=join(model_path, 'embedddings'),
                                     projection_type='3d',
                                     target_perplexity=100,
                                     colors=['rmsd', 'fnc'],
                                     interval=tsne_interval,
                                     wandb_config=wandb_config,
                                     mpi_comm=comm)

    # Train model with callbacks
    callbacks = [loss_callback, checkpoint_callback, save_callback, tsne_callback]

    # see if resume is set
    if resume and (checkpoint is None):
        clist = [x for x in os.listdir(join(model_path, 'checkpoint')) if x.endswith(".pt")]
        checkpoints = sorted(clist, key=lambda x: re.match("epoch-\d*?-(\d*?-\d*?).pt", x).groups()[0])
        if checkpoints:
            checkpoint = join(model_path, 'checkpoint', checkpoints[-1])
            if comm_rank == 0:
                print(f"Resuming from checkpoint {checkpoint}.")
        else:
            if comm_rank == 0:
                print(f"No checkpoint files in directory {join(model_path, 'checkpoint')}, \
                       cannot resume training, will start from scratch.")
    
    # create model
    vae.train(train_loader, valid_loader, epochs,
              checkpoint=checkpoint, callbacks=callbacks)

    if comm_rank == 0:
        # Save loss history to disk.
        loss_callback.save(join(model_path, 'loss.json'))

        # Save hparams to disk
        hparams.save(join(model_path, 'model-hparams.json'))
        optimizer_hparams.save(join(model_path, 'optimizer-hparams.json'))

        # Save final model weights to disk
        vae.save_weights(join(model_path, 'encoder-weights.pt'),
                         join(model_path, 'decoder-weights.pt'))

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
    # |   |__ wandb_cache/

if __name__ == '__main__':
    main()
