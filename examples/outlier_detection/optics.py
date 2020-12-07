import os
import re
import json
import itertools
import time
import click
import shutil
import torch
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
import MDAnalysis as mda
from torch.utils.data import DataLoader, Subset
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from molecules.utils import open_h5
from molecules.ml.unsupervised.cluster import optics_clustering

# plotting
import matplotlib.pyplot as plt

# Helper function for LocalOutlierFactor
def topk(a, k):
    """
    Parameters
    ----------
    a : np.ndarray
        array of dim (N,)
    k : int
        specifies which element to partition upon
    Returns
    -------
    np.ndarray of length k containing indices of input array a
    coresponding to the k smallest values in a.
    """
    return np.argpartition(a, k)[:k]

def parse_list(ctx, param, value):
    """Parse click CLI list "item1,item2,item3..."."""
    if value is not None:
        return value.split(',')


def model_selection_loss_based(model_paths, num_select=1, comm=None):
    """
    Parameters
    ----------
    model_paths : list
        list of paths pointing to a directory structured like below.

    num_select : int
        Number of best models to select. NOT IMPLEMENTED

    Note:
    model_path directory structure
        ├── model_path
        │   ├── checkpoint
        │   │   ├── epoch-1-20200606-125334.pt
        │   │   └── epoch-2-20200606-125338.pt
        │   ├── decoder-weights.pt
        │   ├── encoder-weights.pt
        │   ├── loss.json
        │   ├── model-hparams.json
        │   └── optimizer-hparams.json
        |   |__ wandb_cache/
    """
    # start timer
    t_start = time.time()

    # communicator stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    # Best model checkpoint file for each HPO run and
    # associated validation loss 
    best_checkpoints, best_valid_losses, hparams = [], [], []

    # Loop over all model paths in hyperparameter optimization search.
    for model_path in model_paths:
        
        with open(join(model_path, 'loss.json')) as loss_file:
            loss_log = json.load(loss_file)
        valid_loss, epochs = loss_log['valid_loss'], loss_log['epochs']
        # Need to index into epoch, since run could have been resumed from
        # a checkpoint, meaning that epochs does not start at 1.
        best_ind = np.nanargmin(valid_loss)
        best_valid_loss = valid_loss[best_ind]
        best_epoch = str(epochs[best_ind])
        
        checkpoint_dir = join(model_path, 'checkpoint')
        # Format: epoch-1-20200606-125334.pt
        for checkpoint in os.listdir(checkpoint_dir):
            if checkpoint.split('-')[1] == best_epoch:
                best_checkpoint = checkpoint
                break

        # Colect model checkpoints associated with minimal validation loss
        best_checkpoints.append(join(checkpoint_dir, best_checkpoint))
        best_valid_losses.append(best_valid_loss)
        hparams.append(join(model_path, 'model-hparams.json'))

    best_model = np.argmin(best_valid_losses)
    best_checkpoint = best_checkpoints[best_model]
    best_hparams = hparams[best_model]

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("Model Selection Time: {}s".format(t_end - t_start))

    return best_hparams, best_checkpoint


def model_selection_embedding_loss_based(model_paths, top_variance_threshold=0.1, num_select=1, comm=None):
    # start timer
    t_start = time.time()

    # communicator stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    # Best model checkpoint file for each HPO run and
    # associated validation loss
    results = []

    # create a list of models to work on:
    epochrule = re.compile(r"^embeddings-epoch-(\d{1,}?)-\d{1,}?-\d{1,}?.h5$")
    model_files = []
    for model_path in model_paths:
        try:
            model_files.append(pd.DataFrame([{"model_path": model_path, "epoch": int(epochrule.match(x).groups()[0]), "embeddings_file": x}
                                             for x in os.listdir(os.path.join(model_path, 'embeddings'))
                                             if x.endswith(".h5")]))
        except:
            print(f"Error, {model_path} does not contain embedding directories, skipping.")
            continue
    
    # create df
    modelsdf = pd.concat(model_files)

    # shard the data
    fullsize = modelsdf.shape[0]
    chunksize = fullsize // comm_size
    start = chunksize * comm_rank
    end = start + chunksize
    remainderdf = modelsdf.iloc[comm_size * chunksize:]
    modelsdf = modelsdf.iloc[start:end]
    for idx, (ind, row) in enumerate(remainderdf.iterrows()):
        if idx == comm_rank:
            modelsdf = modelsdf.append(row)

    # Loop over all model paths in hyperparameter optimization search.
    for model_path, item in modelsdf.groupby("model_path"):
        # load losses
        with open(join(model_path, 'loss.json')) as loss_file:
            loss_log = json.load(loss_file)

        # find all embeddings:
        for _, row in item.iterrows():
            
            # extract
            epoch = row["epoch"]
            embfile = row["embeddings_file"]

            with open_h5(os.path.join(model_path, 'embeddings', embfile), 'r', swmr = False) as f:
                embeddings = f["embeddings"][...]
                fnc = f["fnc"][...]
                rmsd = f["rmsd"][...]

            # check for nan
            if np.any(np.isnan(embeddings)):
                print(f"Error, {embfile} includes NaN values, skipping.")
                continue

            # run knn on embeddings
            knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
            nbrs = knn.fit(embeddings)
            distances, indices = nbrs.kneighbors(embeddings)
            # compute average rmsd for group of points
            rmsd_variance = np.var(np.mean(rmsd[indices], axis=1))
            fnc_variance = np.var(np.mean(fnc[indices], axis=1))
            # add to list
            records = {"rmsd_variance": rmsd_variance, "fnc_variance": fnc_variance, 
                       "loss": loss_log["valid_loss"][epoch - 1], 
                       "epoch": epoch, "model_path": model_path}
            results.append(records)

    # sort the list
    if results:
        resultdf = pd.DataFrame(results)
    else:
        resultdf = pd.DataFrame(columns=["rmsd_variance", "fnc_variance",
                                         "loss", "epoch", "model_path"])

    if comm_size > 1:
        resultdf = comm.allgather(resultdf)
        resultdf = pd.concat(resultdf).reset_index(drop=True)

    if comm_rank == 0:
        # select top N% variances
        nselect = int(np.ceil(top_variance_threshold * resultdf.shape[0]))
        resultdf = resultdf.nlargest(nselect, "rmsd_variance")

        # select minimum loss
        resultdf = resultdf.nsmallest(1, "loss")

        # extract the winner
        best_model_path = resultdf["model_path"].values[0]
        best_epoch = resultdf["epoch"].values[0]

        # generate model checkpoint string
        checkpoint_path = os.path.join(best_model_path, 'checkpoint')
        best_checkpoint = None
        for cp in os.listdir(checkpoint_path):
            match_string = r"^(epoch-{}-).*?-.*?pt$".format(best_epoch)
            match = re.match(match_string, cp)
            if match is not None:
                best_checkpoint = cp
                break

        # now we have everything
        best_hparams = os.path.join(best_model_path, 'model-hparams.json')
        best_checkpoint = os.path.join(best_model_path, "checkpoint", best_checkpoint)
    else:
        best_hparams = None
        best_checkpoint = None

    if comm_size > 1:
        best_hparams = comm.bcast(best_hparams, 0)
        best_checkpoint = comm.bcast(best_checkpoint, 0)

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("Model Selection Time: {}s".format(t_end - t_start))
        
    return best_hparams, best_checkpoint


def generate_embeddings(model_type, hparams_path, checkpoint_path, dim1, dim2,
                        device, input_path, cm_format, batch_size, comm = None):

    # start timer
    t_start = time.time()

    # communicator stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    if model_type == 'vae-resnet':
        from molecules.ml.datasets import ContactMapDataset
        from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams, ResnetEncoder
        # Initialize encoder model
        input_shape = (dim1, dim1)
        hparams = ResnetVAEHyperparams().load(hparams_path)
        encoder = ResnetEncoder(input_shape, hparams, checkpoint_path)

        dataset = ContactMapDataset(input_path,
                                    'contact_map',
                                    'rmsd', 'fnc',
                                    input_shape,
                                    split='train',
                                    split_ptc=1.,
                                    cm_format=cm_format)
    elif model_type == 'aae':
       from molecules.ml.datasets import PointCloudDataset
       from molecules.ml.unsupervised.point_autoencoder import AAE3dHyperparams
       from molecules.ml.unsupervised.point_autoencoder.aae import Encoder

       hparams = AAE3dHyperparams().load(hparams_path)
       encoder = Encoder(dim1, hparams.num_features, hparams, checkpoint_path)

       dataset = PointCloudDataset(input_path,
                                   'point_cloud',
                                   'rmsd',
                                   'fnc',
                                   dim1,
                                   hparams.num_features,
                                   split='train',
                                   split_ptc=1.,
                                   normalize='box',
                                   cms_transform=False)
    
    # shard the dataset
    if comm_size > 1:
        fullsize = len(dataset)
        chunksize = fullsize // comm_size
        start = comm_rank * chunksize
        end = start + chunksize
        subset_indices = list(range(start, end))
        # deal with remainder
        for idx, i in enumerate(range(comm_size * chunksize, fullsize)):
            if idx == comm_rank:
                subset_indices.append(i)
        # split the set
        dataset = Subset(dataset, subset_indices)

    # Put encoder on specified CPU/GPU
    encoder.to(device)

    # create data loader
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0)

    # Collect embeddings and associated index into simulation trajectory
    if comm_rank == 0:
        print("Generating embeddings")
    embeddings, indices = [], []
    for i, (data, rmsd, fnc, index) in enumerate(data_loader):
        data = data.to(device)
        embeddings.append(encoder.encode(data).cpu().numpy())
        indices.append(index)
        if (i % 100 == 0) and (comm_rank == 0):
            print(f'Batch {i}/{len(data_loader)}')
        
    if comm_size > 1:
        # gather results
        embeddings = comm.allgather(embeddings)
        embeddings = list(itertools.chain.from_iterable(embeddings))
        indices = comm.allgather(indices)
        indices = list(itertools.chain.from_iterable(indices))
    
    # concatenate
    embeddings = np.concatenate(embeddings)
    indices = np.concatenate(indices)

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("Generating Embeddings Time: {}s".format(t_end - t_start))

    return embeddings, indices


def perform_clustering(eps_path, encoder_weight_path, cm_embeddings, min_samples, eps):
    # TODO: if we decide on OPTICS clustering, then remove all eps code


    # If previous epsilon values have been calculated, load the record from disk.
    # eps_record stores a dictionary from cvae_weight path which uniquely identifies
    # a model, to the epsilon value previously calculated for that model.
    with open(eps_path) as file:
        try:
            eps_record = json.load(file)
        except json.decoder.JSONDecodeError:
            eps_record = {}

    best_eps = eps_record.get(encoder_weight_path)

    # If eps_record contains previous eps value then use it, otherwise use default.
    if best_eps:
        eps = best_eps

    eps, outlier_inds, labels = dbscan_clustering(cm_embeddings, eps, min_samples)
    eps_record[encoder_weight_path] = eps

    # Save the eps for next round of the pipeline
    with open(eps_path, 'w') as file:
        json.dump(eps_record, file)

    return outlier_inds, labels


# LOF
def local_outlier_factor(embeddings, n_outliers=500, plot_dir=None, comm=None, **kwargs): 
    # start timer
    t_start = time.time()

    # mpi stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    # LOF
    if comm_rank == 0:
        print("Running LOF")

    # compute LOF
    clf = LocalOutlierFactor(**kwargs)
    # Array with 1 if inlier, -1 if outlier
    embeddings = np.nan_to_num(embeddings, nan=0.)
    clf.fit_predict(embeddings)

    # print the results
    if (plot_dir is not None) and (comm_rank == 0):
        # create directory
        os.makedirs(plot_dir, exist_ok=True)
        # plot
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.hist(clf.negative_outlier_factor_, bins='fd')
        plt.savefig(os.path.join(plot_dir, 'score_distribution.png'))
        plt.close()
    
    # Only sorts 1 element of negative_outlier_factors_, namely the element
    # that is position k in the sorted array. The elements above and below
    # the kth position are partitioned but not sorted. Returns the indices
    # of the elements of left hand side of the parition i.e. the top k.
    outlier_inds = topk(clf.negative_outlier_factor_, k=n_outliers)
    
    outlier_scores = clf.negative_outlier_factor_[outlier_inds]

    # Only sorts an array of size n_outliers
    sort_inds = np.argsort(outlier_scores)
    
    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("LOF Time: {}s".format(t_end - t_start))

    # Returns n_outlier best outliers sorted from best to worst
    return outlier_inds[sort_inds], outlier_scores[sort_inds]

def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict): 
        if local_frame - traj_dict[key] < 0:            
            return local_frame, key
        else: 
            local_frame -= traj_dict[key]
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))


def find_values(rewards_df, fname, xmin, xmax):
    selection = rewards_df["rewarded_inds"].between(xmin, xmax)
    #sel = rewarded_inds_ts[rewarded_inds_ts.between(xmin, xmax)].dropna().values
    selection = rewards_df[selection].dropna()
    selection["rewarded_inds"] -= xmin
    selection["files"] = fname
    return selection


def write_rewarded_pdbs(rewarded_inds, scores, pdb_out_path, data_path, max_retry_count = 5, comm = None):
    # start timer
    t_start = time.time()

    # mpi stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    # Get list of simulation trajectory files (Assume all are equal length (ns))
    with open_h5(data_path) as h5_file:
        traj_files = np.array(h5_file['traj_file'])
        sim_lens = np.array(h5_file['sim_len'])

    # store file and indices in dataframe
    trajdf = pd.DataFrame({"files": traj_files, "lengths": sim_lens})

    # get a global start index
    trajdf["start_index"] = trajdf.shift(1).fillna(0)["lengths"].cumsum(axis=0).astype(int)
    rewards_df = pd.DataFrame({"rewarded_inds": rewarded_inds, "scores": scores}).sort_values("rewarded_inds").reset_index(drop=True)
    trajdf["selected"] = trajdf.apply(lambda x: find_values(rewards_df, x["files"], x["start_index"], x["start_index"] + x["lengths"] - 1), axis=1)
    trajdf = trajdf[ trajdf["selected"].apply(lambda x: x.empty == False) ].reset_index()
    trajdf = pd.concat(trajdf["selected"].tolist()).sort_values("scores", ascending=True).reset_index(drop=True)
    trajdf["order"] = trajdf.index
    
    # now, chunk the frame for each node
    if comm_size > 1:
        fullsize = trajdf.shape[0]
        chunksize = fullsize // comm_size
        start = chunksize * comm_rank
        end = start + chunksize
        remainderdf = trajdf.iloc[chunksize * comm_size:, :]
        trajdf = trajdf.iloc[start:end, :]
        for idx, (ind, row) in enumerate(remainderdf.iterrows()):
            if idx == comm_rank:
                trajdf = trajdf.append(row)

    # For documentation on mda.Writer methods see:
    #   https://www.mdanalysis.org/mdanalysis/documentation_pages/coordinates/PDB.html
    #   https://www.mdanalysis.org/mdanalysis/_modules/MDAnalysis/coordinates/PDB.html#PDBWriter._update_frame

    # now group by files to improve IO
    trajdf.sort_values(by=["files", "rewarded_inds"], inplace=True)

    # do the IO
    groups = trajdf.groupby("files")
    outlier_pdbs = []
    orders = []
    for traj_file, item in groups:
        
        # extract sim tag
        sim_id = os.path.splitext(os.path.basename(traj_file))[0]

        # we might need more than one retry  
        retry_count = 0
        while (retry_count < max_retry_count):
            try:
                sim_pdb = os.path.realpath(glob(os.path.join(os.path.dirname(traj_file), '*.pdb'))[0])
                load_trajec = time.time()
                u = mda.Universe(sim_pdb, traj_file)
                load_trajec = time.time() - load_trajec
                #print("Load trajectory time: {}s".format(load_trajec))
                try:
                    ligand_id = os.path.basename(sim_pdb).split('_')[1]
                except IndexError as e:
                    ligand_id = None
                break
            except IOError as e:
                retry_count += 1
                
        if retry_count < max_retry_count:
            save_trajec = time.time()
            orders += list(item["order"])
            for frame in item["rewarded_inds"]:
                if ligand_id:
                    out_pdb = os.path.abspath(join(pdb_out_path, f'{sim_id}_{ligand_id}_{frame:06}.pdb'))
                else:
                    out_pdb = os.path.abspath(join(pdb_out_path, f'{sim_id}_{frame:06}.pdb'))
                with mda.Writer(out_pdb) as writer:
                    # Write a single coordinate set to a PDB file
                    writer._update_frame(u)
                    writer._write_timestep(u.trajectory[frame])
                outlier_pdbs.append(out_pdb)
            save_trajec = time.time() - save_trajec

    if comm_size > 1:
        outlier_pdbs = comm.allgather(outlier_pdbs)
        outlier_pdbs = list(itertools.chain.from_iterable(outlier_pdbs))
        orders = comm.allgather(orders)
        orders = list(itertools.chain.from_iterable(orders))
        
        
    # sort by order
    outlier_pdbs = [x[1] for x in sorted(zip(orders, outlier_pdbs))]

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("Write PDB Time: {}s".format(t_end - t_start))

    return outlier_pdbs

def md_checkpoints(sim_path, pdb_out_path, outlier_pdbs):
    # Restarts from check point
    checkpnt_list = sorted(glob(os.path.join(sim_path, 'omm_runs_*/checkpnt.chk')))
    restart_checkpnts = [] 
    for checkpnt in checkpnt_list: 
        checkpnt_filepath = os.path.abspath(join(pdb_out_path, os.path.basename(os.path.dirname(checkpnt) + '.chk')))
        if not os.path.exists(checkpnt_filepath): 
            shutil.copy2(checkpnt, checkpnt_filepath) 
            # Includes only checkpoint of trajectory that contains an outlier 
            if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outlier_pdbs):  
                restart_checkpnts.append(checkpnt_filepath)

    return restart_checkpnts


@click.command()
@click.option('-s', '--sim_path', required=True,
              type=click.Path(exists=True),
              help='OpenMM simulation path containing *.dcd and *.pdb files')

@click.option('-p', '--pdb_out_path', default=join('.', 'outlier_pdbs'),
              help='Path to folder to write output PDB files to.')

@click.option('-o', '--restart_points_path',
              default=os.path.abspath('./restart_points.json'),
              help='Path to write output json file with PDB file list.')

@click.option('-d', '--data_path', required=True,
              type=click.Path(exists=True),
              help='Preprocessed data h5 file path')

@click.option('-m', '--model_paths', required=True,
              callback=parse_list,
              help='List of trained model paths "path1,path2,...".')

@click.option('-t', '--model_type', default='vae-resnet',
             help='Type of model. vae-resnet or aae.')

@click.option('-M', '--min_samples', default=10, type=int,
              help='Value of min_samples in the OPTICS algorithm')

@click.option('-n', '--n_outliers', default=500, type=int,
              help='Number of outlier PDBs to find and output.')

@click.option('-D', '--device', default='cpu',
              help='PyTorch formatted device str. cpu, cuda, cuda:0, etc')

@click.option('-h', '--dim1', required=True, type=int,
              help='H of (H,W) shaped contact matrix')

@click.option('-w', '--dim2', required=True, type=int,
              help='W of (H,W) shaped contact matrix')

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Format of contact map files. Options ' \
                   '[full, sparse-concat, sparse-rowcol]')

@click.option('-b', '--batch_size', default=256, type=int,
              help='Batch size for forward pass. Limited by available ' \
                   ' memory and size of input example. Does not affect output.' \
                   ' The larger the batch size, the faster the computation.')

@click.option('--distributed', is_flag=True,
              help='Enable distributed training')

def main(sim_path, pdb_out_path, restart_points_path, data_path, model_paths, model_type,
         n_outliers, min_samples, device, dim1, dim2, cm_format, batch_size, distributed):

    # start timer
    t_start = time.time()

    comm_size = 1
    comm_rank = 0
    comm = None
    if distributed:
        # get communicator: duplicate from comm world
        from mpi4py import MPI
        comm = MPI.COMM_WORLD.Dup()
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    if comm_rank == 0:
        print('DEBUG outlier detection')
        print('sim_path: ', sim_path)
        print('pdb_out_path: ', pdb_out_path)
        print('restart_points_path: ', restart_points_path)
        print('data_path: ', data_path)
        print('model_paths: ', model_paths)
        print('model_type: ', model_type)
        print('n_outliers: ', n_outliers)
        print('min_samples: ', min_samples)
        print('device: ', device)
        print('dim1: ', dim1)
        print('dim2: ', dim2)
        print('cm_format: ', cm_format)
        print('batch_size: ', batch_size)

    # Make directory to store output PDB files
    if comm_rank == 0:
        os.makedirs(pdb_out_path, exist_ok=True)

    # do distributed model selection
    best_hparams, best_checkpoint = model_selection_embedding_loss_based(model_paths, top_variance_threshold=0.1, 
                                                                         num_select=1, comm=comm)
    if comm_rank == 0:
        print('best hparams: ', best_hparams)
        print('best_checkpoint: ', best_checkpoint)

    # Generate embeddings for all contact matrices produced during MD stage
    embeddings, indices = generate_embeddings(model_type, best_hparams, best_checkpoint, dim1, dim2,
                                              device, data_path, cm_format, batch_size, comm=comm)

    if comm_rank == 0:
        print('embeddings shape: ', embeddings.shape)
        print('indices shape: ', indices.shape)

        print('embeddings vals: ', embeddings[0])
        print('indices vals: ', indices[0])
    
    # Perform DBSCAN clustering on embeddings
    #outlier_inds, labels = perform_clustering(eps_path, best_checkpoint,
    #                                          embeddings, min_samples, eps)

    # Perform OPTICS clustering on embeddings
    #outlier_inds, labels = optics_clustering(embeddings, min_samples=min_samples)

    # Perform LocalOutlierFactor outlier detection on embeddings
    if comm_rank == 0:
        outlier_inds, scores = local_outlier_factor(embeddings,
                                                    n_outliers=n_outliers,
                                                    plot_dir=os.path.join(pdb_out_path, 'figures'),
                                                    n_jobs=-1, comm=comm)
    else:
        outlier_inds = None
        scores = None

    if comm_size > 1:
        outlier_inds = comm.bcast(outlier_inds, 0)
        scores = comm.bcast(scores, 0)

    if comm_rank == 0:
        print('outlier_inds shape: ', outlier_inds.shape)
        for ind, score in zip(outlier_inds, scores):
            print(f'ind, score: {ind}, {score}')

    # Map shuffled indices back to in-order MD frames
    simulation_inds = indices[outlier_inds]

    if comm_rank == 0:
        print('simulation_inds shape', simulation_inds.shape)

    # Write rewarded PDB files to shared path
    outlier_pdbs = write_rewarded_pdbs(simulation_inds, scores, pdb_out_path, data_path, comm = comm)
    
    if comm_rank == 0:
        restart_checkpnts = md_checkpoints(sim_path, pdb_out_path, outlier_pdbs)

    if comm_rank == 0:
        print('outlier_pdbs len: ', len(outlier_pdbs))
        print('restart_checkpnts len: ', len(restart_checkpnts))

    if comm_rank == 0:
        restart_points = restart_checkpnts + outlier_pdbs

    if comm_rank == 0:
        print('restart_points len: ', len(restart_points))
        print('restart_points: ', restart_points)
        
    if comm_rank == 0:
        with open(restart_points_path, 'w') as restart_file:
            json.dump(restart_points, restart_file)
    
    # final barrier
    if comm:
        comm.barrier()

    # end
    t_end = time.time()
    if comm_rank == 0:
        print("Outlier Detection Time: {}s".format(t_end - t_start))

if __name__ == '__main__':
    # set forkserver
    torch.multiprocessing.set_start_method('forkserver', force = True)

    main()
