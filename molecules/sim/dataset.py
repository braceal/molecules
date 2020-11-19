import os
import time
import h5py
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms, align
from molecules.utils import open_h5
import shutil
import uuid



def _save_sparse_contact_maps(h5_file, contact_maps, 
                              cm_format='sparse-concat', **kwargs):
    """
    Saves contact maps in sparse format. Only stores row,col
    indices of the 1s in the contact matrix. Values must be
    recomputed on the fly as an array of ones equal to the
    length of the row,col vectors. Note, row and col vectors
    have the same length.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file to write contact maps to

    contact_maps : tuple
        (row, col, val) where row,col are lists of np.ndarrays.
        row[i], col[i] represents the indices of a contact
        map where val[i] is not zero. If val is empty, quantized
        matrix is expected.

    cm_format : str
        Format to save contact maps with.
        sparse-concat: new format with single dataset
        sparse-rowcol: old format with group containing row,col datasets

    kwargs : dict
        Optional h5py parameters to be used in dataset creation.

   Note: The sparse-concat format has more efficient chunking than
         the sparse-rowcol format.
   """

    rows, cols, vals = contact_maps

     # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype('int16'))
    dtv = h5py.vlen_dtype(np.dtype('float32'))

    # Helper function to create ragged array
    ragged = lambda data: np.array(data, dtype=object)

    if cm_format == 'sparse-concat':
        # list of np arrays of shape (2 * X) where X varies
        data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
        h5_file.create_dataset('contact_map', data=data, chunks=(1,) + data.shape[1:], dtype=dt, **kwargs)
        if vals:
            datav = ragged(vals)
            h5_file.create_dataset('contact_map_values', data=datav, chunks=(1,) + datav.shape[1:], dtype=dtv, **kwargs)

    elif cm_format == 'sparse-rowcol':
        group = h5_file.create_group('contact_map')
        # The i'th element of both row,col dset will be arrays of the
        # same length. However, neighboring arrays may be any variable length.
        group.create_dataset('row', dtype=dt, data=ragged(rows), chunks=(1,), **kwargs)
        group.create_dataset('col', dtype=dt, data=ragged(cols), chunks=(1,), **kwargs)
        if vals:
            group.create_dataset('val', dtype=dtv, data=ragged(vals), chunks=(1,), **kwargs)

def _save(save_file, rmsd=None, fnc=None, point_cloud=None,
          contact_maps=None, cm_format='sparse-concat',
          sim_lens=None, traj_files=None):
    """
    Saves data to h5 file. All data is optional, can save any
    combination of data input arrays.

    Parameters
    ----------
    save_file : str
        Path of output h5 file used to save datasets.

    rmsd : np.ndarray
        Stores floating point rmsd data.

    fnc : np.ndarray
        Stores floating point fraction of native contact data.

    point_cloud : np.ndarray
        Stores matrix of point cloud data. (N, K, 3) where K
        is the number of selected atoms.

    contact_maps : tuple
        (row, col) where row,col are lists of np.arrays.
        row[i], col[i] represents the indices of a contact
        map where a 1 exists.

    cm_format : str
        Format to save contact maps with.
        sparse-concat: new format with single dataset
        sparse-rowcol: old format with group containing row,col datasets
    """
    kwargs = {'fletcher32': True}
    int_kwargs = {'fletcher32': True, 'dtype': 'int32'}
    string_kwargs = {'fletcher32': True, 'dtype': h5py.string_dtype('utf-8')}
    scalar_kwargs = {'fletcher32': True, 'dtype': 'float16', 'chunks':(1,)}

    with open_h5(save_file, 'w', swmr=False) as h5_file:

        # Save simulation length
        if sim_lens is not None:
            h5_file.create_dataset('sim_len', data=sim_lens, **int_kwargs)
        # Save simulation traj file names
        if traj_files is not None:
            utf8_type = h5py.string_dtype('utf-8')
            traj_files_data = np.array([np.array(t, dtype=utf8_type) for t in traj_files])
            h5_file.create_dataset('traj_file', data=traj_files_data, **string_kwargs)
        # Save rmsd
        if rmsd is not None:
            h5_file.create_dataset('rmsd', data=rmsd, **scalar_kwargs)
        # Save fraction of native contacts
        if fnc is not None:
            h5_file.create_dataset('fnc', data=fnc, **scalar_kwargs)
        # Save point cloud
        if point_cloud is not None:
            h5_file.create_dataset('point_cloud', data=point_cloud,
                                   chunks=(1,) + point_cloud.shape[1:],
                                   dtype='float32', **kwargs)
        # Save contact maps
        if contact_maps is not None:
            _save_sparse_contact_maps(h5_file, contact_maps, 
                                      cm_format=cm_format, **kwargs)

def fraction_of_contacts(cm, ref_cm):
    """
    Given two contact matices of equal dimensions, computes
    the fraction of entries which are equal. This is
    comonoly refered to as the fraction of contacts and
    in the case where ref_cm represents the native state
    this is the fraction of native contacts.

    Parameters
    ----------
    cm : np.ndarray
        A contact matrix.

    ref_cm : np.ndarray
        The reference contact matrix for comparison.
    """
    return 1 - (cm != ref_cm).mean()

def wrap_traj(pdb, dcd, output_dcd)
    u = mda.Universe(pdb, dcd)
    system = u.select_atoms('protein and resname UNK')
    with mda.Writer(output_dcd, u.atoms.n_atoms) as W: 
        for _ in u.trajectory: 
            box_size = system.dimensions[:3]
            box_center = box_size / 2 
            trans_vec = box_center - (system.center_of_mass())
            system.translate(trans_vec).wrap() 
            W.write(system)

def _traj_to_dset(topology, ref_topology, traj_file,
                  save_file=None,
                  rmsd=True, fnc=True,
                  point_cloud=True,
                  contact_map=True,
                  distance_kernel_params = {"kernel_type": "threshold", "threshold": 8.},
                  sel='protein and name CA',
                  cm_format='sparse-concat',
                  verbose=False):
    """
    Implementation for generating machine learning datasets
    from raw molecular dynamics trajectory data. This function
    uses MDAnalysis to load the trajectory file and given a
    custom atom selection computes contact matrices, RMSD to
    reference state, fraction of reference contacts and the
    point cloud (xyz coordinates) of each frame in the trajectory.

    Parameters
    ----------
    topology : str
        Path to topology file: CHARMM/XPLOR PSF topology file,
        PDB file or Gromacs GRO file.

    ref_topology : str
        Path to reference topology file for aligning trajectory:
        CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file.

    traj_file : str
        Trajectory file (in CHARMM/NAMD/LAMMPS DCD, Gromacs XTC/TRR,
        or generic. Stores coordinate information for the trajectory.

    save_file : str
        Path to output h5 dataset file name.

    rmsd, fnc, point_cloud, contact_map : bool
        Those flags that are marked true are computed and saved
        into h5 dataset files. Any combination may be selected.
                  
    distance_kernel_params : dict
        dictionary should contain 
            `kernel_type`: allowed values are `threshold` and `laplace`.
            `threshold`: threshold for hard cutoff
            `lambda`: exponential decay constant for largange kernel
                      (exp(-lambda * dist)).
        Only relevant if contact_map = True.
    
    sel : str
        Selection set of atoms in the protein.

    cm_format : str
        Format to save contact maps with.
        sparse-concat: new format with single dataset
        sparse-rowcol: old format with group containing row,col datasets

    verbose: bool
        If true, prints verbose output.

    Returns
    -------
    tuple : rmsd_data, fnc_data, pc_data, contact_map_data
        If not None then they will contain the raw data for each type
        of computation. See _save function parameters for the data format.
    """

    # start timer
    start_time = time.time() 
    
    # Copy traj_file to node local storage and wrap trajectory
    local_traj_file = shutil.copy(traj_file, f'$BBPATH/{uuid.uuid4()}.dcd')
    wrapped_traj_file =  f'$BBPATH/{uuid.uuid4()}.dcd' 
    wrap_traj(topology, local_traj_file, wrapped_traj_file)
 
    # Load simulation and reference structures
    sim = mda.Universe(topology, wrapped_traj_file)
    ref = mda.Universe(ref_topology)

    if verbose:
        print('Traj length: ', len(sim.trajectory))

    # Atom selection for reference
    atoms = sim.select_atoms(sel)
    # Get atomic coordinates of reference atoms
    ref_positions = ref.select_atoms(sel).positions.copy()
    # Get contact map of reference atoms
    ref_cm = distances.contact_matrix(ref_positions, 
                                      float(distance_kernel_params["threshold"]),
                                      returntype='sparse')
    if rmsd or point_cloud:
        # Align trajectory to compute accurate RMSD or point cloud
        align.AlignTraj(sim, ref, select=sel, in_memory=True).run()
        
    # Initialize buffers. Only turn into containers if user specifies to.
    rmsd_data, fnc_data, pc_data, contact_map_data = None, None, None, None

    # Buffers for storing RMSD and fraction of native contacts
    params = {'shape': len(sim.trajectory), 'dtype': np.float16}
    if rmsd:
        rmsd_data = np.empty(**params)
    if fnc:
        fnc_data = np.empty(**params)

    # Buffers for sparse contact row/col data
    if contact_map:
        row, col, val = [], [], []
        contact_map_data = (row, col, val)

    # Buffer for point cloud data
    if point_cloud:
        pc_data = np.empty(shape=(len(sim.trajectory), *ref_positions.shape),
                           dtype=np.float32)

    print_freq = 10
    for i, frame in enumerate(sim.trajectory):

        # Point cloud positions of selected atoms in frame i
        positions = atoms.positions

        if fnc:
            threshold =  float(distance_kernel_params["threshold"])
            # Compute contact map of current frame (scipy lil_matrix form)
            cm = distances.contact_matrix(positions, threshold, returntype='sparse')
            # Compute fraction of contacts to reference state
            fnc_data[i] = fraction_of_contacts(cm, ref_cm)

        if contact_map:
            if (distance_kernel_params["kernel_type"] == "threshold"):
                # Compute contact map of current frame (scipy lil_matrix form)
                if not fnc:
                    threshold =  float(distance_kernel_params["threshold"])
                    cm = distances.contact_matrix(positions, threshold, returntype='sparse')
                # Represent contact map in COO sparse format
                coo = cm.tocoo()
                row.append(coo.row.astype('int16'))
                col.append(coo.col.astype('int16'))
            else:
                dist = distances.self_distance_array(positions.copy(), box=None, backend='serial')
                row_tmp = []
                col_tmp = []
                val_tmp = []
                k = 0
                for ii in range(ref_positions.shape[0]):
                    row_tmp.append(ii)
                    col_tmp.append(ii)
                    val_tmp.append(1.)
                    for jj in range(ii + 1, ref_positions.shape[0]):
                        # check if we care
                        if dist[k] <= threshold:
                            row_tmp.append(ii)
                            col_tmp.append(jj)
                            # compute metric
                            if (distance_kernel_params["kernel_type"] == "laplace"):
                                dval = np.exp(-float(distance_kernel_params["lambda"]) * dist[k])
                                val_tmp.append(dval)
                        # increment counter
                        k += 1
                
                # append globally
                row.append(np.array(row_tmp, dtype=np.int16))
                col.append(np.array(col_tmp, dtype=np.int16))
                val.append(np.array(val_tmp, dtype=np.float32))

        if rmsd:
            # Compute and store RMSD to reference state
            rmsd_data[i] = rms.rmsd(positions, ref_positions, center=True,
                                    superposition=True)

        if point_cloud:
            # Store reference atoms point cloud of current frame
            pc_data[i] = positions.copy()

        if verbose:
            if i % print_freq == 0:
                str_ = f'Frame {i}/{len(sim.trajectory)}'
                if rmsd:
                    str_ += f'\trmsd: {rmsd_data[i]}'
                if fnc:
                    str_ += f'\tfnc: {fnc_data[i]}'
                str_ += f'\tshape: {positions.shape}'

                print(str_)

    if point_cloud:
        pc_data = np.transpose(pc_data, [0,2,1])
    if save_file:
        # Write data to HDF5 file
        _save(save_file, rmsd=rmsd_data, fnc=fnc_data,
              point_cloud=pc_data, contact_maps=contact_map_data,
              cm_format=cm_format)

    sim_len = len(sim.trajectory)

    # end time
    end_time = time.time()
    duration = end_time - start_time

    #move wrapped_traj back to parallel file system
    shutil.move(wrapped_traj_file, traj_file)
    # Any of these could be None based on the user input
    return rmsd_data, fnc_data, pc_data, contact_map_data, sim_len, duration

def _worker(kwargs):
    """Helper function for parallel data collection."""
    id_ = kwargs.pop('id')
    return _traj_to_dset(**kwargs), id_

def traj_to_dset(topology, ref_topology, traj_files, save_file,
                 rmsd=True, fnc=True, point_cloud=True, contact_map=True,
                 distance_kernel_params = {"kernel_type": "threshold", "threshold": 8.}, 
                 sel='protein and name CA', cm_format='sparse-concat',
                 num_workers=None, comm=None, verbose=False):
    """
    User-level function for generating machine learning datasets
    from raw molecular dynamics trajectory data. This function
    uses MDAnalysis to load the trajectory file and given a
    custom atom selection computes contact matrices, RMSD to
    reference state, fraction of reference contacts and the
    point cloud (xyz coordinates) of each frame in the trajectory.
    If multiple traj_files are passed, then they can be processed
    in parallel with a specified number of worker threads.

    Parameters
    ----------
    topology : str
        Path to topology file: CHARMM/XPLOR PSF topology file,
        PDB file or Gromacs GRO file.

    ref_topology : str
        Path to reference topology file for aligning trajectory:
        CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file.

    traj_file : str
        Trajectory file (in CHARMM/NAMD/LAMMPS DCD, Gromacs XTC/TRR,
        or generic. Stores coordinate information for the trajectory.

    save_file : str
        Path to output h5 dataset file name.

    rmsd, fnc, point_cloud, contact_map : bool
        Those flags that are marked true are computed and saved
        into h5 dataset files. Any combination may be selected.

    distance_kernel_params : dict
        dictionary should contain 
            `kernel_type`: allowed values are `threshold` and `lagrange`.
            `threshold`: threshold for hard cutoff
            `lambda`: exponential decay constant for largange kernel
                      (exp(-lambda * dist)).
        Only relevant if contact_map = True.

    sel : str
        Selection set of atoms in the protein.

    cm_format : str
        Format to save contact maps with.
        sparse-concat: new format with single dataset
        sparse-rowcol: old format with group containing row,col datasets

    num_workers : int, None
        Number of worker threads. Parallelism is applied over the number
        of input trajectory files. If left as None, will default to the
        the smaller of the number of input trajectory files and the
        number of available cpus.

    comm: MPI communicator object, None

    verbose: bool
        If true, prints verbose output.

    Returns
    -------
    tuple : rmsd_data, fnc_data, pc_data, contact_map_data
        If not None then they will contain the raw data for each type
        of computation. See _save function parameters for the data format.
    """

    comm_rank = 0
    comm_size = 1
    if comm is not None:
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

    if (num_workers == 1 and comm_size == 1) or isinstance(traj_files, str):

        if verbose:
           num_traj_files = len(traj_files) if isinstance(traj_files, list) else 1
           print(f'Using 1 worker to process {num_traj_files} traj file')

        return _traj_to_dset(topology, ref_topology, traj_files,
                             sel=sel, save_file=save_file,
                             rmsd=rmsd, fnc=fnc,
                             point_cloud=point_cloud, contact_map=contact_map,
                             distance_kernel_params = distance_kernel_params,
                             cm_format=cm_format, verbose=verbose)

    import os
    import itertools
    from concurrent.futures import ProcessPoolExecutor
    
    # sort the list of trajectory files
    traj_files = sorted(traj_files)

    # do round robin assignments because of load balancing
    traj_files_local = [f for idx,f in enumerate(traj_files) if idx%comm_size == comm_rank]

    # Set num_workers to max necessary/possible unless specified by user
    if num_workers is None:
        num_workers = min(os.cpu_count(), len(traj_files_local))

    if verbose:
        print(f'Using {num_workers} workers to process {len(traj_files_local)} traj files')

    # Arguments for workers
    kwargs = [{'topology': topology,
               'ref_topology': ref_topology,
               'traj_file': traj_file,
               'rmsd': rmsd,
               'fnc': fnc,
               'point_cloud': point_cloud,
               'contact_map': contact_map,
               'distance_kernel_params': distance_kernel_params,
               'sel': sel,
               'verbose': verbose and (i % num_workers == 0),
               'id': i}
              for i, traj_file in enumerate(traj_files_local)]

    # initialize buffers
    ids, sim_lens = [], []
    if rmsd:
        rmsds = []
    if fnc:
        fncs = []
    if point_cloud:
        point_clouds = []
    if contact_map:
        rows, cols, vals = [], [], []

    if verbose:
        print("Starting evaluation")

    finished_count = 0
    #with ThreadPoolExecutor(max_workers=num_workers) as executor:
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for (rmsd_data, fnc_data, pc_data, contact_map_data, sim_len, duration), id_ in executor.map(_worker, kwargs):
            if rmsd:
                rmsds.append(rmsd_data)
            if fnc:
                fncs.append(fnc_data)
            if point_cloud:
                point_clouds.append(pc_data)
            if contact_map:
                row, col, val = contact_map_data
                rows.append(row)
                cols.append(col)
                vals.append(val)

            ids.append(id_)
            sim_lens.append(sim_len)

            finished_count += 1

            if verbose:
                print('finished id: ', id_)
                print('duration: ', duration)

    # Initialize buffers. Only turn into containers if user specifies to.
    rmsds_, fncs_, point_clouds_, contact_maps_ = None, None, None, None
    if rmsd:
        rmsds_ = []
    if fnc:
        fncs_ = []
    if point_cloud:
        point_clouds_ = []
    if contact_map:
        rows_, cols_, vals_ = [], [], []

    # Negligible runtime (1e-05 seconds)
    for _, rmsd_data, fnc_data, pc_data, row, col, val  in sorted(zip(ids, rmsds, fncs, point_clouds, rows, cols, vals)):
        if rmsd:
            rmsds_.append(rmsd_data)
        if fnc:
            fncs_.append(fnc_data)
        if point_cloud:
            point_clouds_.append(pc_data)
        if contact_map:
            rows_.append(row)
            cols_.append(col)
            vals_.append(val)

    # we need to gather results here 
    if comm_size > 1:
        traj_files_ = comm.gather(traj_files_local, 0)
        sim_lens = comm.gather(sim_lens, 0)
        if rmsd:
            rmsds_ = comm.gather(rmsds_, 0)
        if fnc:
            fncs_ = comm.gather(fncs_, 0)
        if point_cloud:
            point_clouds_ = comm.gather(point_clouds_, 0)
        if contact_map:
            rows_ = comm.gather(rows_, 0)
            cols_ = comm.gather(cols_, 0)
            if vals_:
                vals_ = comm.gather(vals_, 0)
            else:
                vals_list = []

        if comm_rank == 0:
            traj_files_ = list(itertools.chain.from_iterable(traj_files_))
            sim_lens = list(itertools.chain.from_iterable(sim_lens))
            sim_lens = [x[1] for x in sorted(zip(traj_files_, sim_lens))]
            if rmsd:
                rmsds_ = list(itertools.chain.from_iterable(rmsds_))
                rmsds_ = [x[1] for x in sorted(zip(traj_files_, rmsds_))]

            if fnc:
                fncs_ = list(itertools.chain.from_iterable(fncs_))
                fncs_ = [x[1] for x in sorted(zip(traj_files_, fncs_))]

            if point_cloud:
                point_clouds_ = list(itertools.chain.from_iterable(point_clouds_))
                point_clouds_ = [x[1] for x in sorted(zip(traj_files_, point_clouds_))]

            if contact_map:
                rows_ =list(itertools.chain.from_iterable(rows_))
                rows_ = [x[1] for x in sorted(zip(traj_files_, rows_))]
                
                cols_ =list(itertools.chain.from_iterable(cols_))
                cols_ = [x[1] for x in sorted(zip(traj_files_, cols_))]
                
                if vals_:
                    vals_ =list(itertools.chain.from_iterable(vals_))
                    vals_ = [x[1] for x in sorted(zip(traj_files_, vals_))]

            # sort the filenames as well
            traj_files_ = sorted(traj_files_)
    else:
        traj_files_ = traj_files_local

    # Negligible runtime (1e-2 seconds)
    if comm_rank == 0:
        if rmsd:
            rmsds_ = np.concatenate(rmsds_)
        if fnc:
            fncs_ = np.concatenate(fncs_)
        if point_cloud:
            point_clouds_ = np.concatenate(point_clouds_)
        if contact_map:
            rows_ = list(itertools.chain.from_iterable(rows_))
            cols_ = list(itertools.chain.from_iterable(cols_))
            vals_ = list(itertools.chain.from_iterable(vals_))
            contact_maps_ = (rows_, cols_, vals_)

        # store results
        _traj_files = [os.path.abspath(x) for x in traj_files_]

        _save(save_file, rmsd=rmsds_, fnc=fncs_, point_cloud=point_clouds_,
              contact_maps=contact_maps_, cm_format=cm_format, sim_lens=sim_lens,
              traj_files=_traj_files)

    return rmsds_, fncs_, point_clouds_, contact_maps_

def sparse_contact_maps_from_matrices(contact_maps, rmsd=None, fnc=None,
                                      point_cloud=None, save_file=None,
                                      cm_format='sparse-concat'):
    """
    User-level function. Convert normal full contact matrices to
    a sparse format. Can also be used to save rmsd, fnc, point_cloud
    after converting contact map format (for convenience).

    Parameters
    ----------
    contact_maps : np.ndarray
        Full contact matrices of either dimension (N, W, H, 1)
        or (N, W, H).

    rmsd, fnc, point_cloud, contact_map : bool
        Those flags that are marked true are computed and saved
        into h5 dataset files. Any combination may be selected.

    save_file : str
        Path to output h5 dataset file name.

    cm_format : str
        Format to save contact maps with.
        sparse-concat: new format with single dataset
        sparse-rowcol: old format with group containing row,col datasets
    """
    from scipy.sparse import coo_matrix

    row, col = [], []
    for cm in map(lambda cm: cm.squeeze(), contact_maps):
        # Represent contact matrix in COO sparse format
        coo = coo_matrix(cm, shape=cm.shape)
        row.append(coo.row.astype('int16'))
        col.append(coo.col.astype('int16'))

    if save_file:
        _save(save_file, rmsd=rmsd, fnc=fnc,
              point_cloud=point_clouds,
              contact_maps=(row, col),
              cm_format=cm_format)

    return row, col

