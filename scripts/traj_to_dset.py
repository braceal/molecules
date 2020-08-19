import os
import glob
import click
from molecules.sim.contact_maps import traj_to_dset

@click.command()

@click.option('-p', 'pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing PDB file')

@click.option('-r', 'ref_pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing reference state PDB file')

@click.option('-t', 'traj_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing MD trajectory ' \
                   'OR directory containing many traj files. If ' \
                   'directory, files are sorted lexographically ' \
                   'by their names and then concatenated.')

@click.option('-e', '--ext', default='dcd',
              help='Trajectory file extension.')

@click.option('-o', '--out_path', required=True,
              help='Path to file to write sparse contact matrices to.')

@click.option('-w', '--num_workers', default=None, type=int,
              help='Number of parallel workers for processing multiple ' \
                   'traj files in parallel. Defaults to number of cpus ' \
                   'on machine.')

@click.option('-s', '--selection', default='protein and name CA',
              help='Atom selection for creating contact maps, ' \
                    'point clouds and computing fnc.')

@click.option('-c', '--cutoff', default=8., type=float,
              help='Distanct cutoff measured in angstroms to ' \
                   'compute contact maps and fnc')

@click.option('--rmsd', is_flag=True,
              help='Computes and saves RMSD.')

@click.option('--fnc', is_flag=True,
              help='Computes and saves fraction of reference contacts.')

@click.option('--contact_map', is_flag=True,
              help='Computes and saves contact maps.')

@click.option('--point_cloud', is_flag=True,
              help='Computes and saves point cloud.')

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Options: [sparse-concat, sparse-rowcol]. Refers to format the ' \
                   'the contact maps are stored as. latest gives single ' \
                   'dset format while oldest gives group with row,col dset.')

@click.option('-v', '--verbose', is_flag=True)

def main(pdb_path, ref_pdb_path, traj_path, ext, out_path, num_workers,
         selection, cutoff, rmsd, fnc, contact_map, point_cloud, cm_format, verbose):

    if os.path.isdir(traj_path):
        traj_path = sorted(glob.glob(os.path.join(traj_path, f'*.{ext}')))

        if verbose:
            print(f'Collected {len(traj_path)} {ext} files')
    else:
        # Can't parallelize over single file
        num_workers = 1

    if verbose:
        num_traj_files = len(traj_path) if isinstance(traj_path, list) else 1
        print(f'Using {num_workers} workers to process {num_traj_files} traj files')

    traj_to_dset(pdb_path, ref_pdb_path, out_path, traj_path,
                 rmsd=rmsd, fnc=fnc, point_cloud=point_cloud,
                 contact_map=contact_map, sel=selection,
                 cutoff=cutoff, cm_format=cm_format, verbose=verbose,
                 num_workers=num_workers) 
    
if __name__ == '__main__':
    main()
