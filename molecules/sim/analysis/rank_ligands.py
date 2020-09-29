import MDAnalysis as mda

def prot_lig_contact(pdb_file, lig_name='UNK', cutoff=4.):
    """
    Calculate the heavy atom contact between protein
    and ligand, assuming ligand name is UNK.
    """
    mda_traj = mda.Universe(pdb_file)
    protein_noH = mda_traj.select_atoms('protein and not name H*')
    ligand = mda_traj.select_atoms(f'resname {lig_name}')
    distance_pro_lig = distances.distance_array(
                            protein_noH.positions, 
                            ligand.positions, 
                            box=protein_noH.dimensions)

    return distance_pro_lig < cutoff
