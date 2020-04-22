"""
Wed 22 Apr 2020 02:29:27 PM EDT
By: Rojan Shrestha PhD
"""

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd


def parse_trajectories(path_to_ref, path_to_mobile, path_to_setup, output_file=""):
    """[summary]

    Args:
        path_to_ref ([type]): [description]
        path_to_mobile ([type]): [description]
        path_to_setup ([type]): [description]
        output_file (str, optional): [description]. Defaults to "".
    """
    # reference structure for superpositioning
    u_ref = mda.Universe(path_to_ref)
    u_mob = mda.Universe(path_to_setup, path_to_mobile)

    u_ref_CA = u_ref.select_atoms("name CA")

    u_mob_CAs = u_mob.select_atoms("resid 299-400 and name CA")

    # X,Y,Z will be
    results = {}
    ref_CA_trans = u_ref_CA - u_ref_CA.center_of_mass()
    for iframe, ts in enumerate(u_mob_CAs.trajectory):

        mod_CA_trans = u_mob_CAs - u_mob_CAs.center_of_mass()
        R, rmsd = align.rotation_matrix(mob_CA_trans, ref_CA_trans)
        mod_CA_trans.atoms.rotate(R)
        mod_CA_trans.atoms.translate(ref_CA.center_of_mass())

        # n*3 matrix where n is # of atoms
        positions = mod_CA_trans

        mobile.atoms.write("mobile_on_ref.pdb")
