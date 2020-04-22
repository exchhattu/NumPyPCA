"""
Wed 22 Apr 2020 02:29:27 PM EDT
By: Rojan Shrestha PhD
"""
from __future__ import print_function
import argparse
import sys

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

    u_ref_CA  = u_ref.select_atoms("name CA")
    # u_mob_CAs = u_mob.select_atoms("resid 299-400 and name CA")
    u_mob_CAs = u_mob.select_atoms("name CA")

    # X,Y,Z will be
    results = {}
    ref_CA_trans = u_ref_CA.positions - u_ref_CA.center_of_mass()

    u_mob.trajectory[0]   # rewind trajectory
    for ts in u_mob.trajectory:
        mob_CA_trans = u_mob_CAs.positions - u_mob_CAs.center_of_mass()
        R, rmsd = align.rotation_matrix(mob_CA_trans, ref_CA_trans)
	u_mob_CAs.atoms.translate(-u_mob_CAs.select_atoms('name CA').center_of_mass())
        u_mob_CAs.atoms.rotate(R)
        u_mob_CAs.atoms.translate(u_ref_CA.center_of_mass())
	print("rmsd", rmsd)

        # n*3 matrix where n is # of atoms
        positions = u_mob_CAs.positions
	print(positions.shape, ref_CA_trans.shape)

        # mobile.atoms.write("mobile_on_ref.pdb")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='./prepare_timeseries_data.py', description='time series data')

  parser.add_argument('-r','--ref', required=True, help='path to reference file')
  parser.add_argument('-m','--mob', required=True, help='path to mobile file')
  parser.add_argument('-s','--sfile', required=True, help='path to setup file')
  parser.add_argument('-o','--outfile', required=False, default="data.xyz", help='XYZ coordinates')
  
  args = parser.parse_args(sys.argv[1:])
  parse_trajectories(args.ref, args.mob, args.sfile, output_file=args.outfile)
