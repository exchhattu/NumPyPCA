"""
Wed 22 Apr 2020 02:29:27 PM EDT
By: Rojan Shrestha PhD
"""
from __future__ import print_function
import argparse
import sys
import csv

from collections import OrderedDict 

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd


def parse_trajectories(path_to_ref, path_to_mobile, path_to_setup):
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

        # n*3 matrix where n is # of atoms and each atom has X, Y, and Z
        # However, coordinate is appended from the snapshot captured at each time interval
        positions = u_mob_CAs.positions
	for i in range(positions.shape[0]):
           results[i] = results.get(i, [])
           results[i].append(positions[i,:].tolist())
    return results

def write_file(results, exp_type="W", out_file="result.csv"):
    with open(out_file, 'w') as f:
       for key, value in results.items():
	   merge_value = ",".join([str(inner) for outer in value for inner in outer])
	   # columns - 
	   # 1. type of experiment (M/W) 
	   # 2. sn 
           # 3-5. X,Y,Z and this is repeated for t times 
           f.write("%s,%s,%s\n" %(exp_type, key,merge_value))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='./prepare_timeseries_data.py', description='time series data')

  parser.add_argument('-r','--ref', required=True, help='path to reference file')
  parser.add_argument('-m','--mob', required=True, help='path to mobile file')
  parser.add_argument('-s','--sfile', required=True, help='path to setup file')
  parser.add_argument('-o','--outfile', required=False, default="data.xyz", help='XYZ coordinates')
  # two types of symbol would be used W and M. They represent wild type and mutant
  parser.add_argument('-t','--exp_type', required=False, default="W", help='Type of biological experiment')
  
  args = parser.parse_args(sys.argv[1:])
  result = parse_trajectories(args.ref, args.mob, args.sfile)
  write_file(result, exp_type=args.exp_type, out_file=args.outfile)
