#!/usr/bin/env python
import sys
import h5py
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Not enough arguments")
        return

    # Note to self, currently using this h5:
    # /home/gstavrinos/libs/python/python2.7/pointnet2/data/modelnet40_ply_hdf5_2048/ply_data_test0.h5
    filename = sys.argv[1]

    with h5py.File(filename, 'r') as f:
        print("Keys: %s" % f.keys())
        for key in f.keys():
            print(type(f[key][-1]))
            print(np.shape(f[key]))
            print(f)

main()