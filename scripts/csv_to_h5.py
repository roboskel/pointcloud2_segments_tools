#!/usr/bin/env python
import os
import re
import sys
import h5py

def main():
    if len(sys.argv) < 2:
        print("Not enough arguments")
        return

    # Note to self, currently using this csv:
    # /home/gstavrinos/catkin_ws/src/new_hpr/pointcloud2_segments_tools/dataset0.csv
    input_file = sys.argv[1]
    path, f = sys.argv[1].rsplit(os.sep, 1)
    output_file = path + os.sep + f.rsplit(".", 1)[0] + ".h5"
    print("---")
    print("Your h5 will be written in:")
    print(output_file)
    print("---")

    data = []
    labels = []

    print("Processing csv file...")
    with open(input_file, "rb") as if_:
        # TODO:
        # hdf5 requires matrices with the same number of elements for all of its rows
        # this basically means that I need to create a simple sampling procedure for each pointcloud
        # in order to for all of them to have a target number of points
        # Note: The csv tha I am currently using has the following stats:
        # PCs with less than 1024 points:                       20748
        # PCs with less than 2048 and more than 1024 points:    24805
        # PCs with more than 2048 points:                       8366
        for line in if_.readlines():
            dt, lbl = line.rsplit(",", 1)
            i = 0
            tmp = []
            for d in re.split(r"\(([^)]+)\)", dt):
                # Skip every other element, because we get "," characters between each tuple
                if i % 2 != 0:
                    d = d.split(",")
                    tmp.append([float(d[0]), float(d[1]), float(d[2])])
                i += 1
            data.append(tmp)
            labels.append(int(lbl))

    print("---")
    print("Generating h5 file...")
    h5out = h5py.File(output_file, "w")
    h5out.create_dataset("data", data=data, compression="gzip", compression_opts=4, dtype="float32")
    h5out.create_dataset("label", data=labels, compression="gzip", compression_opts=4, dtype="int")
    h5out.close()
    print("All done! bb")


main()