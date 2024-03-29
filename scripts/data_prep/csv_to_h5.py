#!/usr/bin/env python
import os
import re
import sys
import h5py
import random
import numpy as np
from tqdm import tqdm

poi = np.array([])

def dist(point1):
    global poi
    return np.linalg.norm(point1-poi)

# One way to reach a specified number of points, is to remove points from "bigger" pointclouds
# The idea is to remove the points that are farthest away from the center of the pointcloud
# The problem here:
# Is the shape of the pointcloud maintained, or we mutated it?
# Possible solution:
# Apply a light shape-maintaing sampling method and then remove the farthest points (but is this easy?)
def clearPointcloudOuterPoints(pointcloud, target_no_points):
    global poi
    if len(pointcloud) - target_no_points <= 0:
        return pointcloud
    else:
        pointcloud = np.array(pointcloud)
        cx = np.mean([point[0] for point in pointcloud])
        cy = np.mean([point[1] for point in pointcloud])
        cz = np.mean([point[2] for point in pointcloud])
        poi = np.array([cx,cy,cz])
        dists = np.array(map(dist, pointcloud))
        sorted_dists_ind = np.argpartition(dists, target_no_points-1)
        return pointcloud[sorted_dists_ind][:target_no_points]

# Another way to reach a specified number of points is to add points to "smaller" pointclouds
# The idea is to add points at the exact same position as others in the pointcloud
# The problem here:
# Do these extra points add bias?
def addPointsToPointcloud(pointcloud, target_no_points):
    while len(pointcloud) < target_no_points:
        pointcloud.append(random.choice(pointcloud))
    return pointcloud

def main():
    if len(sys.argv) < 2:
        print("Not enough arguments")
        return

    # Note to self, currently using this csv:
    # /home/gstavrinos/catkin_ws/src/new_hpr/pointcloud2_segments_tools/dataset2.csv
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
        for line in tqdm(if_.readlines()):
            dt, lbl = line.rsplit(",", 1)
            i = 0
            tmp = []
            for d in re.split(r"\(([^)]+)\)", dt):
                # Skip every other element, because we get "," characters between each tuple
                if i % 2 != 0:
                    d = d.split(",")
                    tmp.append(np.array([float(d[0]), float(d[1]), float(d[2])]))
                i += 1
            if len(tmp) > 20:
                tmp = addPointsToPointcloud(tmp, 1024)
                tmp = clearPointcloudOuterPoints(tmp, 1024)
                data.append(tmp)
                labels.append(int(lbl))

    print("---")
    print("Generating h5 file...")
    if os.path.exists(output_file):
        os.remove(output_file)
    with h5py.File(output_file, "a") as h5out:
        h5out.create_dataset("data", data=data, compression="gzip", compression_opts=4, dtype="float32")
        h5out.create_dataset("label", data=labels, compression="gzip", compression_opts=4, dtype="int")
    print("All done! bb")


main()