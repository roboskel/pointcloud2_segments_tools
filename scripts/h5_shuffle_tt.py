#!/usr/bin/env python
import os
import sys
import h5py
import random
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return

    # Note to self, currently using this csv:
    # /home/gstavrinos/catkin_ws/src/new_hpr/pointcloud2_segments_tools/dataset0.h5
    input_file = sys.argv[1]
    percentage = float(sys.argv[2])
    path, f = sys.argv[1].rsplit(os.sep, 1)
    training_output_h5 = path + os.sep + f.rsplit(".", 1)[0] + "_training_" + str(int(percentage*100)) + ".h5"
    training_output_txt = path + os.sep + f.rsplit(".", 1)[0] + "_training_" + str(int(percentage*100)) + ".txt"
    testing_output_h5 = path + os.sep + f.rsplit(".", 1)[0] + "_testing_" + str(int(percentage*100)) + ".h5"
    testing_output_txt = path + os.sep + f.rsplit(".", 1)[0] + "_testing_" + str(int(percentage*100)) + ".txt"
    print("---")
    print("Your training h5 will be written in:")
    print(training_output_h5)
    print("Your training txt will be written in:")
    print(training_output_txt)
    print("Your testing h5 will be written in:")
    print(testing_output_h5)
    print("Your testing txt will be written in:")
    print(testing_output_txt)
    print("---")

    test_data = []
    train_data = []
    test_labels = []
    train_labels = []

    print("Processing h5 file...")

    with h5py.File(input_file, "r") as f:
        classes, num_elements = np.unique(f["label"], return_counts=True)
        print("Classes:")
        print(classes)
        print("Number of elements:")
        print(num_elements)
        for c in range(len(classes)):
            class_indices = [i for i in np.arange(0, sum(num_elements)) if f["label"][i] == c]
            class_prcnt = percentage * num_elements[c]

            training_indices = np.random.choice(class_indices, int(class_prcnt), replace=False)
            training_indices = np.sort(training_indices)

            testing_indices = [i for i in class_indices if i not in training_indices]

            #testing_indices = np.sort(testing_indices)
            train_data.append(f["data"][training_indices][0])
            test_data.append(f["data"][testing_indices][0])
            train_labels.append(f["label"][training_indices][0])
            test_labels.append(f["label"][testing_indices][0])
            print("Completed class " + str(c))

    print("---")
    print("Generating training h5 and txt files...")
    print train_data
    h5out = h5py.File(training_output_h5, "w")
    h5out.create_dataset("data", data=train_data, compression="gzip", compression_opts=4, dtype="float32")
    h5out.create_dataset("label", data=train_labels, compression="gzip", compression_opts=4, dtype="int")
    h5out.close()
    with open(training_output_txt, "w") as f:
        f.write(training_output_h5)
    print("---")
    print("Generating testing h5 and txt files...")
    h5out = h5py.File(testing_output_h5, "w")
    h5out.create_dataset("data", data=test_data, compression="gzip", compression_opts=4, dtype="float32")
    h5out.create_dataset("label", data=test_labels, compression="gzip", compression_opts=4, dtype="int")
    h5out.close()
    with open(testing_output_txt, "w") as f:
        f.write(testing_output_h5)
    print("All done! bb")


main()