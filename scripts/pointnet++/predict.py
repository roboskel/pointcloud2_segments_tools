#!/usr/bin/env python

# General
import os
import sys
import json
import random
import numpy as np

# ROS
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from pointcloud_msgs.msg import PointCloud2_Segments

# TF and PointNet++
import tensorflow as tf
import models.model as model

JSON_DATA = open("params.json").read()
PARAMS = json.loads(JSON_DATA)

# Currently using:
# "/home/gstavrinos/catkin_ws/src/new_hpr/pointcloud2_segments_tools/scripts/pointnet++/logs/log/model.ckpt"
if len(sys.argv) < 2:
    print("Not enough arguments. Please provide a model checkpoint.")
    exit()
CHECKPOINT = sys.argv[1]
GPU_INDEX = 0
NUM_POINT = 2**15
NUM_CLASSES = 2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

tf_session = None

ops = dict()

input_batch = []

poi = None

pub = None

points_helper_arr = np.zeros((NUM_POINT,), dtype=[
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("r", np.uint8),
    ("g", np.uint8),
    ("b", np.uint8)])

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
    if len(pointcloud) <= target_no_points:
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
        pointcloud = np.append(pointcloud, np.reshape(random.choice(pointcloud), (1,3)), axis=0)
    return pointcloud

def fix(pointcloud, target_no_points=NUM_POINT):
    pointcloud = clearPointcloudOuterPoints(pointcloud, target_no_points)
    pointcloud = addPointsToPointcloud(pointcloud, target_no_points)
    return pointcloud

def colourToClass(c):
    if c == 0:
        return (255,0,0)
    return (0,0,255)

def get_xyzrgb(cloud):
    points = np.zeros(cloud.shape + (6,))
    points[...,0] = cloud["x"]
    points[...,1] = cloud["y"]
    points[...,2] = cloud["z"]
    points[...,3] = cloud["vectors"]
    points[...,4] = cloud["g"]
    points[...,5] = cloud["b"]
    return points

def predictCallback(msg):
    global tf_session, ops, pub, points_helper_arr
    for pointcloud in msg.clusters:
        data = fix(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pointcloud))

        # Prediction
        pred_labels = predict_one_input(data)
        colours = np.array(map(colourToClass, pred_labels))
        points_helper_arr["x"] = data[:,0]
        points_helper_arr["y"] = data[:,1]
        points_helper_arr["z"] = data[:,2]
        points_helper_arr["r"] = colours[:,0]
        points_helper_arr["g"] = colours[:,1]
        points_helper_arr["b"] = colours[:,2]
        res = ros_numpy.msgify(PointCloud2, points_helper_arr)
        res.header.stamp = rospy.Time.now()
        res.header.frame_id = "base_link"
        pub.publish(res)
        #print(pred_labels)
        #print(np.count_nonzero(pred_labels))
        #print(len(pred_labels) - np.count_nonzero(pred_labels))

def predict_one_input(data):
    global tf_session, ops
    is_training = False
    batch_data = np.array([data]) # 1 x NUM_POINT x 3
    feed_dict = {ops["pointclouds_pl"]: batch_data,
                 ops["is_training_pl"]: is_training}
    pred_val = tf_session.run([ops["pred"]], feed_dict=feed_dict)
    pred_val = pred_val[0][0] # NUMPOINTSx9
    pred_val = np.argmax(pred_val,1)
    return pred_val

def initTF():
    global tf_session, ops
    with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl, _ = model.placeholder_inputs(1, NUM_POINT)
        print (tf.shape(pointclouds_pl))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, _ = model.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, hyperparams=PARAMS)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    tf_session = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(tf_session, CHECKPOINT)
    print ("Model restored.")

    ops = {"pointclouds_pl": pointclouds_pl,
           "is_training_pl": is_training_pl,
           "pred": pred}

def init():
    global pub
    initTF()
    rospy.init_node("ros_prediction_test")
    pub = rospy.Publisher("ros_pointnet_prediction/pointcloud", PointCloud2, queue_size=1)
    rospy.Subscriber("/laserscan_stacker/scans", PointCloud2_Segments, predictCallback)
    rospy.spin()

if __name__ == "__main__":
    print ("pid: %s"%(str(os.getpid())))
    with tf.Graph().as_default():
        init()
