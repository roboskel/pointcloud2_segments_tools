#!/usr/bin/env python

# General
import os
import random
import numpy as np

# ROS
import rospy
import ros_numpy
from pointcloud_msgs.msg import PointCloud2_Segments

# TF and PointNet++
import model
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_CLASSES = 2
BATCH_SIZE = 1 #16
NUM_POINTS = 2**15

tf_session = None

ops = dict()

input_batch = []

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

def fix(pointcloud, target_no_points):
    pointcloud = clearPointcloudOuterPoints(pointcloud, target_no_points)
    pointcloud = addPointsToPointcloud(pointcloud, target_no_points)
    return [pointcloud]

def pc2s_callback(msg):
    global tf_session, input_batch
    batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES))
    for pointcloud in msg.clusters:
        xyz = ros_numpy.point_cloud2.get_xyz_points(ros_numpy.numpify(pointcloud))
        if len(input_batch) < BATCH_SIZE:
            if len(input_batch) == 0:
                input_batch = np.array(fix(xyz, NUM_POINTS))
            else:
                input_batch = np.append(input_batch, fix(xyz, NUM_POINTS), axis=0)
        else:
            input_batch = np.append(input_batch[1:], fix(xyz, NUM_POINTS), axis=0)
            feed_dict = {ops["pointclouds_pl"]: input_batch,
                        ops["is_training_pl"]: False}
            pred_val = tf_session.run([ops["pred"]], feed_dict=feed_dict)
            print np.shape(pred_val)
            asd
            # batch_pred_sum += pred_val
            # pred_val = np.argmax(batch_pred_sum, 1)
            # print pred_val

def init():
    initTF()
    rospy.init_node("ros_prediction_test")
    rospy.Subscriber("/laserscan_stacker/scans", PointCloud2_Segments, pc2s_callback)
    rospy.spin()

def initTF():
    global tf_session, ops
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        pred, end_points = model.get_model(pointclouds_pl, is_training_pl)
        model.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        tf_session = tf.Session(config=config)
        model_path = "/home/gstavrinos/libs/python/python2.7/pointnet2/log/model.ckpt"
        saver.restore(tf_session, model_path)
        ops = {"pointclouds_pl": pointclouds_pl,
            "is_training_pl": is_training_pl,
            "pred": pred}

if __name__ == "__main__":
    init()
