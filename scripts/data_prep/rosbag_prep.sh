#!/bin/env bash

# Base path: where your rosbags are located (+ the ones to be generated)
base_path="/home/gstavrinos/walking_rosbags/take2/"
# Starting rosbags: where the starting rosbag are located
# (subdir for base_path)
starting_rosbags="starting_rosbags/"
# Humans only: where the rosbags that include only humans will be saved
humans_only="humans_only/"
# Walls only: where the rosbags that include only walls will be saved
walls_only="walls_only/"
# Laserscan only: the name of the folder for laserscan only rosbags
# (subdir for both humans_only and walls_only)
laserscan_only="laserscan_only/"
# Pointcloud segments: the name of the folder for laserscan+pointcloud2_segments rosbags
# (subdir for both humans_only and walls_only)
pointcloud2_segments="pointcloud2_segments/"
# Pointcloud segments one cluster: the name of the folder for laserscan+pointcloud2_segments rosbags
# in which all walls are considered a single cloud 
# (subdir only for walls_only)
pointcloud2_segments_one_cluster="pointcloud2_segments_one_cluster/"

function ls_only(){
    for bag in $base_path$starting_rosbags*.bag
    do
        filename=$(basename $bag)
        if [ "$1" == "$humans_only" ]
        then
            roslaunch pointcloud2_segments_tools laserscan_filter_humans_only.launch &
        else
            roslaunch pointcloud2_segments_tools laserscan_filter_walls_only.launch &
        fi
        sleep 1
        rosbag record /radio_cam/depth_registered/image_raw /radio_cam/rgb/image_raw /scan /scan_filtered /tf /laserscan_stacker/ -O $base_path$1$laserscan_only$filename &
        sleep 1
        rosbag play $bag --clock
        # Not needed with a blocking rosbag play
        # duration=$(rosbag info -y -k duration $bag)
        # echo "Sleeping for $duration seconds"
        # sleep $duration
        sleep 2
        rosnode kill --all
        killall -9 rosmaster
        sleep 1
    done
}

function pc2_seg(){
    for bag in $base_path$1$laserscan_only*.bag
    do
        filename=$(basename $bag)
        roslaunch laserscan_stacker laserscan_stacker.launch &
        sleep 5
        # Kill laserscan_filter that is included in the laserscan_stacker.launch
        # to avoid conflicts with the recorded scan_filtered topic
        rosnode kill laserscan_filter
        sleep 1
        if [ "$1" == "$humans_only" ]
        then
            rosbag record /radio_cam/depth_registered/image_raw /radio_cam/rgb/image_raw /scan /scan_filtered /laserscan_stacker/scans /tf -O $base_path$1$pointcloud2_segments$filename &
        else
            rosbag record /radio_cam/depth_registered/image_raw /radio_cam/rgb/image_raw /scan /scan_filtered /laserscan_stacker/scans /tf -O $base_path$1$pointcloud2_segments_one_cluster$filename &
        fi
        sleep 1
        rosbag play $bag --clock
        # Not needed with a blocking rosbag play
        # duration=$(rosbag info -y -k duration $bag)
        # echo "Sleeping for $duration seconds"
        # sleep $duration
        sleep 2
        rosnode kill --all
        killall -9 rosmaster
        sleep 1
    done
}

function pc2_seg_clustered() {
    for bag in $base_path$walls_only$laserscan_only*.bag
    do
        echo
        filename=$(basename $bag)
        roslaunch laserscan_stacker laserscan_stacker.launch &
        sleep 5
        roslaunch pointcloud2_clustering pointcloud2_clustering.launch &
        sleep 5
        # Kill laserscan_filter that is included in the laserscan_stacker.launch
        # to avoid conflicts with the recorded scan_filtered topic
        rosnode kill laserscan_filter
        sleep 1
        rosbag record /radio_cam/depth_registered/image_raw /radio_cam/rgb/image_raw /scan /scan_filtered /pointcloud2_clustering/clusters /tf -O $base_path$walls_only$pointcloud2_segments$filename &
        sleep 1
        rosbag play $bag --clock
        # Not needed with a blocking rosbag play
        # duration=$(rosbag info -y -k duration $bag)
        # echo "Sleeping for $duration seconds"
        # sleep $duration
        sleep 2
        rosnode kill --all
        killall -9 rosmaster
        sleep 1
    done
}

# Laserscan only for humans
ls_only $humans_only
# Laserscan only for walls
ls_only $walls_only
# Pointcloud2_Segments for humans
pc2_seg $humans_only
# Pointcloud2_Segments for walls (one cluster)
pc2_seg $walls_only
# Pointcloud2_Segments for walls (multiple clusters)
pc2_seg_clustered
