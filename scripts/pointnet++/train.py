#!/usr/bin/env python
import os
import sys
import time
import json
import importlib
import numpy as np
import tensorflow as tf
import models.model as model
import multiprocessing as mp
from datetime import datetime
import utils.metric as metric
import dataset.semantic as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

JSON_DATA = open("params.json").read()
PARAMS = json.loads(JSON_DATA)

GPU_INDEX = PARAMS["gpu"]
MOMENTUM = PARAMS["momentum"]
NUM_POINT = PARAMS["num_point"]
MAX_EPOCH = PARAMS["max_epoch"]
OPTIMIZER = PARAMS["optimizer"]
BATCH_SIZE = PARAMS["batch_size"]
DECAY_STEP = PARAMS["decay_step"]
BASE_LEARNING_RATE = PARAMS["learning_rate"]
DECAY_RATE = PARAMS["learning_rate_decay_rate"]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

LOG_DIR = PARAMS["logdir"]
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

# Batch normalisation
BN_INIT_DECAY = PARAMS["bn_init_decay"]
BN_DECAY_DECAY_RATE = PARAMS["bn_decay_decay_rate"]
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = PARAMS["bn_decay_clip"]

TRAIN_DATASET = None
TEST_DATASET = None

# Dummy initialization here. NUM_CLASSES is updated just before starting training
NUM_CLASSES = 2

# Start logging
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "w")

EPOCH_CNT = 0

def log_string(out_str):
    LOG_FOUT.write(out_str+"\n")
    LOG_FOUT.flush()
    print(out_str)

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a "halt".
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = round(float(progress),2)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def get_learning_rate(batch):
    """Compute the learning rate for a given batch size and global parameters
    
    Args:
        batch (tf.Variable): the batch size
    
    Returns:
        scalar tf.Tensor: the decayed learning rate
    """

    learning_rate = tf.compat.v1.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    """Compute the batch normalisation exponential decay
    
    Args:
        batch (tf.Variable): the batch size
    
    Returns:
        scalar tf.Tensor: the batch norm decay
    """
    
    bn_momentum = tf.compat.v1.train.exponential_decay(
    BN_INIT_DECAY,
    batch*BATCH_SIZE,
    BN_DECAY_DECAY_STEP,
    BN_DECAY_DECAY_RATE,
    staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_batch(split):
    np.random.seed()
    if split=="train":
        return TRAIN_DATASET.next_batch(BATCH_SIZE,True,True)
    else:
        return TEST_DATASET.next_batch(BATCH_SIZE,False,False)

def fill_queues(stack_train,stack_test,maxsize_train,maxsize_test):
    pool = mp.Pool(processes=mp.cpu_count())
    launched_train = 0
    launched_test = 0
    results_train = []
    results_test = []
    # Launch as much as n
    while True:
        if stack_train.qsize()+launched_train<maxsize_train:
            results_train.append(pool.apply_async(get_batch,args=("train",)))
            launched_train += 1
        elif stack_test.qsize()+launched_test<maxsize_test:
            results_test.append(pool.apply_async(get_batch,args=("test",)))
            launched_test += 1
        for p in results_train:
            if p.ready():
                stack_train.put(p.get())
                results_train.remove(p)
                launched_train -= 1
        for p in results_test:
            if p.ready():
                stack_test.put(p.get())
                results_test.remove(p)
                launched_test -= 1
        # Stability
        time.sleep(0.01)

def init_stacking():
    with tf.device("/cpu:0"):
        # Queues that contain several batches in advance
        num_train_batches = TRAIN_DATASET.get_num_batches(BATCH_SIZE)
        num_test_batches = TEST_DATASET.get_num_batches(BATCH_SIZE)
        stack_train = mp.Queue(num_train_batches)
        stack_test = mp.Queue(num_test_batches)
        stacker = mp.Process(target=fill_queues, args=(stack_train,stack_test,num_train_batches,num_test_batches))
        stacker.start()
        return stacker, stack_test, stack_train


def train():
    global TRAIN_DATASET, TEST_DATASET, NUM_CLASSES
    if len(sys.argv) < 3:
        print("Not enough arguments. Please provide training and test sets.")
        return
    # Currently using:
    # /home/gstavrinos/catkin_ws/src/new_hpr/pointcloud2_segments_tools/dataset2_training_75.h5
    # /home/gstavrinos/catkin_ws/src/new_hpr/pointcloud2_segments_tools/dataset2_testing_75.h5
    TRAIN_DATASET = data.Dataset(sys.argv[1], npoints=NUM_POINT, training=True)
    TEST_DATASET = data.Dataset(sys.argv[2], npoints=NUM_POINT, training=False)
    NUM_CLASSES = TRAIN_DATASET.num_classes
    with tf.Graph().as_default():
        stacker, stack_test, stack_train = init_stacking()

        with tf.device("/gpu:"+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the "batch" parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar("bn_decay", bn_decay)

            print ("--- Get model and loss")
            # Get model and loss 
            pred, end_points = model.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, hyperparams=PARAMS, bn_decay=bn_decay)
            loss = model.get_loss(pred, labels_pl, smpws_pl, end_points)
            tf.summary.scalar("loss", loss)

            # Compute accuracy
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar("accuracy", accuracy)

            # Computer mean intersection over union
            mean_intersection_over_union, update_iou_op = tf.compat.v1.metrics.mean_iou(tf.to_int32(labels_pl), tf.to_int32(tf.argmax(pred, 2)), NUM_CLASSES)
            tf.summary.scalar("mIoU", tf.to_float(mean_intersection_over_union))

            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar("learning_rate", learning_rate)
            if OPTIMIZER == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == "adam":
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()

        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, "train"), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, "test"), sess.graph)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer()) # important for mIoU

        ops = {"pointclouds_pl": pointclouds_pl,
               "labels_pl": labels_pl,
               "smpws_pl": smpws_pl,
               "is_training_pl": is_training_pl,
               "pred": pred,
               "loss": loss,
               "train_op": train_op,
               "merged": merged,
               "step": batch,
               "end_points": end_points,
               "update_iou": update_iou_op}

        training_loop(sess, ops, saver, stacker, train_writer, stack_train, test_writer, stack_test)

def training_loop(sess, ops, saver, stacker, train_writer, stack_train, test_writer, stack_test):
    best_acc = -1
    # Train for MAX_EPOCH epochs
    try:
        for epoch in range(MAX_EPOCH):
            log_string("**** EPOCH %03d ****" % (epoch))
            sys.stdout.flush()

            # Train one epoch
            train_one_epoch(sess, ops, train_writer, stack_train)

            # Evaluate, save, and compute the accuracy
            if epoch % 5 == 0:
                acc = eval_one_epoch(sess, ops, test_writer, stack_test) 
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
    except Exception as e:
        print e
    finally:
        # Kill the process, close the file and exit
        stacker.terminate()
        LOG_FOUT.close()
        sys.exit()


def train_one_epoch(sess, ops, train_writer, stack):
    """Train one epoch
    
    Args:
        sess (tf.Session): the session to evaluate Tensors and ops
        ops (dict of tf.Operation): contain multiple operation mapped with with strings
        train_writer (tf.FileSaver): enable to log the training with TensorBoard
        compute_class_iou (bool): it takes time to compute the iou per class, so you can disable it here
    """

    is_training = True

    num_batches = TRAIN_DATASET.get_num_batches(BATCH_SIZE)

    log_string(str(datetime.now()))
    update_progress(0)
    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)
    # Train over num_batches batches
    for batch_idx in range(num_batches):
        # Refill more batches if empty
        progress = float(batch_idx)/float(num_batches)
        update_progress(round(progress,2))
        batch_data, batch_label, batch_weights = stack.get()

        # Get predicted labels
        feed_dict = {ops["pointclouds_pl"]: batch_data,
                     ops["labels_pl"]: batch_label,
                     ops["smpws_pl"]: batch_weights,
                     ops["is_training_pl"]: is_training,}
        summary, step, _, loss_val, pred_val, _ = sess.run([ops["merged"], ops["step"],
                                                         ops["train_op"], ops["loss"], ops["pred"], ops["update_iou"]], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        
        # Update metrics
        for i in range(len(pred_val)):
            for j in range(len(pred_val[i])):
                confusion_matrix.count_predicted(batch_label[i][j], pred_val[i][j])
        loss_sum += loss_val
    update_progress(1)
    log_string("Mean Loss: %f" % (loss_sum / float(num_batches)))
    log_string("Overall Accuracy : %f" %(confusion_matrix.get_overall_accuracy()))
    log_string("Average IoU : %f" %(confusion_matrix.get_average_intersection_union()))
    iou_per_class = confusion_matrix.get_intersection_union_per_class()
    for i in range(NUM_CLASSES):
        log_string("IoU of %s : %f" % (i,iou_per_class[i]))

def eval_one_epoch(sess, ops, test_writer, stack):
    """Evaluate one epoch
    
    Args:
        sess (tf.Session): the session to evaluate tensors and operations
        ops (tf.Operation): the dict of operations
        test_writer (tf.compat.v1.summary.FileWriter): enable to log the evaluation on TensorBoard
    
    Returns:
        float: the overall accuracy computed on the test set
    """

    global EPOCH_CNT

    is_training = False

    num_batches = TEST_DATASET.get_num_batches(BATCH_SIZE)

    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

    log_string(str(datetime.now()))

    log_string("---- EPOCH %03d EVALUATION ----"%(EPOCH_CNT))

    update_progress(0)

    for batch_idx in range(num_batches):
        progress = float(batch_idx)/float(num_batches)
        update_progress(round(progress,2))
        batch_data, batch_label, batch_weights = stack.get()
        
        feed_dict = {ops["pointclouds_pl"]: batch_data,
                     ops["labels_pl"]: batch_label,
                     ops["smpws_pl"]: batch_weights,
                     ops["is_training_pl"]: is_training}
        summary, step, loss_val, pred_val = sess.run([ops["merged"], ops["step"],
                                                      ops["loss"], ops["pred"]], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        
        # Update metrics
        for i in range(len(pred_val)):
            for j in range(len(pred_val[i])):
                confusion_matrix.count_predicted(batch_label[i][j], pred_val[i][j])
        loss_sum += loss_val
    
    update_progress(1)

    iou_per_class = confusion_matrix.get_intersection_union_per_class()

    # Display metrics
    log_string("mean loss: %f" % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" %(confusion_matrix.get_overall_accuracy()))
    log_string("Average IoU : %f" %(confusion_matrix.get_average_intersection_union()))
    for i in range(0,NUM_CLASSES):
        log_string("IoU of %s : %f" % (i,iou_per_class[i]))
    
    EPOCH_CNT += 5
    return confusion_matrix.get_overall_accuracy()

if __name__ == "__main__":
    log_string("pid: %s"%(str(os.getpid())))
    train()
