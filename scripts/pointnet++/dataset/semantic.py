#!/usr/bin/env python
import os
import sys
import h5py
ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import numpy as np
import utils.provider as provider

class Dataset():

    def __init__(self, filename, npoints=1024, box_size=10, dropout_max=0, training=True):
        """Create a dataset holder
            npoints (int): Defaults to 8192. The number of point in each input
            color (bool): Defaults to True. Whether to use colors or not
            box_size (int): Defaults to 10. The size of the extracted cube.
            path (float): Defaults to "dataset/semantic_data/". 
            dropout_max (float): Defaults to 0.875. Maximum dropout to apply on the inputs.
            accept_rate (float): Minimum rate (between 0.0 and 1.0) of points in the box to accept it. E.g : npoints = 100, then you need at least 50 points.
        """
        # Dataset parameters
        self.filename = filename
        self.npoints = npoints
        self.box_size = box_size
        self.dropout_max = dropout_max
        self.num_classes = 2

        # Load the data
        self.load_data()

        # Precompute the random scene probabilities
        self.compute_random_scene_index_proba()
        
        # Prepare the points weights if it is a training set
        if training:
            # Compute the weights
            labelweights = np.zeros(2)
            # First, compute the histogram of each labels
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(3))
                labelweights += tmp
            # Then, an heuristic gives the weights : 1/log(1.2 + probability of occurrence)
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(2)

    def load_data(self):
        print("Loading semantic data...")
        self.scene_points_list = list()
        self.semantic_labels_list = list()
        f = h5py.File(self.filename, "r")
        data_points = f["data"][:]
        data_labels = f["label"][:]
        # sort according to x to speed up computation of boxes and z-boxes
        # print np.shape(data_points)
        # print np.shape(data_points[:,0])
        # print np.shape(data_points[:,:,0])
        # sort_idx = np.argsort(data_points[:,:,0])
        # print np.shape(sort_idx)
        # print np.shape(data_labels)
        # data_points = data_points[sort_idx,0]
        # data_labels = data_labels[sort_idx]
        # self.scene_points_list = data_points
        # self.semantic_labels_list = data_labels.astype(np.int8)
        # 
        # 
        # print np.shape(data_points)
        # print np.shape(data_points[:,0])
        # print np.shape(data_points[:,:,0])
        for i in range(len(data_points)):
            #pc = data_points[i]
            #sort_idx = np.argsort(pc[:,0])
            #print np.shape(sort_idx)
            #print np.shape(data_labels)
            #pc = data_points[i,sort_idx]
            #dl = data_labels[i,sort_idx]
            #self.scene_points_list.append(data_points[i])
            #self.semantic_labels_list.append(data_labels[i].astype(np.int8))
            break

        self.scene_points_list = data_points
        #self.semantic_labels_list = data_labels.astype(np.int8)
        for i in range(len(data_labels)):
            self.semantic_labels_list.append([data_labels[i]] * self.npoints)

        # Set min to (0,0,0)
        self.scene_max_list = list()
        self.scene_min_list = list()
        self.raw_scene_min_list = list()
        for i in range(len(self.scene_points_list)):
            self.raw_scene_min_list.append(np.min(self.scene_points_list[i],axis=0))
            self.scene_points_list[i] = self.scene_points_list[i]-np.min(self.scene_points_list[i], axis=0)
            self.scene_max_list.append(np.max(self.scene_points_list[i],axis=0))
            self.scene_min_list.append(np.min(self.scene_points_list[i],axis=0))

    def __getitem__(self, index):
        """
        input : index of a scene
        output : the whole scene of npointsx3 (xyz) points of the scene and their labels, and colors if colors are used
        """
        point_set = self.scene_points_list[index]
        labels = self.semantic_labels_list[index].astype(np.int32)
        return point_set, labels

    def next_batch(self,batch_size,augment=True,dropout=True):
        batch_data = []
        batch_label = []
        batch_weights = []
        for _ in range(batch_size):
            data, label, weights = self.next_input(dropout)
            batch_data.append(data)
            batch_label.append(label)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        # Optional batch augmentation
        if augment:
            batch_data = provider.rotate_point_cloud(batch_data)

        return batch_data, batch_label, batch_weights

    def next_input(self,dropout=False,sample=True, verbose=False, visu=False, predicting=False):
        input_ok = False
        count_try = 0

        # Try to find a non-empty cloud to process
        while not input_ok:
            count_try += 1
            # Randomly choose a scene, taking account that some scenes contains more points than others
            scene_index = self.get_random_scene_index()
            
            # Randomly choose a seed
            scene = self.scene_points_list[scene_index] # [[x,y,z],...[x,y,z]]
            scene_labels = np.array(self.semantic_labels_list[scene_index]).astype(np.int8)

            # Random (on points)
            seed_index = np.random.randint(0,len(scene))
            seed = scene[seed_index] # [x,y,z]

            # Random (space)
            #scene_max = np.max(scene,axis=0)
            #scene_min = np.min(scene,axis=0)
            #seed = np.random.uniform(scene_min,scene_max,3)

            # Crop a z-box around that seed
            scene_extract_mask = self.extract_z_box(seed, scene, scene_index)
            # Verify the cloud is not empty
            if np.sum(scene_extract_mask) == 0:
                if verbose:
                    print ("Warning : empty box")
                continue
            else:
                if verbose:
                    print ("There are %i points in the box" %(np.sum(scene_extract_mask)))
                input_ok = True
                if visu:
                    return scene_index, scene_extract_mask, np.histogram(scene_labels[scene_extract_mask], range(10))[0], seed

            data = scene[scene_extract_mask]
            labels = scene_labels[scene_extract_mask]

        if sample:
            if len(data) - self.npoints > 0:
                trueArray = np.ones(self.npoints, dtype = bool)
                falseArray = np.zeros(len(data) - self.npoints, dtype = bool)
                sample_mask = np.concatenate((trueArray, falseArray), axis=0)
                np.random.shuffle(sample_mask)
            else:
                # Not enough points, recopy the data until there are enough points
                sample_mask = np.arange(len(data))
                while (len(sample_mask) < self.npoints):
                    sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
                sample_mask = sample_mask[np.arange(self.npoints)]
            raw_data = data[sample_mask]

            # Center the box in 2D
            data = self.center_box(raw_data)

            labels = labels[sample_mask]

            # Compute the weights
            weights = self.labelweights[labels]

            # Optional dropout
            if dropout:
                drop_index = self.input_dropout(data)
                weights[drop_index] = 0

        if predicting:
            return scene_index, data, raw_data+self.raw_scene_min_list[scene_index], labels, weights
        else:
            return data, labels, weights

    def get_random_scene_index(self):
        #return np.random.randint(0,len(self.scene_points_list)) # Does not take into account the scene number of points
        return np.random.choice(np.arange(0, len(self.scene_points_list)), p=self.scenes_proba)
                

    def compute_random_scene_index_proba(self):
        # Precompute the probability of picking a point
        # in a given scene. This is useful to compute the scene index later,
        # in order to pick more seeds in bigger scenes
        self.scenes_proba = []
        total = self.get_total_num_points()
        proba = 0
        for scene_index in range(len(self)):
            proba = float(len(self.scene_points_list[scene_index]))/float(total)
            self.scenes_proba.append(proba)

    
    def center_box(self,data):
        # Shift the box so that z= 0 is the min and x=0 and y=0 is the center of the box horizontally
        box_min = np.min(data, axis=0)
        shift = np.array([box_min[0]+self.box_size/2, box_min[1]+self.box_size/2, box_min[2]]) 
        return data-shift

    def extract_box(self,seed,scene):
        # 10 meters seems intuitively to be a good value to understand the scene, we must test that

        box_min = seed - [self.box_size/2, self.box_size/2, self.box_size/2]
        box_max = seed + [self.box_size/2, self.box_size/2, self.box_size/2]
        
        i_min = np.searchsorted(scene[:,0], box_min[0])
        i_max = np.searchsorted(scene[:,0], box_max[0])
        mask = np.sum((scene[i_min[0]:i_max,:] >= box_min)*(scene[i_min[0]:i_max,:] <= box_max),axis=1) == 3
        mask = np.hstack((np.zeros(i_min, dtype=bool), mask, np.zeros(len(scene)-i_max, dtype=bool)))
        print(mask.shape)
        return mask

    def extract_z_box(self,seed,scene, scene_idx):
        ## TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        # 2D crop, takes all the z axis

        scene_max = self.scene_max_list[scene_idx]
        scene_min = self.scene_min_list[scene_idx]
        scene_z_size = scene_max[2]-scene_min[2]
        box_min = seed - [self.box_size/2, self.box_size/2, scene_z_size]
        box_max = seed + [self.box_size/2, self.box_size/2, scene_z_size]

        i_min = np.searchsorted(scene[:,0], box_min[0])
        i_max = np.searchsorted(scene[:,0], box_max[0])
        mask = np.sum((scene[i_min:i_max,:] >= box_min)*(scene[i_min:i_max,:] <= box_max),axis=1) == 3
        mask = np.hstack((np.zeros(i_min, dtype=bool), mask, np.zeros(len(scene)-i_max, dtype=bool)))

        #mask = np.sum((scene>=box_min)*(scene<=box_max),axis=1) == 3
        return mask

    def input_dropout(self,input):
        dropout_ratio = np.random.random()*self.dropout_max
        drop_index = np.where(np.random.random((input.shape[0]))<=dropout_ratio)[0]
        return drop_index
    
    def get_total_num_points(self):
        # total = 0
        # for scene_index in range(len(self)):
        #     total += len(self.scene_points_list[scene_index])
        #print "len = " + str(len(self))
        #print "total = " + str(total)
        #print "shape = " + str(np.shape(self.scene_points_list))
        #return total
        s = np.shape(self.scene_points_list)
        if len(s) == 3:
            return s[0] * s[1]
        else:
            return s[0]

    def get_num_batches(self, batch_size):
        return int(self.get_total_num_points()/(batch_size*self.npoints))

    def __len__(self):
        return len(self.scene_points_list)
    
    def get_hist(self):
        labelweights = np.zeros(2)
            # First, compute the histogram of each labels
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg,range(3))
            labelweights += tmp
        return labelweights
    
    # def get_list_classes_str(self):
    #     return "unlabeled, man-made terrain, natural terrain, high vegetation, low vegetation, buildings, hard scape, scanning artefacts, cars"

    # def get_data_filenames(self):
    #     return self.data_filenames
        
    # def get_scene_shape(self, scene_index):
    #     return self.scene_points_list[scene_index].shape
        
    # def get_scene(self, scene_index):
    #     return self.scene_points_list[scene_index]
