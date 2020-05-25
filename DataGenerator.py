import numpy as np
import cv2 as cv
import sys
import os
import argparse
import json
import shutil
import csv
import os
import keras
import pickle

def generate_indices(time_history,train_size):
# All we need to do is store the indices
    FrameIndices=[]
    SpeedIndices=[]

    for j in range(0,train_size-time_history+1):
        tempFrame=[]
        for i in range(0,time_history):
            tempFrame.append(i+j)
        if time_history==2:
            SpeedIndices.append(max(tempFrame))
        else:
            SpeedIndices.append(max(tempFrame))
        FrameIndices.append(tempFrame)
    return FrameIndices,SpeedIndices

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, frame,frame_augmented,speed_list,FrameIndices,SpeedIndices,train_size,indexes = None, validation_mode = False,augment=False):

        self.frame_augmented = frame_augmented
        self.frame = frame
        self.speed_list=speed_list
        self.FrameIndices=FrameIndices
        self.SpeedIndices=SpeedIndices
        self.augment=augment

        if indexes is None:
            with indexes is None:
                 with h5py.File(self.hdf5_path, "r") as f:
                        self.indexes=np.arrange(train_size)
        else:
            self.indexes=indexes
        self.batch_size=batch_size
        self.validation_mode=validation_mode
        if self.validation_mode==False:
            print("shuffling")
            self.on_epoch_end()



    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.indexes)/self.batch_size))

    def __getitem__(self,index):
        # Generate indexes of the batch
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate the data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        # updates indexes after each epoch
        if self.validation_mode == False:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        # Generates data containing batch samples
        indexes=list(indexes)
        indexes.sort()## We are making the Generator Class to read the video and the speed file
        speeds=[]
        op_flow=[]
        for index in indexes:
            speeds.append(self.speed_list[self.SpeedIndices[index]])
            frames=[]
            for frameIndex in self.FrameIndices[index]:
                r=(np.random.random_integers(1,100))
                if r%2==0 and self.augment:
                  frames.append(self.frame_augmented[frameIndex])
                else:
                  frames.append(self.frame[frameIndex])
            op_flow.append(frames)
        op_flow=np.array(op_flow)
        speeds=np.array(speeds)
        return [op_flow,speeds]
