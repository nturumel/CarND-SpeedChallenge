import os
import numpy as np
import cv2 as cv
import h5py
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv2D, MaxPool2D, CuDNNGRU, GlobalMaxPool2D, Reshape, \
concatenate, Input, TimeDistributed, Dense, BatchNormalization, SpatialDropout2D, SpatialDropout1D, Dropout, GlobalAvgPool2D, Flatten
from keras import Model
from keras.applications import Xception
import keras.backend as k
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import csv

vidPath=r".\data\train\train.mp4"
imgStoragePath=r".\data\train\IMG"
speedcsv=r".\data\train\train.txt"

resize_size=(240, 320, 2)
resize_frame_size=(240, 320, 3)

def write_frames():
    IMGList=[]
    count=0
    vidCap=cv.VideoCapture(vidPath)
    success=True
    while success:
        if(count%1000==0) and count >0:
            print(count)
        success,frame1=vidCap.read()
        if (success == True):
            frame1=cv.cvtColor(frame1,cv.COLOR_BGR2RGB)
            frame1=cv.resize(frame1, (0,0), fx=0.5, fy=0.5)
            name=imgStoragePath
            filename=str(count)+r".jpg"
            IMGList.append(os.path.join(name, filename))
            cv.imwrite(os.path.join(name, filename),frame1)
            count+=1
        else:
            print("video reading completed")
            continue
    return count,IMGList

def reset_hdf5(num):
    hdf5_path=r".\data\train\"
    hdf5_path=os.path.join(hdf5_path,"train",str(num))
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("frames", shape = (1, resize_frame_size[0], resize_frame_size[1], resize_frame_size[2]),
                maxshape = (None, resize_frame_size[0], resize_frame_size[1], resize_frame_size[2]),
                chunks = (1, resize_frame_size[0], resize_frame_size[1], resize_frame_size[2]))
        f.create_dataset("speeds", shape = (1,1), maxshape = (None,1))
    f.close()

def opticalFlowDense(image_current,image_next):
    image_current=np.array(image_current)
    image_next=np.array(image_current)

    #convert to grayscale
    gray_current=cv.cvtColor(image_current,cv.COLOR_RGB2GRAY)
    gray_next=cv.cvtColor(image_next,cv.COLOR_RGB2GRAY)
    flow=cv.calcOpticalFlowFarneback(gray_current,gray_next,None,0.5,1,15,2,5,1.3,0)
    return flow


def augment(image_current, image_next):
    brightness=np.random.uniform(0.5,1.5)
    image_current_1 = cv.cvtColor(image_current,cv.COLOR_RGB2HSV)
    image_current_1[:,:,2] = image_current_1[:,:,2]*brightness

    image_next_1 = cv.cvtColor(image_next,cv.COLOR_RGB2HSV)
    image_next_1[:,:,2] = image_next_1[:,:,2]*brightness

    image_current_1 = cv.cvtColor(image_current_1,cv.COLOR_HSV2RGB)
    image_next_1 = cv.cvtColor(image_next_1,cv.COLOR_HSV2RGB)
    #image_current_1 = cv.normalize(image_current_1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    #image_next_1 = cv.normalize(image_next_1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    return image_current_1,image_next_1

def flipimage(image_current, image_next):
    image_current=cv.flip(image_current, 1)
    image_next=cv.flip(image_next,1)
    return image_current,image_next

def augment(image_current):
    brightness=np.random.uniform(0.5,1.5)
    image_current_1 = cv.cvtColor(image_current,cv.COLOR_RGB2HSV)
    image_current_1[:,:,2] = image_current_1[:,:,2]*brightness

    image_current_1 = cv.cvtColor(image_current_1,cv.COLOR_HSV2RGB)

    return image_current_1

def flipimage(image_current):
    image_current=cv.flip(image_current, 1)
    return image_current

def batch_def(Sequence_size,Batch_size):
    Num_batches=int(np.ceil(Sequence_size/Batch_size))
    print(Num_batches,' : Number of Batches')
    All_indices=[item for item in range(1,Sequence_size)]
    return All_indices,Num_batches


def generate_indices(Batch_num):
    return All_indices[Batch_num*Batch_size:(Batch_num+1)*Batch_size]

def write_hdf5(hdf5_path, frames, speeds):
    with h5py.File(hdf5_path) as f:
        print(len(frames), len(speeds))
        print(f["frames"], f["speeds"])
        f["frames"].resize(f["frames"].len() + len(frames), axis = 0)
        f["speeds"].resize(f["speeds"].len() + len(speeds), axis = 0)
        f["frames"][-len(frames):] = frames
        f["speeds"][-len(speeds):] = speeds

def write_frame_and_hdf5(num,first):
    reset_hdf5(num)
    hdf5_path=r".\data\train\"
    hdf5_path=os.path.join(hdf5_path,"train",str(num))
    speeds=[]

    if first==False:
        count,IMGList=write_frames()
        speeds=[]
        with open(speedcsv) as csvfile:
            reader=csv.reader(csvfile)
            for speed in reader:
                speeds.append(speed)
        count=0
        Sequence_size=len(speeds)
        All_indices,Num_batches=batch_def(Sequence_size,100)
        if num==1:
             for batch in range(0,Num_batches):
                #print(generate_indices(batch))
                indexes=generate_indices(batch)
                frames=[]
                speed=[]
                for index in indexes:
                    #print([indexes[i],indexes[i+1]])

                    if(count%1000==0) and count >0:
                        print(count)

                    frame2=cv.imread(IMGList[index])
                    frame2=cv.cvtColor(frame2,cv.COLOR_BGR2RGB)

                    frame2=augment(frame2)
                    frame2=flipimage(frame2)


                    frames.append(np.array(frame2))
                    speed.append(speeds[index])
                    count+=1

                frames=np.array(frames,dtype=np.uint8)
                speed=np.array(speed)
                speed=speed.astype(float)
                write_hdf5(hdf5_path, frames, speed)
            print("Completed Video Processing")
            return count

        if num==2:
             for batch in range(0,Num_batches):
                #print(generate_indices(batch))
                indexes=generate_indices(batch)
                frames=[]
                speed=[]
                for index in indexes:
                    #print([indexes[i],indexes[i+1]])

                    if(count%1000==0) and count >0:
                        print(count)

                    frame2=cv.imread(IMGList[index])
                    frame2=cv.cvtColor(frame2,cv.COLOR_BGR2RGB)

                    frame2=flipimage(frame2)


                    frames.append(np.array(frame2))
                    speed.append(speeds[index])
                    count+=1

                frames=np.array(frames,dtype=np.uint8)
                speed=np.array(speed)
                speed=speed.astype(float)
                write_hdf5(hdf5_path, frames, speed)
            print("Completed Video Processing")
            return count

        if num==3:
             for batch in range(0,Num_batches):
                #print(generate_indices(batch))
                indexes=generate_indices(batch)
                frames=[]
                speed=[]
                for index in indexes:
                    #print([indexes[i],indexes[i+1]])

                    if(count%1000==0) and count >0:
                        print(count)

                    frame2=cv.imread(IMGList[index])
                    frame2=cv.cvtColor(frame2,cv.COLOR_BGR2RGB)

                    frame2=augment(frame2)


                    frames.append(np.array(frame2))
                    speed.append(speeds[index])
                    count+=1

                frames=np.array(frames,dtype=np.uint8)
                speed=np.array(speed)
                speed=speed.astype(float)
                write_hdf5(hdf5_path, frames, speed)
            print("Completed Video Processing")
            return count

        if num==4:
             for batch in range(0,Num_batches):
                #print(generate_indices(batch))
                indexes=generate_indices(batch)
                frames=[]
                speed=[]
                for index in indexes:
                    #print([indexes[i],indexes[i+1]])

                    if(count%1000==0) and count >0:
                        print(count)

                    frame2=cv.imread(IMGList[index])
                    frame2=cv.cvtColor(frame2,cv.COLOR_BGR2RGB)

                    frames.append(np.array(frame2))
                    speed.append(speeds[index])
                    count+=1

                frames=np.array(frames,dtype=np.uint8)
                speed=np.array(speed)
                speed=speed.astype(float)
                write_hdf5(hdf5_path, frames, speed)
            print("Completed Video Processing")
            return count

def reset_hdf5_opflow(num):
    hdf5_path=r".\data\train\"
    hdf5_path=os.path.join(hdf5_path,"op_train",str(num))
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("op_flow", shape = (1, resize_size[0], resize_size[1], resize_size[2]),
                        maxshape = (None, resize_size[0], resize_size[1], resize_size[2]),
                        chunks = (1, resize_size[0], resize_size[1], resize_size[2]))
    f.close()

def write_hdf5_opflow(hdf5_path, op_flows):
    with h5py.File(hdf5_path) as f:
        print( len(op_flows))
        print( f["op_flow"])
        f["op_flow"].resize(f["op_flow"].len() + len(op_flows), axis = 0)
        f["op_flow"][-len(op_flows):] = op_flows

def write_opflow_and_hdf5(num):
    reset_hdf5_opflow(num)
    hdf5_path=r".\data\train\"
    hdf5_path=os.path.join(hdf5_path,"op_train",str(num))
    if num==1:
        op_flows = []
        count = 0
        vidcap = cv.VideoCapture(vidPath)
        success,frame1 = vidcap.read()
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame1 = cv.resize(frame1, (0,0), fx=0.5, fy=0.5)
        while success:
            if (count % 100 == 0) and count > 0:
                print(count)
            success,frame2 = vidcap.read()
            if success == True:
                frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
                frame2 = cv.resize(frame2, (0,0), fx=0.5, fy=0.5)

                frame1,frame2=augment(frame1,frame2)
                frame1,frame2=flipimage(frame1,frame2)
                flow = opticalFlowDense(frame1, frame2)

                op_flows.append(flow)

                frame1 = frame2
                count+=1
            else:
                print("video reading completed")
                continue

        write_hdf5_opflow(hdf5_path1,np.array(op_flows))
        return count
    if num==2:
        op_flows = []
        count = 0
        vidcap = cv.VideoCapture(vidPath)
        success,frame1 = vidcap.read()
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame1 = cv.resize(frame1, (0,0), fx=0.5, fy=0.5)
        while success:
            if (count % 100 == 0) and count > 0:
                print(count)
            success,frame2 = vidcap.read()
            if success == True:
                frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
                frame2 = cv.resize(frame2, (0,0), fx=0.5, fy=0.5)

                frame1,frame2=flipimage(frame1,frame2)
                flow = opticalFlowDense(frame1, frame2)

                op_flows.append(flow)

                frame1 = frame2
                count+=1
            else:
                print("video reading completed")
                continue

        write_hdf5_opflow(hdf5_path1,np.array(op_flows))
        return count

    if num==3:
        op_flows = []
        count = 0
        vidcap = cv.VideoCapture(vidPath)
        success,frame1 = vidcap.read()
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame1 = cv.resize(frame1, (0,0), fx=0.5, fy=0.5)
        while success:
            if (count % 100 == 0) and count > 0:
                print(count)
            success,frame2 = vidcap.read()
            if success == True:
                frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
                frame2 = cv.resize(frame2, (0,0), fx=0.5, fy=0.5)

                frame1,frame2=augment(frame1,frame2)
                flow = opticalFlowDense(frame1, frame2)

                op_flows.append(flow)

                frame1 = frame2
                count+=1
            else:
                print("video reading completed")
                continue

        write_hdf5_opflow(hdf5_path1,np.array(op_flows))
        return count

    if num==4:
        op_flows = []
        count = 0
        vidcap = cv.VideoCapture(vidPath)
        success,frame1 = vidcap.read()
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame1 = cv.resize(frame1, (0,0), fx=0.5, fy=0.5)
        while success:
            if (count % 100 == 0) and count > 0:
                print(count)
            success,frame2 = vidcap.read()
            if success == True:
                frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
                frame2 = cv.resize(frame2, (0,0), fx=0.5, fy=0.5)

                flow = opticalFlowDense(frame1, frame2)

                op_flows.append(flow)

                frame1 = frame2
                count+=1
            else:
                print("video reading completed")
                continue

        write_hdf5_opflow(hdf5_path1,np.array(op_flows))
        return count

# last fuction is to flip all images in sequence and augment
def line_augment(imageSequence):
    augmented=[]
    for image in imageSequence:
        image1=augment(image)
        augmented.append(image1)
    return augmented

def line_flip(imageSequence):
    augmented=[]
    for image in imageSequence:
        image1=flipimage(image)
        augmented.append(image1)
    return augmented
