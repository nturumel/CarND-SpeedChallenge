import numpy as np
import cv2 as cv
import sys
import os
import argparse
import json
import shutil
import csv
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import keras
from keras.layers import Lambda,Conv2D,Convolution2D, MaxPool2D,MaxPooling2D, CuDNNGRU, GlobalMaxPool2D, Reshape, GRU,LSTM, \
concatenate, Input, TimeDistributed ,ELU, Dense,Activation, BatchNormalization, SpatialDropout2D, SpatialDropout1D, Dropout, GlobalAvgPool2D, Flatten
from keras import Model
from keras.applications import Xception
import keras.backend as k
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras.layers import InputLayer
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import Reshape
from keras.layers import Lambda
from tensorflow.keras.optimizers import Nadam
import pickle


import DataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class finalSpeedchallenge:
    def augment_brightness(self,frame):
        brightness=np.random.uniform(0.5,1.5)
        image_current_1 = cv.cvtColor(frame,cv.COLOR_RGB2HSV)
        image_current_1[:,:,2] = image_current_1[:,:,2]*brightness
        frame_augmented = cv.cvtColor(image_current_1,cv.COLOR_HSV2RGB)
        return frame_augmented

    def process_frame(self,frame):
        frame=frame[200:400]
        frame=cv.resize(frame, (0,0), fx=self.DSIZE[0], fy=self.DSIZE[1])
        frame_augmented=self.augment_brightness(frame)
        return frame,frame_augmented

    def optflowProcess(self,frame1,frame2):
        
        '''
        image_current = np.array(frame1)
        image_next = np.array(frame2)
        gray_current = cv.cvtColor(image_current, cv.COLOR_RGB2GRAY)
        gray_next = cv.cvtColor(image_next, cv.COLOR_RGB2GRAY)
        flow = cv.calcOpticalFlowFarneback(gray_current, gray_next, None, 0.5, 1, 15, 2, 5, 1.3, 0)
        
        '''
        
        flow = np.zeros_like(frame1)
        prev = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

        nxt = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow_data = cv.calcOpticalFlowFarneback(prev, nxt, None,0.4, 1, 12, 2, 8, 1.2, 0)
        #convert data to hsv
        mag, ang = cv.cartToPolar(flow_data[...,0], flow_data[...,1])
        flow[...,1] = 255
        flow[...,0] = ang*180/np.pi/2
        flow[...,2] = (mag *15).astype(int)
        flow/127.5 - 1.0
        flow = cv.cvtColor(flow,cv.COLOR_HSV2BGR)
        return flow

    def prep_data_predict(self,video_file):
        print("Prepping Data")
        #clear processed Data
        if os.path.isdir(self.optflowProcess_dir[0]):
            print("wiping existing data...")
            shutil.rmtree(self.optflowProcess_dir[0])

        if not os.path.isdir(self.optflowProcess_dir[0]) :
             print("preprocessing data...")
             os.mkdir(self.optflowProcess_dir[0])

             # writing down opflow images in the directory
             # store the path names in an array
             opflow=[]
             opflow_augmented=[]


             vidcap = cv.VideoCapture(video_file)
             success,frame1 = vidcap.read()
             frame1,frame1_augmented=self.process_frame(frame1)
             count = 0
             while success:
                 if (count % 500 == 0) and count > 0:
                     print(count),

                 success,frame2=vidcap.read()
                 if success:
                     frame2,frame2_augmented=self.process_frame(frame2)
                     flow=self.optflowProcess(frame1, frame2)
                     name=self.optflowProcess_dir[0] + '/' + str(count)
                     arr=np.array(flow)
                     cv.imwrite(self.optflowProcess_dir[0] + '/' + str(count) + ".png", flow)
                     opflow.append(name)

                     frame1=frame2
                     count+=1
                 else:
                    print("video reading completed")
                    continue


             self.opflow=opflow
             self.op_flow.sort()
             frame_cnt=count
             print("\ndone converting " + str(frame_cnt) + " frames")

    def prep_data(self,video_file,speed_file,wipe=False):
        print("Prepping Data")
        print("reading speed file")
        data=[]
        with open (speed_file) as csvfile:
            reader = csv.reader(csvfile)
            for speed in reader:
                data.append(speed)
        speed_data = np.array(data[0:-1], dtype = 'float32')
        self.speed_data = speed_data
        print("loaded " + str(len(self.speed_data)) + " speed entries")

        #clear processed Data
        if wipe and os.path.isdir(self.optflowProcess_dir[0]):
            print("wiping preprocessed data...")
            shutil.rmtree(self.optflowProcess_dir[0])
            shutil.rmtree(self.optflowProcess_dir[1])

        frame_cnt=0
        if not os.path.isdir(self.optflowProcess_dir[0]) and not os.path.isdir(self.optflowProcess_dir[1]):
             print("preprocessing data...")
             os.mkdir(self.optflowProcess_dir[0])
             os.mkdir(self.optflowProcess_dir[1])
             # writing down opflow images in the directory
             # store the path names in an array
             opflow=[]
             opflow_augmented=[]


             vidcap = cv.VideoCapture(video_file)
             success,frame1 = vidcap.read()
             frame1,frame1_augmented=self.process_frame(frame1)
             count = 0
             while success:
                 if (count % 500 == 0) and count > 0:
                     print(count)

                 success,frame2=vidcap.read()
                 if success:
                     frame2,frame2_augmented=self.process_frame(frame2)
                     flow=self.optflowProcess(frame1, frame2)
                     name=self.optflowProcess_dir[0] + '/' + str(count)
                     arr=np.array(flow)
                     cv.imwrite(self.optflowProcess_dir[0] + '/' + str(count) + ".png", flow)
                     opflow.append(name)

                     flow_augmented=self.optflowProcess(frame1_augmented, frame2_augmented)
                     name=self.optflowProcess_dir[1] + '/' + str(count)
                     arr=np.array(flow_augmented)
                     cv.imwrite(self.optflowProcess_dir[1] + '/' + str(count) + ".png", flow_augmented)
                     opflow_augmented.append(name)

                     frame1=frame2
                     frame1_augmented=frame2_augmented
                     count+=1
                 else:
                    print("video reading completed")
                    continue


             self.opflow=opflow
             self.opflow_augmented=opflow_augmented
             self.opflow.sort()
             self.opflow_augmented.sort()
             frame_cnt=count

        else:
            print("Found preprocessed data")
            frame_cnt = len(os.listdir(self.optflowProcess_dir[0]))

            for r, d, f in os.walk(self.optflowProcess_dir[0]):
                for file in f:
                    self.opflow.append(os.path.join(r, file))

            for r, d, f in os.walk(self.optflowProcess_dir[0]):
                for file in f:
                    self.opflow_augmented.append(os.path.join(r, file))
            self.opflow.sort()
            self.opflow_augmented.sort()

        print("\ndone loading " + str(frame_cnt) + " frames")





    def __init__(self):
        self.DSIZE=(0.5,0.5)
        self.W_FILE="finalSolution_opFlow_2frames.h5"
        self.EPOCHS=2
        self.BATCH_SIZE=30
        self.HISTORY=2
        self.opflow=[]
        self.opflow_augmented=[]
        self.split=0.2
        self.split_start=0
        self.split_end=0

    def main(self, args):
        
        self.EPOCHS=args.epoch
        self.W_FILE=args.model
        self.HISTORY=args.history
        if args.split:
            self.split=args.split
        if args.split_start and args.split_end:
          self.split_start=args.split_start
          self.split_end=args.split_end
        
        # make the model
        self.create_model()
        self.optflowProcess_dir=[args.video_file.split('.')[0] + "_optflowProcess",args.video_file.split('.')[0] + "_optflowProcess_augmented"]

        # train the model
        if args.mode == "train":
            #load existing weights
            if args.resume:
                self.load_weights()

            #start training session
            self.train(args.video_file,args.speed_file,args.wipe)


        #predictions from the model
        elif  args.mode == "predict":
            self.predict(args.video_file,args.speed_file)

    def create_model(self):
        print("Compliling Model")

        op_flow_inp=Input(shape=(self.HISTORY,100,320,2))


        filter_size = (3,3)
        op_flow = TimeDistributed(Dropout(.5))(op_flow_inp)
        op_flow = TimeDistributed(Conv2D(4, filter_size, activation = "relu", data_format = "channels_last"))(op_flow)
        op_flow = TimeDistributed(MaxPool2D())(op_flow)
        op_flow = TimeDistributed(BatchNormalization())(op_flow)
        op_flow = TimeDistributed(Dropout(.5))(op_flow)
        
        op_flow = TimeDistributed(Conv2D(8, filter_size, activation = "relu", data_format = "channels_last"))(op_flow)
        op_flow = TimeDistributed(MaxPool2D())(op_flow)
        op_flow = TimeDistributed(BatchNormalization())(op_flow)
        op_flow = TimeDistributed(Dropout(.5))(op_flow)
        
        op_flow = TimeDistributed(Conv2D(32, filter_size, activation = "relu", data_format = "channels_last"))(op_flow)
        op_flow = TimeDistributed(MaxPool2D())(op_flow)
        op_flow = TimeDistributed(BatchNormalization())(op_flow)
        op_flow = TimeDistributed(Dropout(.5))(op_flow)
        
        op_flow = TimeDistributed(Conv2D(64, filter_size, activation = "relu", data_format = "channels_last"))(op_flow)
        op_flow = TimeDistributed(Dropout(.3))(op_flow)
        op_flow = TimeDistributed(MaxPool2D())(op_flow)
        op_flow = TimeDistributed(Conv2D(128, filter_size, activation = "relu", data_format = "channels_last"))(op_flow)
        op_flow = TimeDistributed(MaxPool2D())(op_flow)
        op_flow = TimeDistributed(BatchNormalization())(op_flow)
        op_flow = TimeDistributed(Dropout(.5))(op_flow)
        
        op_flow_max = TimeDistributed(GlobalMaxPool2D())(op_flow)
        op_flow_avg = TimeDistributed(GlobalAvgPool2D())(op_flow)
        conc=concatenate([op_flow_max,op_flow_avg],axis=1)
        conc = SpatialDropout1D(.5)(conc)
        
        conc = GRU(256)(conc)
        conc = Dense(100, activation = "relu")(conc)
        conc = Dropout(.5)(conc)
        conc = Dense(50, activation = "relu")(conc)
        conc = Dropout(.5)(conc)
        result = Dense(1, activation='linear')(conc)
        
        model = Model(inputs=op_flow_inp, outputs=[result])
        print(model.summary())
        adam = Nadam()
        model.compile(optimizer='adam', loss='mse')
        self.model= model

    def load_weights(self):
        try:
            print("loading weights")
            self.model.load_weights(self.W_FILE)
            return True
        except ValueError:
            print("Unable to load weights. Model has changed")
            print("Please retrain model")
            return False
        except IOError:
            print("Unable to load weights. No previous weights found")
            print("Please train model")
            return False


    def train(self,X_src,Y_src,wipe):
        #load data
        self.prep_data(X_src,Y_src,wipe = wipe)
        print("Length of Speed Data: ",len(self.speed_data))
        print("Length of Frame Data: ",len(self.opflow))
        print("Length of Augmented Frame Data: ",len(self.opflow_augmented))

        FrameIndices,SpeedIndices=DataGenerator.generate_indices(self.HISTORY,len(self.speed_data))
        train_size=len(SpeedIndices)
        train_indexes=[]
        val_indexes=[]
        
        if self.split_start and self.split_end:
          All_indices=np.arange(int(train_size))
          train_indexes.extend(All_indices[:self.split_start])
          train_indexes.extend(All_indices[self.split_end:])
          val_indexes=All_indices[self.split_start:self.split_end]
        else:
          train_indexes, val_indexes=train_test_split(np.arange(int(train_size)), shuffle = False, test_size = self.split)

        print(train_size,'Training data size per Aug')
        print(len(train_indexes),'Train indices size')
        print(len(val_indexes),'Val indices size')
        #---------------Remove----------------
        print("First Frame Index: ",FrameIndices[0])
        print("First Speed Index: ",SpeedIndices[0])
        

        print('Checking if within indices:')
        frame=cv.imread(self.opflow[FrameIndices[-1][-1]])
        print(frame.mean())
        print(self.speed_data[SpeedIndices[-1]],'Last Speed Indice Value')
        #---------------Remove----------------

        #Training
        train_generator=DataGenerator.DataGenerator(self.BATCH_SIZE,self.opflow,self.opflow_augmented,self.speed_data,FrameIndices,SpeedIndices,train_size,indexes=train_indexes,validation_mode=False)
        valid_generator=DataGenerator.DataGenerator(self.BATCH_SIZE,self.opflow,self.opflow_augmented,self.speed_data,FrameIndices,SpeedIndices,train_size,indexes=val_indexes,validation_mode=False)
        self.model.fit_generator(train_generator, validation_data=valid_generator, epochs = self.EPOCHS,callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3), ModelCheckpoint(filepath=self.W_FILE, save_weights_only=False)])

    def predict(self,X_src, Y_out):
        #load data
        self.prep_data_predict(X_src)

        #load weights
        ret=self.load_weights()
        if ret:
            predictions=[]
            FrameIndices,SpeedIndices=DataGenerator.generate_indices(self.HISTORY,len(self.opflow))
            train_size=len(SpeedIndices)
            print("Number of frames to be predicted: ",train_size)
            for sequence in FrameIndices:
                framesequence=[]
                for index in sequence:
                    frame=cv.imread(self.opflow[index])
                    framesequence.append(frame)
                framesequence=np.array(framesequence)
                #np.expand_dims(framesequence, axis=0)
                framesequence = framesequence[None,...]
                print(framesequence.shape)
                speed_predict=self.model.predict(framesequence)
                print("For Frame: ",SpeedIndices[sequence[0]]+1," Predicted speed: ",speed_predict)
                predictions.append(speed_predict)

            with open(Y_out, 'w') as filehandle:
                for listitem in predictions:
                    filehandle.write('%s\n' % listitem)
        else:
            print("Prediction failed to complete with improper weights")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # files
    parser.add_argument("video_file",
                        help="video file name")
    parser.add_argument("speed_file",
                        help="speed data file name")
    parser.add_argument("--model", type=str, default='output.h5',
                        help="output model name")
    #flags split
    parser.add_argument("--split", type=float, default=0.2,
                        help="percentage of train data for validation")
    parser.add_argument("--split_start", type=int, default=0,
                        help="frame of train data to start validation")
    parser.add_argument("--split_end", type=int, default=0,
                        help="frame of train data to end validation")
    
    
    #flag history and epochs
    parser.add_argument("--history", type=int, default=2,
                        help="number of frames to look back into")
    parser.add_argument("--epoch", type=int, default=5,
                        help="number of epochs to train")

    # flags for training or predicting
    parser.add_argument("--mode", choices=["train", "predict"], default='train',
                        help="Train or predict")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    args = parser.parse_args()
    print("Running finalSpeedchallenge")
    net = finalSpeedchallenge()
    net.main(args)
