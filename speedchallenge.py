
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, TimeDistributed, ConvLSTM2D,LSTM,GlobalMaxPool2D,BatchNormalization,GlobalAvgPool2D,concatenate
from keras import Model
from tensorflow import keras

import numpy as np
import cv2
import sys
import os
import os.path
from os import path
import argparse
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from PIL import Image, ImageEnhance


import DataGenerator


class SpeedNet:

    DSIZE = (100,100)
    W_FILE = "weights.h5"
    EPOCHS = 100
    BATCH_SIZE = 32
    split_start=0
    split_end=0
    HISTORY=2
    def main(self, args):

        self.HISTORY=args.history
        self.EPOCHS=args.epoch
        self.W_FILE=args.model
        self.HISTORY=args.history
        self.split=args.split
        self.split_start=args.split_start
        self.split_end=args.split_end
        self.LR=args.LR
        #compile model
        self.create_model()

        if args.split_start and args.split_end:
          self.split_start=args.split_start
          self.split_end=args.split_end


        self.optflow_dir = [args.video_file.split('.')[0] + "_optflowProcess",
            args.video_file.split('.')[0] + "_optflowProcess_augmented",
            args.video_file.split('.')[0] + "_Process"]

        #train the model
        if args.mode == "train":
            #load existing weights
            if args.resume:
                self.load_weights()
            #start training session
            self.train(args.video_file,args.speed_file,args.wipe,self.EPOCHS,self.BATCH_SIZE,args.augment)

        #test the model
        elif args.mode == "test":
            self.test(args.video_file,args.speed_file)

        elif args.mode == "predict":
            self.predict(args.video_file,args.speed_file)


    def process_frame(self,frame):
        frame = cv2.resize(frame, self.DSIZE, interpolation = cv2.INTER_AREA)
        frame = frame/127.5 - 1.0
        return frame

    def augment_brightness(self,prev,nxt):
       brightness=np.random.uniform(0.5,1.5)
       imgPrev = Image.fromarray(prev)
       imgNxt = Image.fromarray(nxt)
       frame_augmented_prev = ImageEnhance.Brightness(imgPrev).enhance(brightness)
       frame_augmented_prev = np.array(frame_augmented_prev)
       frame_augmented_nxt = ImageEnhance.Brightness(imgNxt).enhance(brightness)
       frame_augmented_nxt = np.array(frame_augmented_nxt)
       return frame_augmented_prev,frame_augmented_nxt


    def optflow(self,frame1,frame2):
        frame1 = frame1[200:400]
        frame1 = cv2.resize(frame1, (0,0), fx = 0.4, fy=0.5)
        frame2 = frame2[200:400]
        frame2 = cv2.resize(frame2, (0,0), fx = 0.4, fy=0.5)
        flow = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow_data = cv2.calcOpticalFlowFarneback(prev, nxt,None, 0.4, 1, 12, 2, 8, 1.2, 0)
        #convert data to hsv
        mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
        flow[...,1] = 255
        flow[...,0] = ang*180/np.pi/2
        flow[...,2] = (mag *15).astype(int)
        return flow

    def prep_data(self,video_file,speed_file,wipe = False,augment=False):
        print ("Prepping data")
        #decode speed data
        speed_data=[]
        if path.exists(speed_file):
            print ("Decoding speed data")
            f = open(speed_file,'r')
            data = f.readlines()
            speed_data = np.array(data[:-1], dtype = 'float32')
            print ("loaded " + str(len(speed_data)) + " speed entries")

        #clear preprocessed data
        if wipe:
            print ("wiping preprocessed data...")
            if(os.path.isdir(self.optflow_dir[0])):
                shutil.rmtree(self.optflow_dir[0])
            if(os.path.isdir(self.optflow_dir[1])):
                shutil.rmtree(self.optflow_dir[1])
            if(os.path.isdir(self.optflow_dir[2])):
                shutil.rmtree(self.optflow_dir[2])

        #process video data if it doesn't exist
        processed_video_opflow = None
        processed_video_opflow_augment= None
        if not os.path.isdir(self.optflow_dir[0]):
            print ("preprocessing data...")
            os.mkdir(self.optflow_dir[0])
            os.mkdir(self.optflow_dir[2])
            if augment:
                os.mkdir(self.optflow_dir[1])
            #Decode video frames
            vid = cv2.VideoCapture(video_file)
            frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

            # Add from second frame onwards
            processed_video = np.empty((frame_cnt-1,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8') # normal frames
            processed_video_opflow = np.empty((frame_cnt-1,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8') # normal opflow frames
            processed_video_opflow_augment=np.empty((frame_cnt-1,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8') #augmented opflow frames

            ret, prev = vid.read()
            i = 0
            #insert
            while True:
                ret, nxt  = vid.read()
                if not ret: #EOF
                    break
                #crop and resize frame
                #___________________
                nxt_resize=nxt[200:400]
                nxt_resize=cv2.resize(nxt_resize, (0,0), fx = 0.4, fy=0.5)
                nxt_resize=cv2.resize(nxt_resize, self.DSIZE, interpolation = cv2.INTER_AREA)
                nxt_resize=nxt_resize/127.5 - 1.0
                processed_video[i]=nxt_resize
                cv2.imwrite(self.optflow_dir[2] + '/' + str(i) + ".png", nxt_resize)
                #___________________

                #___________________
                flow = self.optflow(prev,nxt)
                flow = cv2.resize(flow, self.DSIZE, interpolation = cv2.INTER_AREA)
                processed_video_opflow[i] = flow/127.5 - 1.0
                cv2.imwrite(self.optflow_dir[0] + '/' + str(i) + ".png", flow)
                #__________________

                #__________________
                if augment:
                    prev_augment,nxt_augment=self.augment_brightness(prev,nxt)
                    flow_augment=self.optflow(prev_augment,nxt_augment)
                    flow = cv2.resize(flow_augment, self.DSIZE, interpolation = cv2.INTER_AREA)
                    processed_video_opflow_augment[i] = flow/127.5 - 1.0
                    cv2.imwrite(self.optflow_dir[1] + '/' + str(i) + ".png", flow)
                #_________________

                prev = nxt
                sys.stdout.write("\rProcessed " + str(i) + " frames" )
                i+=1
            print ("\ndone processing " + str(frame_cnt) + " frames")
        #preprocessed data exists
        else:
            print ("Found preprocessed data")
            frame_cnt = len(os.listdir(self.optflow_dir[0]))
            processed_video = np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='float32')
            processed_video_opflow = np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='float32')
            processed_video_opflow_augment=np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='float32')
            for i in range(0,frame_cnt):
                frame=cv2.imread(self.optflow_dir[2] + '/' + str(i) + ".png")
                processed_video[i]=frame/127.5 - 1.0

                flow = cv2.imread(self.optflow_dir[0] + '/' + str(i) + ".png")
                processed_video_opflow[i] = flow/127.5 - 1.0

                if augment:
                    flow = cv2.imread(self.optflow_dir[1] + '/' + str(i) + ".png")
                    processed_video_opflow_augment[i] = flow/127.5 - 1.0

                sys.stdout.write("\rLoading frame " + str(i))
            print ("\ndone loading " + str(frame_cnt) + " frames")
        print ("Done prepping data")
        return (processed_video_opflow,processed_video_opflow_augment,processed_video,speed_data)


    def create_model(self):

        print ("Compiling Model")


        #-----------------------------------------
        flow_inp=Input(shape=(self.HISTORY,self.DSIZE[0],self.DSIZE[1],3))

        # flow layer -----------------------------
        flow=(ConvLSTM2D(32, 8,8 ,border_mode='same', subsample=(4,4),return_sequences=True,activation="relu", dropout=0.5))(flow_inp)
        flow=(BatchNormalization())(flow)


        flow=(ConvLSTM2D(128, 8,8 ,border_mode='same', subsample=(4,4),return_sequences=False,activation="relu", dropout=0.5))(flow)
        flow=(BatchNormalization())(flow)

        flow=Flatten()(flow)
        flow_out=Dense(256)(flow)
        #----------------------------------------
        conc=Activation('relu')(flow_out)
        #---------------------------------------------

        #----------------------------------------
        op_flow_inp=Input(shape=(self.DSIZE[0],self.DSIZE[1],2))

        op_flow=(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))(op_flow_inp)
        op_flow=(Activation('relu'))(op_flow)
        op_flow=(BatchNormalization())(op_flow)
        op_flow=(Dropout(0.5))(op_flow)

        op_flow=(Convolution2D(128, 8,8 ,border_mode='same', subsample=(4,4)))(op_flow)
        op_flow=(Activation('relu'))(op_flow)
        op_flow=(BatchNormalization())(op_flow)
        op_flow=(Dropout(0.5))(op_flow)

        op_flow=(Dense(256))(op_flow)
        op_flow_out=(Flatten())(op_flow)
        #----------------------------------------


        conc=concatenate([conc,op_flow_out])

        conc=Activation('relu')(conc)
        conc=Dropout(0.5)(conc)

        conc=Dense(128)(conc)
        conc=Activation('relu')(conc)
        conc=Dropout(0.5)(conc)

        result=Dense(1)(conc)

        model = Model(inputs=[flow_inp,op_flow_inp], outputs=[result])

        opt = keras.optimizers.Adam(learning_rate=self.LR)
        model.compile(optimizer=opt, loss='mse')
        print(model.summary())
        self.model= model


    def load_weights(self):
        try:
            print ("loading weights")
            self.model.load_weights(self.W_FILE)
            return True
        except ValueError:
            print ("Unable to load weights. Model has changed")
            print ("Please retrain model")
            return False
        except IOError:
            print ("Unable to load weights. No previous weights found")
            print ("Please train model")
            return False

    def train(self,X_src,Y_src, wipe,n_epochs = 50, batch_size= 32,augment=False):
        #load data
        X,X_augment,X_frames,Y = self.prep_data(X_src,Y_src,wipe = wipe,augment=augment)

        X = X[:,:,:,[0,2]] #extract channels with data
        X_augment=X_augment[:,:,:,[0,2]]

        FrameIndices,SpeedIndices=DataGenerator.generate_indices(self.HISTORY,len(Y))

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

        print (" This is the range of train: ",train_indexes)
        print (" This is the range of val: ",val_indexes)


        print ("Starting training")
        checkpoint = ModelCheckpoint(self.W_FILE, monitor='loss', verbose=1,
          save_best_only=True, mode='auto', period=1)

        train_generator=DataGenerator.DataGenerator(batch_size,X_frames,X,
            X_augment,Y,FrameIndices,SpeedIndices,train_size,
            indexes=train_indexes,validation_mode=False,augment=augment)

        valid_generator=DataGenerator.DataGenerator(batch_size,X_frames,X,
            X_augment,Y,FrameIndices,SpeedIndices,train_size,
            indexes=val_indexes,validation_mode=True,augment=False)

        self.model.fit_generator(train_generator,
                    validation_data=valid_generator,
                    epochs=n_epochs,
                    callbacks=[checkpoint])

        #save weights
        print ("Done training. Saved weights")

    def test(self,X_src, Y_src):
        #load data
        X_test,X_test_augment,X_test_frames,Y_test = self.prep_data(X_src,Y_src,augment=False,wipe=True)
        X_test = X_test[:,:,:,[0,2]] #extract channels with data


        #load weights
        ret = self.load_weights()
        if ret:
            #test the model on unseen data
            print ("Starting testing")
            predictions=[]
            FrameIndices,SpeedIndices=DataGenerator.generate_indices(self.HISTORY,len(X_test))
            train_size=len(SpeedIndices)
            print("Number of frames to be evaluated: ",train_size)
            for sequenceFrame,sequenceSpeed in zip(FrameIndices,SpeedIndices):
                opflow=X_test[sequenceSpeed]
                framesequence=[]
                for index in sequenceFrame:
                    frame=X_test_frames[index]
                    framesequence.append(frame)

                framesequence=np.array(framesequence)
                opflow=np.array(opflow)
                framesequence = framesequence[None,:,:,:,:]
                opflow=opflow[None,...]
                speed_predict=self.model.predict([framesequence,opflow])
                speed_predict=float(speed_predict[0])
                predictions.append(speed_predict)

            #adjust for time history:
            Y_test=Y_test[self.HISTORY-1:]
            mean_square=mean_squared_error(predictions,Y_test)
            print("The mean square error is: ",mean_square)
            print ("Done testing")
        else:
            print ("Test failed to complete with improper weights")

    def predict(self,X_src, Y_out):
        X,X_augment,X_frames,Y = self.prep_data(X_src,Y_out,augment=False,wipe=True)
        X=X[:,:,:,[0,2]]

        ret=self.load_weights()
        if ret:
            predictions=[]
            FrameIndices,SpeedIndices=DataGenerator.generate_indices(self.HISTORY,len(X))
            train_size=len(SpeedIndices)
            print("Number of frames to be predicted: ",train_size)
            for sequenceFrame,sequenceSpeed in zip(FrameIndices,SpeedIndices):
                opflow=X[sequenceSpeed]
                framesequence=[]
                for index in sequenceFrame:
                    frame=X_frames[index]
                    framesequence.append(frame)

                framesequence=np.array(framesequence)
                opflow=np.array(opflow)
                opflow=opflow[None,...]
                framesequence = framesequence[None,:,:,:,:]
                speed_predict=self.model.predict([framesequence,opflow])
                speed_predict=float(speed_predict[0])
                prediction_str=str(speed_predict)
                frame_num=str( SpeedIndices[sequenceFrame[0]]+1)
                sys.stdout.write("\rFor Frame: " +frame_num+ " Predicted speed: "+ prediction_str )
                predictions.append(speed_predict)

            with open(Y_out, 'w') as filehandle:
                for listitem in predictions:
                    filehandle.write('%f\n' % listitem)
        else:
            print("Prediction failed to complete with improper weights")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file",
                        help="video file name")
    parser.add_argument("speed_file",
                        help="speed data file name")
    parser.add_argument("--model", type=str, default='output.h5',
                        help="output model name")

    parser.add_argument("--split", type=float, default=0.2,
                        help="percentage of train data for validation")
    parser.add_argument("--LR", type=float, default=0.01,
                        help="percentage of train data for validation")
    parser.add_argument("--split_start", type=int, default=0,
                        help="frame of train data to start validation")
    parser.add_argument("--split_end", type=int, default=0,
                        help="frame of train data to end validation")

    parser.add_argument("--history", type=int, default=1,
                        help="number of frames to look back into")
    parser.add_argument("--epoch", type=int, default=50,
                        help="number of epochs to train")
    parser.add_argument("--augment", action='store_true',
                        help="clears existing preprocessed data")

    parser.add_argument("--mode", choices=["train", "test", "predict"], default='train',
                        help="Train, Test, or predict model")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    args = parser.parse_args()
    print ("ML for Speed")
    net = SpeedNet()
    net.main(args)
