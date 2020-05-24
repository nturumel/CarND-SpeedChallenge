from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import cv2
import sys
import os
import argparse
import json
import shutil
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from PIL import Image, ImageEnhance


class SpeedNet:

    DSIZE = (100,100)
    W_FILE = "weights.h5"
    EPOCHS = 100
    BATCH_SIZE = 32
    split_start=0
    split_end=0

    def main(self, args):
        #compile model
        self.create_model()

        self.EPOCHS=args.epoch
        self.W_FILE=args.model
        self.HISTORY=args.history

        if args.split_start and args.split_end:
          self.split_start=args.split_start
          self.split_end=args.split_end


        self.optflow_dir = [args.video_file.split('.')[0] + "_optflowProcess",args.video_file.split('.')[0] + "_optflowProcess_augmented"]

        #train the model
        if args.mode == "train":
            #load existing weights
            if args.resume:
                self.load_weights()
            #start training session
            self.train(args.video_file,args.speed_file,args.split,args.wipe,self.EPOCHS,self.BATCH_SIZE,args.augment)

        #test the model
        elif args.mode == "test":
            self.test(args.video_file,args.speed_file)

        elif args.mode == "play":
            self.play(args.video_file,args.speed_file)


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
       #image_current_1[:,:,2] = image_current_1[:,:,2]*brightness
       #frame_augmented = cv.cvtColor(image_current_1,cv.COLOR_HSV2RGB)
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

    def prep_data(self,video_file,speed_file, shuffle = False,wipe = False,augment=False):
        print ("Prepping data")
        #decode speed data
        print ("Decoding speed data")
        f = open(speed_file,'r')
        data = f.readlines()
        speed_data = np.array(data[0:-1], dtype = 'float32')
        print ("loaded " + str(len(speed_data)) + " speed entries")

        #clear preprocessed data
        if wipe and (os.path.isdir(self.optflow_dir[0]) and (augment or os.path.isdir(self.optflow_dir[1]))):
            print ("wiping preprocessed data...")
            if(os.path.isdir(self.optflow_dir[0])):
                shutil.rmtree(self.optflow_dir[0])
            if(os.path.isdir(self.optflow_dir[1])):
                shutil.rmtree(self.optflow_dir[1])


        #process video data if it doesn't exist
        processed_video = None
        processed_video_augment= None
        if not os.path.isdir(self.optflow_dir[0]):
            print ("preprocessing data...")
            os.mkdir(self.optflow_dir[0])
            if augment:
                os.mkdir(self.optflow_dir[1])
            #Decode video frames
            vid = cv2.VideoCapture(video_file)
            frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_video = np.empty((frame_cnt-1,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8')
            processed_video_augment=np.empty((frame_cnt-1,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8')
            ret, prev = vid.read()
            i = 0
            while True:
                ret, nxt  = vid.read()
                if not ret: #EOF
                    break
                #crop and resize frame
                flow = self.optflow(prev,nxt)
                flow = cv2.resize(flow, self.DSIZE, interpolation = cv2.INTER_AREA)
                processed_video[i] = flow/127.5 - 1.0
                cv2.imwrite(self.optflow_dir[0] + '/' + str(i) + ".png", flow)

                if augment:
                    prev_augment,nxt_augment=self.augment_brightness(prev,nxt)
                    flow_augment=self.optflow(prev_augment,nxt_augment)
                    flow = cv2.resize(flow_augment, self.DSIZE, interpolation = cv2.INTER_AREA)
                    processed_video_augment[i] = flow/127.5 - 1.0
                    cv2.imwrite(self.optflow_dir[1] + '/' + str(i) + ".png", flow)

                prev = nxt
                sys.stdout.write("\rProcessed " + str(i) + " frames" )
                i+=1
            print ("\ndone processing " + str(frame_cnt) + "frames")
        #preprocessed data exists
        else:
            print ("Found preprocessed data")
            frame_cnt = len(os.listdir(self.optflow_dir[0]))
            processed_video = np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='float32')
            processed_video_augment=np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='float32')
            for i in range(0,frame_cnt):
                flow = cv2.imread(self.optflow_dir[0] + '/' + str(i) + ".png")
                processed_video[i] = flow/127.5 - 1.0

                if augment:
                    flow = cv2.imread(self.optflow_dir[1] + '/' + str(i) + ".png")
                    processed_video_augment[i] = flow/127.5 - 1.0

                sys.stdout.write("\rLoading frame " + str(i))
            print ("\ndone loading " + str(frame_cnt) + " frames")

        
        print ("Done prepping data")
        return (processed_video,processed_video_augment, speed_data)


    def create_model(self):

        print ("Compiling Model")

        self.model = Sequential()
        self.model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4),input_shape=(self.DSIZE[0],self.DSIZE[1],2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')


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

    def train(self,X_src,Y_src,val_split, wipe,n_epochs = 50, batch_size= 32,augment=False):
        #load data
        X,X_augment,Y = self.prep_data(X_src,Y_src,shuffle = True,wipe = wipe,augment=augment)
        X = X[:,:,:,[0,2]] #extract channels with data
        X_augment=X_augment[:,:,:,[0,2]]

        if augment:
            Xtemp=[]
            for i in range(len(X)):
                random=np.random.random_integers(1,15)
                if random%3==0:
                    Xtemp.append(X_augment[i])
                else:
                    Xtemp.append(X[i])
            X=Xtemp
            X=np.array(X,dtype='uint8')

        #train model
        print ("Starting training")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=val_split, random_state=42)
        checkpoint = ModelCheckpoint(self.W_FILE, monitor='loss', verbose=1,
          save_best_only=True, mode='auto', period=1)

        self.model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint])
        #save weights
        print ("Done training. Saved weights")

    def test(self,X_src, Y_src):
        #load data
        X_test,X_test_augment,Y_test = self.prep_data(X_src,Y_src,shuffle = False,augment=False)
        X_test = X_test[:,:,:,[0,2]] #extract channels with data

        #load weights
        ret = self.load_weights()
        if ret:
            #test the model on unseen data
            print ("Starting testing")
            print (self.model.evaluate(X_test,Y_test))
            print ("Done testing")
        else:
            print ("Test failed to complete with improper weights")

    def play(self, X_src, Y_src):
        print ("Starting testing")
        #load data
        X,X_augment,Y = self.prep_data(X_src,Y_src,shuffle = False,augment=False)
        rec = cv2.VideoWriter('flow.avi',int(cv2.cv.CV_FOURCC('M','J','P','G')),48,(300,300))
        #load weights
        ret = self.load_weights()
        if ret:
            #test the model on unseen data
            for x,y in zip(X,Y):
                flow = ((x + 1) * 127.5).astype('uint8')
                flow = cv2.cvtColor(flow,cv2.COLOR_HSV2BGR)
                flow = cv2.resize(flow,(300,300))
                pred_y = self.model.predict(np.array([x[:,:,[0,2]]]))[0,0]
                error = abs(y-pred_y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(flow,"Predicted Speed: " + str(pred_y),(5,15),font, 0.55,(255,255,255),2)
                cv2.putText(flow,"Actual Speed: " + str(y),(5,45),font, 0.55,(255,255,255),2)
                cv2.putText(flow,"Error: " + str(error),(5,75),font, 0.55,(255,255,255),2)
                rec.write(flow)
            rec.release()
            print ("Done predicting")
        else:
            print ("Prediction failed to complete with improper weights")


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
    parser.add_argument("--split_start", type=int, default=0,
                        help="frame of train data to start validation")
    parser.add_argument("--split_end", type=int, default=0,
                        help="frame of train data to end validation")

    parser.add_argument("--history", type=int, default=2,
                        help="number of frames to look back into")
    parser.add_argument("--epoch", type=int, default=50,
                        help="number of epochs to train")
    parser.add_argument("--augment", action='store_true',
                        help="clears existing preprocessed data")

    parser.add_argument("--mode", choices=["train", "test", "play"], default='train',
                        help="Train, Test, or Play model")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    args = parser.parse_args()
    print ("ML for Speed")
    net = SpeedNet()
    net.main(args)
