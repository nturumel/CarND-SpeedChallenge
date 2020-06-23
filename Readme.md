# Speedchallenge

A huge thanks to Daniel Nugent and Ryan Chesler for sharing their solutions.

For someone like me starting out in AI and ML and not so proficient in python it was a huge help to walk through their solutions and grab the key ingredients for optical flow calculations, setting up models for training, and compressing all the relevant instructions into a single python file. 

## Data Analysis:
One of the biggest challenges of the project was the limited availability of data for training.  There was a single mp4 video with accompanying speed data.  The video consists of 20400 frames and the speed vs frame graph is as follows:

<img src=".\pics\SpeedData.png" align="center" alt="SpeedData" style="zoom:%;" /> 

The average speed is 12.18 and the speed varies from 30 to 0. Also we can see that from frame 7700 to 12100 there is a precipitous drop of velocity. This will prove to be useful later as it will be used as our validation data once the model architecture is finalized.

<img src=".\pics\ValidData.png" align="center" alt="ValidData" style="zoom:100%; " />



## Histogram:

#### 	Speed Histogram:



<img src=".\pics\SpeedDataHistogram.png" align="center" alt="ValidData" style="zoom:100%; " />

#### 	Validation Histogram:

#### 	 

<img src=".\pics\ValidDataHistogram.png" align="center" alt="SpeedData" style="zoom:100%; " />

So as you can see we have covered a lot of ground in our validation set and our validation set is well selected.

## Approaches:

Based on my research there seem to be two main approaches to solving the problem.  They are the following:

1. #### Optical flow Approach:

   This approach is based on taking successive frames through an optical flow process. Open cv provides us with inbuilt methods to calculate the dense optical flow and sparse optical flow based on successive frames. Once we have this we can map the optical flow images to the speed data using a simple CNN or a time series CNN.

2. #### Time Series Analysis:

   This approach is to enable the model to make connections across time that is not evident purely by looking at the optical flow data. We process individual frames through a CNN and then feed it into an LSTM and map it to a speed.

   

3. #### Combined approach:

   The third approach is based on work by Ryan, where he combined time series analysis of normal frames and then combining it with opflow data and then finally mapping it to the speed. 

   

Based on my analysis of the three methods I found that a time series analysis of pure op flow images produced the best results.

## Data Augmentation:

Due to the limited availability of data, creativity was needed for augmentation.

I used to types of augmentation:

1. #### Brightness Augmentation:

   The frames of the images fed into the opflow calculator were first augmented by randomly changing their brightness.

   

   ```python
   def augment_brightness(self,prev,nxt):
          brightness=np.random.uniform(0.5,1.5)
       imgPrev = Image.fromarray(prev)
          imgNxt = Image.fromarray(nxt)
       frame_augmented_prev = ImageEnhance.Brightness(imgPrev).enhance(brightness)
          frame_augmented_prev = np.array(frame_augmented_prev)
       frame_augmented_nxt = ImageEnhance.Brightness(imgNxt).enhance(brightness)
          frame_augmented_nxt = np.array(frame_augmented_nxt)
       return frame_augmented_prev,frame_augmented_nxt
   
   ```
   

These provide a way to deal with sudden changes in brightness. However the performance of a model trained with augmentation was problematic, so it was dropped.

2. #### Flipping the video:

   The second augmentation was achieved by flipping the entire video from left to right. It stood to reason that a flipped video would and should have the exact same speed values for each frame and a good model should be able to detect that. This double  the amount of data we have.

   About half of the flipped video was split and used purely for validation, enabling us to validate  on a decently sized piece of unseen data.
   
   
   
3. #### **Data Generator**:

   The data generator has an augment flag which when turned on, randomly inserts augmented images in the training pipeline and randomly flips images.

   

   ```python
         def __data_generation(self,indexes):
           # Generates data containing batch samples
           indexes=list(indexes)
           indexes.sort()## We are making the Generator Class to read the video and the speed file
           speeds=[]
           op_flow=[]
           frame=[]
           for index in indexes:
               speeds.append(self.speed_list[self.SpeedIndices[index]])
               r=(np.random.random_integers(1,100))
               if r%2==0 and self.augment:
                   op_flow.append(self.frame_opflow_augmented[self.SpeedIndices[index]])
               else:
                   op_flow.append(self.frame_opflow[self.SpeedIndices[index]])
   
               # second augmentation flip
               r=(np.random.random_integers(1,100))
               if r%5==0 and self.augment:
                   cv.flip( op_flow[-1],1)
   
               temp=[]
               for frame_Index in self.FrameIndices[index]:
                   temp.append(self.frame[frame_Index])
               frame.append(temp)
   
           frame=np.array(frame)
           op_flow=np.array(op_flow)
           speeds=np.array(speeds)
           return [[frame,op_flow],speeds]
   ```
   

   <img src="\pics\optflow.png" alt="optflow" style="zoom:250%;" />*OPTFLOW IMAGE*

   

   <img src="\pics\flow.png" alt="flow" style="zoom:250%;" />*FLOW IMAGE*

   ​														

## Final Model Architecture:

One of the biggest challenges of the project was the limited availability of data for training.  There was also a huge problem of overfitting that I wanted to avoid. Hence, I added dropouts, batch normalizations and also widened the layers. I also added pooling layers.



```python
#-----------------------------------------
        flow_inp=Input(shape=(self.HISTORY,self.DSIZE[0],self.DSIZE[1],3))

        # flow layer -----------------------------
        flow=(ConvLSTM2D(32, 8,8 ,border_mode='same', subsample=(4,4),return_sequences=True,activation="relu", dropout=0.5))(flow_inp)
        flow=(BatchNormalization())(flow)


        flow=(ConvLSTM2D(64, 8,8 ,border_mode='same', subsample=(4,4),return_sequences=False,activation="relu", dropout=0.5))(flow)
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

        op_flow=(Convolution2D(64, 8,8 ,border_mode='same', subsample=(4,4)))(op_flow)
        op_flow=(Activation('relu'))(op_flow)
        op_flow=(BatchNormalization())(op_flow)
        op_flow=(Dropout(0.5))(op_flow)

        op_flow=MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding='valid')(op_flow)

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
```

  

This particular model works well with a LR of 0.0001.

## Training:

After a lot of trial and error, we concluded that the models needed augmentation to avoid overfitting. We trained with augmentation without flip and with flip for 1000 epochs, callbacks ensured that we were not saving the point of overfitting.

`checkpoint = ModelCheckpoint(self.W_FILE, monitor='val_loss', verbose=1,
          save_best_only=True, mode='auto', period=1)`

Finally the model was trained further by the clipped flipped video reserved for training. 



## Model Results and Performance:

In the end the resulting model has a training accuracy of  and a validation accuracy of  .

It performed with an accuracy of -- with the unseen flipped video.

The results can be found in ... 

## Potential Problems:

Sudden brightness changes can cause problems, also we may need to look further back in time.

The biggest problem however is if suddenly all objects in the previous frame disappear.

One way to combat this problem that I considered was to look ahead in time and predict the speed of the median frame based on the frames around it.

However this involves looking ahead in time and is not possible physically speaking.
