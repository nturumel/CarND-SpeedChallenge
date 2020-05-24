# Speedchallenge

A huge thanks to Daniel Nugent and Ryan Chesler for sharing their solutions.

For someone like me starting out in AI and ML and not so proficient in python it was a huge help to walk through their solutions and grab the key ingredients for optical flow calculations, setting up models for training, and compressing all the relevant instructions into a single python file. 

## Data Analysis:
One of the biggest challenges of the project was the limited availability of data for training.  There was a single mp4 video with accompanying speed data.  The video consists of 20400 frames and the speed vs frame graph is as follows:

<img src="C:\Users\Nihar\Google Drive\speedchallenge\SpeedData.png" alt="SpeedData" style="zoom:75%;" />

The average speed is 12.18 and the speed varies from 30 to 0. Also we can see that from frame 7700 to 12100 there is a precipitous drop of velocity. This fact will prove to be useful later as that will be used as our validation data once the model architecture is finalized.

<img src="C:\Users\Nihar\Google Drive\speedchallenge\ValidData.png" alt="ValidData" style="zoom:75%;" />



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

    `def augment_brightness(self,prev,nxt):`

   ​    `brightness=np.random.uniform(0.5,1.5)`

   ​    `imgPrev = Image.fromarray(prev)`

   ​    `imgNxt = Image.fromarray(nxt)`

   ​    `frame_augmented_prev = ImageEnhance.Brightness(imgPrev).enhance(brightness)`

   ​    `frame_augmented_prev = np.array(frame_augmented_prev)`

   ​    `frame_augmented_nxt = ImageEnhance.Brightness(imgNxt).enhance(brightness)`

   ​    `frame_augmented_nxt = np.array(frame_augmented_nxt)`

   ​    `return frame_augmented_prev,frame_augmented_nxt`

   These provide a way to deal with sudden changes in brightness. However the performance of a model trained with augmentation was problematic, so it was dropped.

2. #### Flipping the video:

   The second augmentation was achieved by flipping the entire video from left to right. It stood to reason that a flipped video would and should have the exact same speed values for each frame and a good model should be able to detect that. This double  the amount of data we have.

   About half of the flipped video was split and used purely for validation, enabling us to validate  on a decently sized piece of unseen data.

## Final Model Architecture:

One of the biggest challenges of the project was the limited availability of data for training.  There 

## Model Results and Performance:

One of the biggest challenges of the project was the limited availability of data for training.  There 

## Potential Problems:

One of the biggest challenges of the project was the limited availability of data for training.  There 