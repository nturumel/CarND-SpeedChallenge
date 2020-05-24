# Speedchallenge

These are the set of instructions to train a model to predict speed from a dashcam footage, to test the model and to predict speeds of unseen footage.

## Train
To train Speedchallenge:

`python Speedchallenge.py video.mp4 speed.txt -- model output.h5 --mode=train --split=0.3`

There are two sets of commands to split the training validation sets.

split=0.2 lets you provide a percentage split(default value of 0.2)

This can be overridden by split_start and split_end which specify a specific range of data for validation.

If you'd like to continue training using the pretrained network then add the `--resume` flag to that line.

If any modifications are made to the optical flow part of the model then `--wipe` must be used to reprocess the data

## Test

To test SpeedNet use:

`python Speedchallenge.py video.mp4 speedGroundTruth.txt -- model output.h5 --mode=predict`

This will print the mean squared error.

## Predict

To test SpeedNet use:

`python finalSpeedchallenge.py video.mp4 speedPredictionOut.txt -- model output.h5 --mode=predict`

This will print the predicted speeds in the output speed file .
