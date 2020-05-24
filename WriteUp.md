# Speedchallenge

These are the set of instructions to train a model to predict speed from a dashcam footage, to test the model and to predict speeds of unseen footage.

## Train
finalSpeedchallenge is pretrained but if you'd like to retrain it use:

`python Speedchallenge.py video.mp4 speed.txt --mode=train --split=0.3`

If you'd like to continue training using the pretrained network then add the `--resume` flag to that line.

If any modifications are made to the optical flow part of the model then `--wipe` must be used to reprocess the data

## Test

To test SpeedNet use:

`python Speedchallenge.py video.mp4 speedGroundTruth.txt --mode=predict`

This will print the mean squared error.

## Predict

To test SpeedNet use:

`python finalSpeedchallenge.py video.mp4 speedPredictionOut.txt --mode=predict`

This will print the predicted speeds in the output speed file .
