import os
import cv2
import sys
import glob

os.chdir(sys.path[0])
path = os.getcwd()

print(path)
def generateFrames():
    #getting videos
    cap= cv2.VideoCapture(r".\data\train.mp4")

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(r".\data\IMG\ "+'nihar'+str(i)+'.jpg',frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()

def getFilePaths():
    # for data file paths
    f=glob.glob(r'.\data\IMG\*.jpg')
    return f
