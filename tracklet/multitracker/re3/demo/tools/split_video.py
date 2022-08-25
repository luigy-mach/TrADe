import cv2
import glob
import numpy as np
import sys
import os.path



#import cv2

import glob
import numpy as np
import sys
import os.path
import random

from PIL import Image

#from darkflow.net.build import TFNet
import cv2





#cv2.namedWindow('Image', cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 640, 480)


cap = cv2.VideoCapture('josemiguel_2.avi')
name_crop='/home/luigy/isabo/re3/demo/tools/josemiguel_2/'

if (cap.isOpened()==False): 
    print("Error opening video stream or file")
ret, frame = cap.read()


flag=0
initial_bbox=[]
people=[]

count_frame=0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        nfile = '{0:06}'.format(count_frame)
        cv2.imwrite(name_crop+nfile+'.png',frame)
        count_frame+=1   
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Image', frame)
        cv2.waitKey(1)
    else:
        break