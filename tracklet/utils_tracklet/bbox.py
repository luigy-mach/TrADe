

import glob
import numpy as np
import sys
import os.path
import os
import random

from PIL import Image

import cv2
import time
import json 

import multiprocessing.dummy as mp


from ..library.class_tracklet import *
from ..library.parallel_save import *




def is_into(min, value, max):
    return min<=value and value<=max

def box1_into_box2(boxA, boxB):
    xA0 = int(boxA[0])
    yA0 = int(boxA[1])
    xA1 = int(boxA[2])
    yA1 = int(boxA[3])
    xB0 = int(boxB[0])
    yB0 = int(boxB[1])
    xB1 = int(boxB[2])
    yB1 = int(boxB[3])
    #A into B
    if is_into(xB0,xA0,xB1) and is_into(xB0,xA1,xB1):
        if is_into(yB0,yA0,yB1) and is_into(yB0,yA1,yB1):
            return True
    return False


# bb_IOU (Intersection Over Union) output values between 0 and 1
def bb_IOU(boxA, boxB):
    if box1_into_box2(boxA,boxB) or box1_into_box2(boxB,boxA):
        return 1
    xA = max(int(boxA[0]), int(boxB[0]))
    yA = max(int(boxA[1]), int(boxB[1]))
    xB = min(int(boxA[2]), int(boxB[2]))
    yB = min(int(boxA[3]), int(boxB[3]))
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (int(boxA[2]) - int(boxA[0]) + 1) * (int(boxA[3]) - int(boxA[1]) + 1)
    boxBArea = (int(boxB[2]) - int(boxB[0]) + 1) * (int(boxB[3]) - int(boxB[1]) + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



# find_biggest_id find the biggest ID
def max_id(initBoundingBox):
    tmp = int(-999)
    for u_id in initBoundingBox.keys():
        if int(u_id) > int(tmp):
            tmp = u_id
    return int(tmp)




def setbox2str(bbox):
    return str(int(bbox[1]))+'_'+str(int(bbox[3]))+'_'+str(int(bbox[0]))+'_'+str(int(bbox[2]))
    # return str(int(bbox[1]))+','+str(int(bbox[3]))+','+str(int(bbox[0]))+','+str(int(bbox[2]))



def create_path_crop(dir_gallery, key, save_separate=False):
    base_path =''
    if save_separate:
        base_path = join_path(dir_gallery, str(key))
        create_dir(base_path, show_msg=False)
    else:
        base_path = dir_gallery
    return base_path

def crop_frame(frame, bbox):
    # frame is type numpy array (imagen RGB) 
    # bbox is a list = [x0,y0, x1,y1]   
    return frame[y0:y1, x0,x1] 



