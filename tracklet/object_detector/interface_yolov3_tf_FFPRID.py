# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

import glob
import os
import tensorflow as tf
import time

import sys
import os.path



from .pYOLO.core import utils
#from pReID import cuhk03_dataset


# basedir = os.path.abspath(os.path.dirname(__file__))
basedir = '/home/luigy/luigy/develop/re3/tracking/tracklet/object_detector'


# Network Constaqnts
# CROP_SIZE        = 227
# CROP_PAD         = 2
# MAX_TRACK_LENGTH = 32
# LSTM_SIZE        = 512

LOG_DIR          = os.path.join(os.path.dirname(__file__), 'logs')
GPU_ID           = '0'

# Drawing constants
# OUTPUT_WIDTH     = 640
# OUTPUT_HEIGHT    = 480
# PADDING          = 2



#****1********PARAMETERS OD**********
#PARAMETRO
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./tracklet/object_detector/pYOLO/yolov3_coco.pb"


# PARA LA DETECCION DE OBJETOS
num_classes     = 80
input_size      = 416
graph           = tf.Graph()
#
#****2********PARAMETERS OD**********

#****1********PARAMETERS REID**********
IMAGE_WIDTH  = 60
IMAGE_HEIGHT = 160

#****2********PARAMETERS REID**********

class obj_detection(object):
    def __init__(self):
        print("********************OBJ DETECTION FFPRID***************************1")
        basedir              = os.path.dirname(__file__)
        self.return_elements = return_elements
        self.pb_file         = pb_file
        self.num_classes     = num_classes
        self.input_size      = input_size
        self.graph           = graph
        self.return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
        self.sess            = tf.Session(  graph = self.graph, 
                                            config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.30), 
                                                                                            allow_soft_placement=True)
                                          )


        # self.info            = ''
        self.initBoundingBox = dict()
        self.ids             = list()
        self.id_ite          = 1
        print("********************OBJ DETECTION FFPRID***************************2")

 
    def detect(self, frame_rgb):
        frame_size = frame_rgb.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame_rgb), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run( [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
                                                            feed_dict={ self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)) ,
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)) ,
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))],
                                    axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        #print("***** cropping *******")
        self.ids.clear()
        self.initBoundingBox.clear()
        # for bbox in bboxes:
                    # if self.classes[labels_[i]]=='person': 
        ### bbox[5] == 0 , es solo para personas 
        for i in range(len(bboxes)):
            x0, y0, x1, y1, score, class_name= bboxes[i]
            if class_name == 0: #n_frame % num_crop == 0
                initial_bb = []
                initial_bb.append((int)(x0))
                initial_bb.append((int)(y0))
                initial_bb.append((int)(x1))              
                initial_bb.append((int)(y1))
                # self.ids.append(str(self.id_ite))
                # self.initBoundingBox.update({str(self.id_ite): initial_bb})
                self.ids.append(self.id_ite)
                self.initBoundingBox.update({self.id_ite: initial_bb})
                self.id_ite+=1

        return self.ids, self.initBoundingBox 


# Main testing 
if __name__ == "__main__":
    
    photo                = '/home/luigy/LUIGY/tracking/demo/yolov3_tf/data/demo_data/luchardor2.jpg'

    #agregar opencv open photo
    myobjdetec           = obj_detection()
    ids, initBoundingBox = myobjdetec.detect(photo)

    
  