# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from .yolov3_tf.utils.misc_utils import parse_anchors, read_class_names
from .yolov3_tf.utils.nms_utils import gpu_nms
from .yolov3_tf.utils.plot_utils import get_color_table, plot_one_box
from .yolov3_tf.utils.data_aug import letterbox_resize
from .yolov3_tf.model import yolov3


import glob
import numpy as np
import os
import tensorflow as tf
import time

import sys
import os.path


# basedir = os.path.abspath(os.path.dirname(__file__))
basedir = '/home/luigy/luigy/develop/re3/tracking/tracklet/object_detector'


# Network Constaqnts
# CROP_SIZE        = 227
# CROP_PAD         = 2
# MAX_TRACK_LENGTH = 32
# LSTM_SIZE        = 512

import os.path
LOG_DIR          = os.path.join(os.path.dirname(__file__), 'logs')
GPU_ID           = '0'

# Drawing constants
# OUTPUT_WIDTH     = 640
# OUTPUT_HEIGHT    = 480
# PADDING          = 2





class obj_detection():
    def __init__(self,gpu_id=GPU_ID):
        print("********************OBJ DETECTION CLASSIC***************************1")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        basedir                            = os.path.dirname(__file__)
        self.g2                            = tf.Graph()
        self.id_ite                        = 1
    
        self.sess = tf.Session( graph=self.g2,
                                config=tf.ConfigProto(  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30), 
                                                        allow_soft_placement=True)
                               )
        # self.sess = tf.Session(graph=self.g2,config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4), allow_soft_placement=True))

        with self.g2.as_default():
            # self.anchor_path      = "./demo/yolov3_tf/data/yolo_anchors.txt"
            self.anchor_path        = os.path.join(basedir,"yolov3_tf/data/yolo_anchors.txt")
            self.new_size           = [416, 416]
            self.myletterbox_resize = lambda x: (str(x).lower() == 'true')
            #myletterbox_resize     = True
            self.class_name_path    = os.path.join(basedir,"yolov3_tf/data/coco.names")
            self.restore_path       = os.path.join(basedir,"yolov3_tf/data/darknet_weights/yolov3.ckpt")
            
            
            self.anchors            = parse_anchors(self.anchor_path)
            self.classes            = read_class_names(self.class_name_path)
            self.num_class          = len(self.classes)
            self.color_table        = get_color_table(self.num_class)
            
            self.initBoundingBox    = dict()
            self.ids                = list()
            
            #self.sess              = tf.Session(config                                                           = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True), allow_soft_placement = True))
            #sess                   = tf.Session()
            self.input_data         = tf.placeholder(tf.float32, [1, self.new_size[1], self.new_size[0], 3], name = 'input_data')
            self.yolo_model         = yolov3(self.num_class, self.anchors)
            
            #tf.variable_scope('yolov3') 
            #self.pred_feature_maps = self.yolo_model.forward(self.input_data, False)
            with tf.variable_scope('yolov3'):
                self.pred_feature_maps = self.yolo_model.forward(self.input_data, False)
            #self.saver = tf.train.Saver()
            #self.saver.restore(sess, self.restore_path)
                
            #tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4), allow_soft_placement=True))

            
            # self.sess = tf.Session(graph=self.g2,config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4), allow_soft_placement=True))
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))

            self.sess.run(tf.global_variables_initializer())
            #self.ckpt = tf.train.get_checkpoint_state('/home/luigy/isabo/re3/demo/yolov3_tf/data/darknet_weights')
            #if self.ckpt is None:
            #    raise IOError(
            #            ('Checkpoint model could not be found. '
            #            'Did you download the pretrained weights? '
            #            'Download them here: http://bit.ly/2L5deYF and read the Model section of the Readme.'))
            #self.sees.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.restore_path)

            self.pred_boxes, self.pred_confs, self.pred_probs = self.yolo_model.predict(self.pred_feature_maps)
            self.pred_scores                                  = self.pred_confs * self.pred_probs
            # print("******************** DELAY ***************************1")
            self.boxes, self.scores, self.labels              = gpu_nms(self.pred_boxes, 
                                                                        self.pred_scores, 
                                                                        self.num_class, 
                                                                        max_boxes    = 200, 
                                                                        score_thresh = 0.5, 
                                                                        nms_thresh   = 0.45)
            # self.boxes, self.scores, self.labels = gpu_nms(self.pred_boxes, self.pred_scores, self.num_class, max_boxes=200, score_thresh=0.2, nms_thresh=0.45)
            # print("******************** DELAY ***************************2")
        

        print("********************OBJ DETECTION CLASSIC***************************2")


    def detect(self, new_frame):

        #cap = cv2.VideoCapture(input_video_)
        #if (cap.isOpened()==False): 
        #    print("Error opening video stream or file: object detection video")
        #self.ret, self.frame = cap.read()  
        #self.input_image = self.frame 
        
        #self.sess = tf.Session(graph=self.g2)
        # print("******************** DETECT ***************************1")

        with self.g2.as_default():
            #with tf.Session(graph=self.g2) as sess:
            #self.img_ori = self.input_image
            img_ori = new_frame

            if self.myletterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, self.new_size[0], self.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img                   = cv2.resize(img_ori, tuple(self.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.


            #pred_feature_maps = self.yolo_model.forward(self.input_data, False)

            ## pred_boxes, pred_confs, pred_probs = self.yolo_model.predict(self.pred_feature_maps)
            ## pred_scores = pred_confs * pred_probs
            ## # print("******************** DELAY ***************************1")
            ## boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
            ## # print("******************** DELAY ***************************2")
        
            #saver = tf.train.Saver()
            #saver.restore(sess, self.restore_path)

            
            # print("******************** self.sess.run ***************************1")
            boxes_, scores_, labels_ = self.sess.run([self.boxes, self.scores, self.labels], feed_dict={self.input_data: img})
            # print("******************** self.sess.run ***************************2")
        
            # rescale the coordinates to the original image
            if self.myletterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))
        
            #print("box coords:")
            #print(boxes_)
            #print('*' * 30)
            #print("scores:")
            #print(scores_)
            #print('*' * 30)
            #print("labels:")
            #print(labels_)
        
            #self.id_ite=1
            self.ids.clear()
            self.initBoundingBox.clear()
            for i in range(len(boxes_)):
                if self.classes[labels_[i]]=='person': 
                    x0, y0, x1, y1 = boxes_[i]
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
                    #plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
            #cv2.imshow('Detection result', img_ori)
            #cv2.imwrite('detection_result.jpg', img_ori)
            #cv2.waitKey(0)

            
        #     print("self.ids:")
        #     print(self.ids)
        #     print("self.initBoundingBox")
        #     print(self.initBoundingBox)
        # print("******************** DETECT ***************************2")

        return self.ids, self.initBoundingBox 



# Main testing 
if __name__ == "__main__":
    
    photo                = '/home/luigy/LUIGY/tracking/demo/yolov3_tf/data/demo_data/luchardor2.jpg'

    dir_gallery          = '/home/luigy/isabo/videos_copacabana/trash47-test/'
    # create_dir(dir_gallery)

    #agregar opencv open photo
    myobjdetec           = obj_detection()
    ids, initBoundingBox = myobjdetec.detect(photo)

    
  