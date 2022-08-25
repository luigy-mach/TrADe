import cv2
import time
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
import os
#********************

import matplotlib.pyplot as plt
import glob

# Con esto, retrocedi un carpeta .. para agregar , pYOLO
#basedir = os.path.dirname(__file__)
#print('base : ',basedir)
#sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
#print('base2 :',os.path.dirname(__file__))
##************************************************************************

from pYOLO.core import utils
#from pReID import cuhk03_dataset


#****1********PARAMETERS OD**********
#PARAMETRO
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "pYOLO/yolov3_coco.pb"

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

class YOLOcropper(object):
    def __init__(self):
        self.return_elements = return_elements
        self.pb_file         = pb_file
        self.num_classes     = num_classes
        self.input_size      = input_size
        self.graph           = graph
        self.return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
        self.sess            = tf.Session(graph = self.graph)
        self.info            = ''

    def personCropping(self, video_in_path, cropps_out_path):
        
        vid = cv2.VideoCapture(video_in_path)        
        n_frame = 1
        #num_crop = 10#para q vaya de 10 en 10    
        ID_p = 1
        print("CROPPING..............")
        while True:
            return_value, frame = vid.read()
            #print('hay video: ' , return_value)
            #print(video_in_path)
            if return_value:
                frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                copy_frame = frame.copy()
                image      = Image.fromarray(frame)
            else:
                #raise ValueError("No image!")
                break
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time  = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
                [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
                        feed_dict={ self.return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            ###############
            #print("***** cropping *******")
            for bbox in bboxes:
                pos_bb  = str(int(bbox[1]))+'_'+str(int(bbox[3]))+'_'+str(int(bbox[0]))+'_'+str(int(bbox[2]))
                pos_bb2 = str(int(bbox[1]))+','+str(int(bbox[3]))+','+str(int(bbox[0]))+','+str(int(bbox[2]))
                ### bbox[5] == 0 , es solo para personas 
                if bbox[5] == 0: #n_frame % num_crop == 0
                    name_crop = cropps_out_path +'/'+ str(ID_p) +'_'+str(n_frame)+'_'+pos_bb+'.png'
                    crop_img  = copy_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    crop_img  = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                    ID_p     += 1
                    cv2.imwrite(name_crop,crop_img)

            n_frame +=1
            #print("********************")
            ##############
            curr_time = time.time()
            exec_time = curr_time - prev_time
            #result = np.asarray(image)
            self.info = "time of cropping: %.2f ms" %(1000*exec_time)
        print(self.info)