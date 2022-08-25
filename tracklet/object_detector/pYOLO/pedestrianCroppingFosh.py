#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : pedestrianCropping.py
#   Author      : FelixSumari
#   Created date: 2020-02-01 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
#video_path      = "./docs/images/road.mp4"
#video_path      = "./dataPRID/cam_b.avi"
video_path = "/home/oliver/Documentos/2019B/EO/PRID2011-RAW/prid2011_videos/CamA_seqVideo/000118/video_in.avi"
# video_path      = 0
#out_crop_path   = "./results/auto"
#out_crop_path   = "./results/cropsPRID/CamB_v2"
out_crop_path = "/home/oliver/Documentos/2019B/EO/PRID2011-RAW/prid2011_videos/CamA_seqVideo/000118/cropps"

if not os.path.isfile(out_crop_path):
    os.system('mkdir '+out_crop_path)

num_classes     = 80
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    
    n_frame = 1
    #num_crop = 10#para q vaya de 10 en 10
    
    ID_p = 1


    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            copy_frame=frame.copy()
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        ##############################################
        

        #print("frame > bbox",n_frame, bboxes)
        # SOLO PARA PINTAR, LO COMENTE #
        #image = utils.draw_bbox(frame, bboxes)
        ###############3
        print("***** cropping *******")

        for bbox in bboxes:
            pos_bb=str(int(bbox[1]))+'_'+str(int(bbox[3]))+'_'+str(int(bbox[0]))+'_'+str(int(bbox[2]))
            pos_bb2=str(int(bbox[1]))+','+str(int(bbox[3]))+','+str(int(bbox[0]))+','+str(int(bbox[2]))
            ### bbox[5] == 0 , es solo para personas 
            if bbox[5] == 0: #n_frame % num_crop == 0
                name_crop= out_crop_path+'/'+ str(ID_p) +'_'+str(n_frame)+'_'+pos_bb+'.png'
                crop_img = copy_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                ID_p += 1
                cv2.imwrite(name_crop,crop_img)

        n_frame +=1
        print("***** ******* *******")

        #############3
        curr_time = time.time()
        exec_time = curr_time - prev_time
        #result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        #####
        #####cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        
        #result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #####result = cv2.resize(result, (960, 540))
        #cv2.imshow("result", result)
        #if cv2.waitKey(1) & 0xFF == ord('q'): break




