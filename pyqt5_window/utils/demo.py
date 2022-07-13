



import pickle
import pandas as pd
import numpy  as np
import cv2
import os
import shutil

from class_inliers          import inliersFromDir
from handler_video          import draw_bboxes_over_video
from class_read_tracklets   import trackletsFromDir
from handler_file           import *


if __name__ == '__main__':

    main_path        = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing'
    path_inliers     = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/separate'
    
    path_crops       = '/home/luigy/luigy/develop/re3/tracking/resultados/pack_5_min_complete_v3/cropping/TownCentreXVID'
   
    # path_video       = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_30seg.avi'
    path_video       = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID.avi'


    img_query_path   = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/query/query_id_92.png'
    # img_query_path = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/query/query_id_13.png'
    
    check_exists(main_path, path_crops, path_video, path_inliers, img_query_path)

    
    pickup_inliers = 2
    top_reid       = 10 


    ##################################################################

    dst_gallery, out_reid_path , out_path_video = generate_new_pack(main_path, posfix='top_{}_numInlier_{}'.format(top_reid, pickup_inliers))

    inliersClass = inliersFromDir(path_inliers)
    inliersClass.run_inliers(top=pickup_inliers)    
    inliersClass.save_inliers(dst_gallery)

    # dst_gallery = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/pack_v6/gallery_inliers'
    ################ READ out-Reid old version FF-PRID-2020
    import sys
    import os.path
    basedir = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
    from pReID.personReID import personReIdentifier

    ##################################################################

    reidentifier  = personReIdentifier()
    dict_reid     = reidentifier.PersonReIdentification(img_query_path, dst_gallery, out_reid_path, topN=top_reid)
    # reidentifier.PersonReIdentification(img_query_path, dst_gallery, out_reid_path, topN=30)
    
    print("***********************************************")
    print("***********************************************")
    # print(len(dict_reid))
    # print("***********************************************")
    # print("***********************************************")
    # print(dict_reid)
    print("***********************************************")
    print("***********************************************")

    query_reid = list() 
    for k,v in dict_reid.items():
        query_reid.append(v['id'][0])
    print("***********************************************")
    # print(query_reid)
    print("***********************************************")

    # # #################################################################

    test         = trackletsFromDir( path = path_crops, key_words = key_words )
    out          = test.run_gallery()
    query_bboxes = test.query_gallery(query_reid)
    print("///////////////////////////////")
    # print(query_bboxes)
    print("///////////////////////////////")

    # #################################################################

    number_frame, fps, frame_width, frame_height = draw_bboxes_over_video(path_video, out_path_video, query_bboxes)

    save_set_time(dict_reid, fps, name_save='dict_reid.json', path_save=out_path_video )
