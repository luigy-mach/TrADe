

import pickle
import pandas as pd
import numpy as np
import cv2
import os
import shutil


from .handler_others import Dictlist
from .handler_file import *
from .myglobal import key_words as dictkeywords
        
        
class trackletsFromDir():
    def __init__(self, path=None,  keywords=None ):
        self._structGallery  = dict()
        self._path_crops   = path
        if keywords:
            self._keywords      = self.set_keywords(keywords)
        else:
            self._keywords      = self.set_keywords(dictkeywords)

    
    def get_struct(self):
        return self._structGallery
    
    def set_keywords(self, keywords):
        self._keywords = keywords

    def set_pathGallery(self, path):
        self._path_crops = path 
        
    def compile_gallery(self):
        assert self._path_crops != None, "err, you need setting a path"
        assert os.path.exists(self._path_crops), 'doesnt exist: {}'.format(self._path_crops)
        list_crops = sorted(os.listdir(self._path_crops))
        for i, id_dir in enumerate(list_crops):
            dpath = os.path.join(self._path_crops, id_dir)
            tmp_dict  = dict()
            list_temp = sorted(os.listdir(dpath))
            if len(list_temp)>0:
                for file in list_temp:
                    mini_dict = split_fpath(file)
                    tmp_dict[mini_dict['frame'][0]] = mini_dict
                self._structGallery[id_dir] = tmp_dict
        return self._structGallery
    
    
    def query_args(self, *ids_query):
        result = Dictlist()
        for idx in ids_query:
            tmp = self._structGallery[idx]
            for key, value in tmp.items():
                result[key] = value

                
    def query_gallery(self, ids_query):
        if isinstance(ids_query,dict):
            keys = ids_query.keys()
        elif isinstance(ids_query,list):
            keys = ids_query
        else:
            print("error input query_gallery")
            return None

        result = Dictlist()
        for idx in keys:
            tmp = self._structGallery.get(idx)
            if tmp is not None:
                for key, value in tmp.items():
                    result[key] = value
        return result
    
    def show():
        for k,v in out_query.items():
#             print(k,v)
            print(k)
            for i in v:
                print("    >: {}".format(i['id']))
      



if __name__ == '__main__':
    
    from handler_video import draw_bboxes_over_video

       ################ READ out-Reid old version FF-PRID-2020
    path2 = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/out_reid_top10/score_results.csv'
    data  = pd.read_csv(path2, header= None)

    out_reid = list()

    for i,value in data.iterrows():
        out_reid.append([i,value[1],value[3]])

    # for i in out_reid:
    #     print(i)

    querys_dict = dict() 
    for i in out_reid:
        tmp                    = split_fpath(i[1])
        tmp['reid_pos']        = [i[0]]
        tmp['score']           = [i[2]]
        tmp2 = querys_dict.get(tmp['id'][0])
    #     if  tmp2 is not None:
    #         score = min(tmp['score'][0], tmp2['score'][0] )
    #     tmp['score']           = [score]
        querys_dict[tmp['id'][0]] = tmp

    query_reid = list(querys_dict.keys())
    print(len(query_reid))
    print(query_reid)

    #################################################################

    path_crops = '/home/luigy/luigy/develop/re3/tracking/resultados/pack_5_min_complete_v3/cropping/TownCentreXVID' 
    test         = trackletsFromDir(path=path_crops, key_words=key_words)
    # test.set_pathGallery(path_gallery)
    out          = test.run_gallery()
    query_bboxes    = test.query_gallery(query_reid)
    print("///////////////////////////////")
    # print(query_bboxes)
    print("///////////////////////////////")

    #################################################################

    path_video = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_30seg.avi'
    path_dest  = './basura/'

    draw_bboxes_over_video(path_video, path_dest, query_bboxes)