
import pickle
import pandas as pd
import numpy as np
import cv2
import os
import shutil


from handler_others import Dictlist
from handler_file import *
from myglobal import key_words 

        
        
class inliersFromDir():
    def __init__(self, path_inliers=None,  key_words=None ):
        self._structInliers  = Dictlist()
        self._path_inliers   = path_inliers
        self._keywords       = self.set_keywords(key_words)
    
    def get_struct(self):
        return self._structInliers
    
    def set_keywords(self, keywords):
        self._keywords = keywords

    def set_pathInliers(self, path):
        self._path_inliers = path 

    
    def run_inliers(self, top = 1):
        assert self._path_inliers != None, "err, you need setting a path {}".format(self._path_inliers)
        assert os.path.exists(self._path_inliers), 'doesnt exist: {}'.format(self._path_inliers)
        
        dir_inliers = sorted(os.listdir(self._path_inliers))
        for i, id_dir in enumerate(dir_inliers):
            dpath = os.path.join(self._path_inliers, id_dir, 'inliers')
            list_temp = sorted(os.listdir(dpath))[:top]
            if len(list_temp)>0:
                for file in list_temp:
                    mini_dict = None
                    mini_dict = split_fpath(file)
                    mini_dict['path'] = os.path.join(dpath, file)
                    self._structInliers[id_dir] = mini_dict
        return self._structInliers
    
    
    def save_inliers(self, path_gallery_dst):
        assert len(self._structInliers)>0, 'err, inliers are empty'
        assert os.path.exists(path_gallery_dst), 'err, {} doesnt exist'.format(path_gallery_dst)
        keys = sorted(list(self._structInliers.keys()))
        for k in keys:
            out = self._structInliers[k]
            if len(out)>0:
                for i in out:
                    path_src = i['path']
                    shutil.copy2(path_src, path_gallery_dst)
                    
        return
        




if __name__ == '__main__':

    path_inliers = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/separate'
    dst_gallery  = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/borrar_gallery_test'
    test1        = inliersFromDir(path_inliers=path_inliers, key_words=key_words)
    inliers      = test1.run_inliers(top=2)
    test1.save_inliers(dst_gallery)
