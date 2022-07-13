
import cv2

from PyQt5 import QtGui

import pandas as pd
import os
import numpy as np

from .others import save_data


ERROR_MSG = 'ERROR'


def check_id_into_gts(gts_dfPath, id):
    if isinstance(id,str):
        id = int(id)
    assert id>=0, 'error, id is negative'
    df = pd.read_csv(gts_dfPath)
    ids = df['id'].to_numpy()
    ids = np.unique(ids)
    qID = np.where(ids==id)
    if qID[0].size > 0:
        return True
    else:
        return False
    


class class_model():

    def __init__(self, col_names, dict_result, savePath):
        
        self._dataFrame   = None
        self._col_names   = None
        self._dict_result = None
        self._col_names   = col_names
        self._dict_result = dict_result
        self._dataFrame   = pd.DataFrame(columns=self._col_names)
        self._savePath    = savePath


    def get_savePath(self):
        return self._savePath

    def get_dictNames(self):
        return self._dict_result

    def _add_row(self, array_in):
    	self._dataFrame.loc[len(self._dataFrame)] = array_in

    def _save_data(self, _dict_result):
        save_data(self._col_names, _dict_result, self._savePath)
        # self._dataFrame   = None
        # self._col_names   = None
        # self._dict_result = None
    