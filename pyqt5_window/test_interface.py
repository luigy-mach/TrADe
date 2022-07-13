import os
import sys
import fnmatch
import numpy as np
import pandas as pd


from .gui.others      import save_data, find_files, check_id_into_gts
from interface        import main_evaluation


if __name__ == '__main__':
    
    # query_path, imgs_path = open_images()
    beta                  = 0.5
    eta                   = 10 # top rank (ex: 10, 20)
    tau                   = 10 # number frames (ex: 10, 100, 1000)
    # save_path             = '/home/luigy/luigy/develop/re3/tracking/pyqt5_window/out_test'


    mainDir     = '/home/luigy/luigy/develop/re3/tracking/pyqt5_window/dir_test/A-B_000001'
    queryName   = 'otherCam_person_0017.png'
    id_query    = 17
    qName, qExt = os.path.splitext(queryName)
    qAbsPath    = os.path.join(mainDir, queryName)

    galleryDir  = '/home/luigy/luigy/develop/re3/tracking/pyqt5_window/dir_test/A-B_000001/tau_01000/seq_00002/outcome_otherCam_person_0017_all'
   	

    imgs_path   = find_files(galleryDir, '*png', type='absolute')
    name_file   = 'out_GUI_result.csv'
    
    file_gts    = '/home/luigy/luigy/develop/re3/tracking/pyqt5_window/dir_test/A-B_000001/tau_01000/seq_00002/gts_tau_frameStart_00002000.csv'
    file_reid   = os.path.join(galleryDir, 'outcome_{}_dataframe.csv'.format(qName))
    # print(file_reid)
    
    main_evaluation(beta, eta, tau, id_query, qAbsPath, imgs_path, file_gts, file_reid, save_path=galleryDir, name_file=name_file)


