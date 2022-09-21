import os
import sys
import fnmatch
import numpy  as np
import pandas as pd

from .gui_v2.view        import class_view
from .gui_v2.controller  import class_controller
from .gui_v2.model       import class_model
from .gui_v2.others      import save_data, find_files, check_id_into_gts
from PyQt5.QtWidgets     import QApplication



qIsIn_eta  = 'qIsIn_eta'
resultType = 'type'


def main_GUI_eval( beta, eta, tau, query_path, imgs_path, numShowImgs, colNames, dictResult, save_path, title=None, addtext=None):
  
    # Create an instance of `QApplication`
    # pyEval      = QApplication(sys.argv)
    pyEval      = QApplication([])
    # pyEval.exec_()

    """Main function."""
    model       = class_model(colNames, dictResult, save_path)

    # Show the GUI
    # numShowImgs = eta if len(imgs_path)>eta else len(imgs_path)
    view        = class_view(query_path, imgs_path, numShowImgs, beta, eta, tau, title, addtext)
    # view.show()
    
    # Create GUI of the model and the controller
    controller = class_controller(model, view, pyEval, dictResult)
    controller._view.show()                         
    # Execute GUI's main loop
    # sys.exit(pyEval.exec_())
    # sys.exit(pyEval.exec())
    # pyEval.quit()
    pyEval.exec()
    pyEval.exit()
    # pyEval.quit()
    # del pyEval
    # del model
    # del view


def main_evaluation(beta, eta, tau, query_id, query_path, imgs_path, gts_csv, reid_csv, save_path, name_file, title=None, addtext=None, show_debug=False):
    assert os.path.exists(query_path), "error, dont find query_path {}".format(save_path)
    assert os.path.exists(save_path) , "error, dont find save_path {}".format(save_path)
    assert len(imgs_path)>0          , "error, didnt find images in  {}".format(imgs_path)
    assert os.path.exists(gts_csv)   , "error, dont find gts_csv {}".format(gts_csv)
    assert os.path.exists(reid_csv)  , "error, dont find reid_csv {}".format(reid_csv)
    for i in imgs_path:
        assert os.path.exists(i), "error, dont find image {}".format(i)

    absSavePath     = os.path.join(save_path, name_file)
    
    _qId            = query_id
    _gts            = pd.read_csv(reid_csv).fillna(value=0)
    _highSimilarity = _gts.iloc[0][2]
    _truth_gt       = check_id_into_gts(gts_csv, _qId)
    _beta           = beta
    _eta            = eta
    _tau            = tau
    _qPath          = query_path
    _imgsPath       = imgs_path
    _numImgs        = _eta if len(_imgsPath)>_eta else len(_imgsPath)
    _savePath       = absSavePath

    _col_names      = ['queryAbsPath','qId','qIsPesentGT','beta','eta','tau','highSimilarity','qIsIn_eta','type']
    values          = [-1 for _ in range(len(_col_names))]

    _dict_result                   = dict(zip(_col_names, values))
    _dict_result['queryAbsPath']   = _qPath
    _dict_result['qId']            = _qId
    _dict_result['qIsPesentGT']    = _truth_gt
    _dict_result['beta']           = _beta
    _dict_result['eta']            = _eta
    _dict_result['tau']            = _tau
    _dict_result['highSimilarity'] = _highSimilarity
    _dict_result['qIsIn_eta']      = 0
    _dict_result['type']           = 'Undefined'



    if _truth_gt == True:
        if _beta < _highSimilarity:
            ## TrueCall
            ## TrueMissedCall
            # print("----------------------------------------------------------")
            # print('TrueCall/TrueMissedCall')
            # pass
            main_GUI_eval(_beta, _eta, _tau, _qPath, _imgsPath, _numImgs, _col_names,  _dict_result, _savePath, title, addtext)
            
            # print("self._eta: ",_eta)
            # print("self._numImgs:", _numImgs)
            # print("self._imgsPath ",_imgsPath)
            
            # print("----------------------------------------------------------")
            
        elif _beta > _highSimilarity:
            ## FalseSilence
            if show_debug:
                print('FalseSilence')
            _dict_result[resultType] = 'FalseSilence'
            save_data( _col_names, _dict_result, _savePath )

        else:
            breakpoint()
            print('ERROR 1')

    elif _truth_gt == False:
        if _beta < _highSimilarity:
            ## FalseCall
            if show_debug:
                print('FalseCall')
            _dict_result[resultType] = 'FalseCall'
            save_data( _col_names, _dict_result, _savePath )

        elif _beta > _highSimilarity:
            ## TrueSilence
            if show_debug:
                print('TrueSilence')
            _dict_result[resultType] = 'TrueSilence'
            save_data( _col_names, _dict_result, _savePath )

        else:
            print('ERROR 2')


    else:
        print("ERROR _truth_gt - {}".format(_truth_gt))
        _dict_result[resultType] = 'errorGUI'
        save_data( _col_names, _dict_result, _savePath )
        


