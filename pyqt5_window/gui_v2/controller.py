

from functools import partial
# Create a Controller class to connect the GUI and the model
from PyQt5.QtWidgets import qApp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSlot

import time

import sys


qIsIn_eta  = 'qIsIn_eta'
resultType = 'type'

class class_controller:
    """PyCalc's Controller."""
    def __init__(self, model, view, pyEval, dictResult):
        """Controller initializer."""
        self._model             = model
        self._view              = view
        self._pyEval            = pyEval

        self._dictResult        = None
        self._dictResult_backup = None
        self._dictResult        = dictResult.copy()
        self._dictResult_backup = self._dictResult.copy()

        self._connectSignalsButtons()
        self._connectSignalsButtonFail()

        
    # def reset_dictNames(self):
    #     self._dictResult = self._dictResult_backup.copy()
    #     return self._dictResult_backup



    def _connectSignalsButtons(self):
        """Connect signals and slots."""
        self._view.buttonTrueCall.clicked.connect(partial(self._actionButtons))


    # @pyqtSlot()
    def _actionButtons(self):

        if self._view.buttonTrueCall.isChecked():
            self._view.buttonTrueCall.setStyleSheet("background-color : green")
            # self._view.buttonTrueCall.setEnabled(True)
            # self._dictResult[qIsIn_eta] = 1
        # else:
        #     self._view.buttonTrueCall.setStyleSheet("background-color : lightgrey")
        #     # self._view.buttonTrueCall.setStyleSheet("")
        #     # self._view.buttonTrueCall.setEnabled(True)
        #     self._dictResult[qIsIn_eta] = 0

        
        self._dictResult[qIsIn_eta]  = 1
        self._dictResult[resultType] = 'TrueCall'


        self._model._save_data(self._dictResult)
        self._dictResult        = None
        self._dictResult_backup = None

        self._view.close()





    def _connectSignalsButtonFail(self):
        """Connect signals and slots."""
        self._view.buttonFail.clicked.connect(self._actionButtonFail)

    # @pyqtSlot()
    def _actionButtonFail(self):

        if self._view.buttonFail.isChecked():
            self._view.buttonFail.setStyleSheet("background-color : red")
            self._view.buttonFail.setEnabled(False)
        # else:
        #     self._view.buttonFail.setStyleSheet("background-color : lightgrey")
        #     # self._view.buttonFail.setStyleSheet("")
        #     # self._view.buttonFail.setEnabled(True)

    
        self._dictResult[qIsIn_eta] = 0
        self._dictResult[resultType] = 'TrueMissedCall'


        self._model._save_data(self._dictResult)
        self._dictResult        = None
        self._dictResult_backup = None

        self._view.close()






