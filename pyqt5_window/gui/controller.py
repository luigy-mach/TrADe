

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
        self._connectSignalsClear()
        self._connectSignalsSaveExit()

        
    def reset_dictNames(self):
        self._dictResult = self._dictResult_backup.copy()
        return self._dictResult_backup

    def _connectSignalsButtons(self):
        """Connect signals and slots."""
        for btnText, btn in self._view.buttons.items():
            btn.clicked.connect(partial(self._actionButtons, btnText))
            # btn.pressed.connect(partial(self._actionButtons, btnText))
            # btn.released.connect(partial(self._actionButtons2, btnText))


    # @pyqtSlot()
    def _actionButtons(self, btnText):
        for _btnText, _ in self._view.buttons.items():
            if _btnText == btnText:
                if _btnText != 'Query':
                    if self._view.buttons[_btnText].isChecked():
                        self._view.buttons[_btnText].setStyleSheet("background-color : green")
                        # self._view.buttons[_btnText].setEnabled(True)
                        self._dictResult[_btnText] = 1
                    else:
                        self._view.buttons[_btnText].setStyleSheet("background-color : lightgrey")
                        # self._view.buttons[_btnText].setStyleSheet("")
                        # self._view.buttons[_btnText].setEnabled(True)
                        self._dictResult[_btnText] = 0


        self._dictResult[qIsIn_eta] = self._check_ranks()

        if self._dictResult[qIsIn_eta]>0:
            self._view.buttonSave['save'].setEnabled(True)
        else: 
            self._view.buttonSave['save'].setEnabled(False)


        if self._dictResult[qIsIn_eta]>0:
            self._view.buttonFail['fail'].setEnabled(True)
            self._view.buttonFail['fail'].setCheckable(False)
            self._view.buttonFail['fail'].setCheckable(True)
            self._view.buttonClear['clear'].setEnabled(True)
        else: 
            self._view.buttonFail['fail'].setEnabled(False)
        

    # def _actionButtons2(self, btnText, flag):

    #     self._dictResult[qIsIn_eta] = self._check_ranks()

    #     if self._dictResult[qIsIn_eta]>0:
    #         self._view.buttonFail['fail'].setEnabled(True)
    #         self._view.buttonFail['fail'].setCheckable(False)
    #         self._view.buttonFail['fail'].setCheckable(True)
    #         self._view.buttonClear['clear'].setEnabled(True)
    #         self._view.buttonSave['save'].setEnabled(True)
    #     else: 
    #         self._view.buttonFail['fail'].setEnabled(False)
    #         self._view.buttonSave['save'].setEnabled(False)



    def _check_ranks(self,):
        count = 0
        for _btnText, _ in self._view.buttons.items():
            if _btnText != 'Query':
                count += self._dictResult[_btnText] 
        return count


    def _connectSignalsButtonFail(self):
        """Connect signals and slots."""
        for btnText, btn in self._view.buttonFail.items():
            btn.clicked.connect(partial(self._actionButtonFail, btnText))

    # @pyqtSlot()
    def _actionButtonFail(self, btnText):
        self._dictResult = self.reset_dictNames()
        for _btnText, _ in self._view.buttonFail.items():
            if _btnText == btnText:
                # self._view.buttonFail[btnText].setEnabled(False)
                if self._view.buttonFail[_btnText].isChecked():
                    self._view.buttonFail[_btnText].setStyleSheet("background-color : red")
                    self._view.buttonFail[_btnText].setEnabled(False)
                else:
                    self._view.buttonFail[_btnText].setStyleSheet("background-color : lightgrey")
                    # self._view.buttonFail[_btnText].setStyleSheet("")
                    # self._view.buttonFail[_btnText].setEnabled(True)

        for _btnText, _ in self._view.buttons.items():
            if _btnText != 'Query':
                self._view.buttons[_btnText].setStyleSheet("background-color : lightgrey")
                # self._view.buttons[_btnText].setStyleSheet("")
                # self._view.buttons[_btnText].setDefault(True)
                self._view.buttons[_btnText].setEnabled(False)
                self._view.buttons[_btnText].setCheckable(False)
                self._view.buttons[_btnText].setCheckable(True)
                self._dictResult[_btnText] = 0

        self._view.buttonClear['clear'].setEnabled(True)
        self._view.buttonSave['save'].setEnabled(True)



    def _connectSignalsClear(self):
        """Connect signals and slots."""
        for btnText, btn in self._view.buttonClear.items():
            # self._view.buttonClear[btnText].
            btn.clicked.connect(partial(self._actionButtonClear, btnText))
        
        # self._connectSignalsSaveExit()
        

    # @pyqtSlot()
    def _actionButtonClear(self, btnText):
        self._view.buttonSave['save'].setEnabled(False)
        self._dictResult  = self.reset_dictNames()

        for _btnText, _ in self._view.buttonClear.items():
                self._view.buttonClear[_btnText].setEnabled(False)
                self._view.buttonClear[_btnText].setCheckable(False)
                self._view.buttonClear[_btnText].setCheckable(True)

        for _btnText, _ in self._view.buttonFail.items():
                self._view.buttonFail[_btnText].setEnabled(True)
                self._view.buttonFail[_btnText].setCheckable(False)
                self._view.buttonFail[_btnText].setCheckable(True)
                self._view.buttonFail[_btnText].setStyleSheet("background-color : lightgrey")


        for _btnText, _ in self._view.buttons.items():
            if _btnText != 'Query':
                self._view.buttons[_btnText].setStyleSheet("background-color : lightgrey")
                # self._view.buttons[_btnText].setStyleSheet("")
                # self._view.buttons[_btnText].setDefault(True)
                self._view.buttons[_btnText].setEnabled(True)
                self._view.buttons[_btnText].setCheckable(False)
                self._view.buttons[_btnText].setCheckable(True)
                self._dictResult[_btnText] = 0


    def _connectSignalsSaveExit(self):
        """Connect signals and slots."""
        for btnText, btn in self._view.buttonSave.items():
            btn.clicked.connect(partial(self._actionButtonSaveExit, btnText))


    # @pyqtSlot()
    def _actionButtonSaveExit(self, btnText, dictResult):
        self._view.exitButton()
        # breakpoint()
        if self._dictResult[qIsIn_eta]>0:
            self._dictResult[resultType] = 'TrueCall'
        else:
            self._dictResult[resultType] = 'TrueMissedCall'
            
        self._model._save_data(self._dictResult)
        self._dictResult        = None
        self._dictResult_backup = None

        self._view.close()
 
        # self._model             = None
        # self._view              = None
        # self._pyEval            = None


    def _exitButton(self):
        self._view.exitButton()






