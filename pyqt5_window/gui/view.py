



import sys

# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget

#####

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QStackedLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui     import QPixmap





## Create a subclass of QMainWindow to setup the calculator's GUI
class class_view(QMainWindow):
    """PyCalc's View (GUI)."""
    def __init__(self, query_path, imgs_path, numShowImgs, beta, eta, tau, title=None):
        """View initializer."""
        # super().__init__()
        # super(class_view,self).__init__()
        QMainWindow.__init__(self)

# class class_view(QWidget):
#     def __init__(self, query_path, imgs_path, numShowImgs, beta, eta, tau, title=None):
#         QWidget.__init__(self, parent=None)

        self.buttons     = None
        self.buttonSave  = None
        self.buttonFail  = None
        self.buttonClear = None
        self._title        = title
        self._beta         = beta
        self._eta          = eta
        self._tau          = tau
        self._qPath        = query_path
        self._imgsPath     = imgs_path
        self._numShowImags = numShowImgs
        
        try:   
            self.setWindowTitle('Re-ID Alert')
            # self.width_win  = 635
            # self.height_win = 635
            # self.setFixedSize(self.width_win , self.height_win)
            self._centralWidget = QWidget()
            # self._generalLayout = QGridLayout(self)
            self._generalLayout = QVBoxLayout()
            # self._generalLayout = QStackedLayout(self)
            # self._layout.addLayout(self._generalLayout,0,0)

            # Set the central widget
            # set the central widget and the general layout
            # self.setLayout(self._generalLayout,0,0)
            self._createTitle()
            self._createImages()
            # self._createButtons()
            self._createButtonFail()
            self._createButtonClear()
            self._createButtonSaveExit()
            
            # ## QWidget
            # self.setLayout(self._generalLayout)
            # QMainWindow
            self._centralWidget.setLayout(self._generalLayout)
            self.setCentralWidget(self._centralWidget)
            
            # _centralWidget = QWidget()
            # _centralWidget.setLayout(self._generalLayout)
            # self.setCentralWidget(_centralWidget)
            
        except :
            print("ERROR")
            # breakpoint()
        # Set some main window's properties
      
        # # show all the widgets
        # self.update()
        # self.show()

        
    def _createTitle(self):

        imagesLayout = QGridLayout()
        title = QLabel()
        title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("border: 1px solid black;")
        title.setStyleSheet("color : red;")
        title.setText('Beta = {:.3f}  --   Tau = {:04d}  --   Eta = {:04d}'.format(self._beta,  self._tau, self._eta,))
        # title.setFixedSize(50, 10)
        imagesLayout.addWidget(title, 0, 0)  

        title2 = QLabel()
        title2.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("border: 1px solid black;")
        title2.setStyleSheet("color : Green;")
        title2.setText(self._title)
        # title.setFixedSize(50, 10)
        imagesLayout.addWidget(title2, 1, 0)        

        self._generalLayout.addLayout(imagesLayout)



    def _createImages(self):
        # assert len(self._imgsPath)<=self._eta, 'error, you need len(imgs_path)== top_rank, {}<={}'.format(len(self._imgsPath),self._eta)
        
        imagesLayout      = QGridLayout()
        query             = QLabel()
        query.setAlignment(Qt.AlignCenter)
        query.setStyleSheet("border: 3px solid green;")
        image             = QPixmap(self._qPath)
        # image             = image.scaled(128, 256, Qt.KeepAspectRatio)
        # image = image.scaled(128, 256)
        # image = image.scaled(64, 128, Qt.KeepAspectRatio)
        query.setPixmap(image)
        imagesLayout.addWidget(query, 0, 0)        
        # query.setText('My Label')
        # query.setFixedSize(50, 10)

        for i, _imgPath in enumerate(self._imgsPath[:self._numShowImags], start=1):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            image = QPixmap(_imgPath)
            # image = image.scaled(128, 256, Qt.KeepAspectRatio)
            # image = image.scaled(128, 256)
            # image = image.scaled(64, 128, Qt.KeepAspectRatio)
            label.setPixmap(image)
            imagesLayout.addWidget(label, 0, i)


        # assert len(self._imgsPath)<=self._eta, 'error, you need len(imgs_path)<= top_rank, {}=={}'.format(len(self._imgsPath),self._eta)
        """Create the buttons."""
        self.buttons    = dict()
        # buttonsLayout = QGridLayout()
        # buttonsLayout   = self.imagesLayout
        # Button text | position on the QGridLayout
    
        text_buttons         = {'Query': (1,0)}
        # buttons['Rank_{:02d}'.format(1)] = (1,1)
        for i in range(1, len(self._imgsPath[:self._numShowImags])+1):
        # for i in range(1, len(self._imgsPath[:self._numShowImags])+1):
            # buttons['Rank_{:02d}'.format(i)] = (1,i)
            text_buttons['Rank_{:02d}'.format(i)] = (1,i)
            # buttons['Rank_{}'.format(i)] = (1,i)

        # Create the buttons and add them to the grid layout
        for btnText, pos in text_buttons.items():
            self.buttons[btnText] = QPushButton(btnText)
            # self.buttons[btnText].setFixedSize(80, 40)
            self.buttons[btnText].setCheckable(True)
            # self.buttons[btnText].toggle()
            self.buttons[btnText].setStyleSheet("background-color : lightgrey")

            if btnText == 'Query':
                self.buttons[btnText].setEnabled(False)
            imagesLayout.addWidget(self.buttons[btnText], pos[0], pos[1])
        # Add buttonsLayout to the general layout


        self._generalLayout.addLayout(imagesLayout)





    # def _createButtons(self):
    #     # assert len(self._imgsPath)<=self._eta, 'error, you need len(imgs_path)<= top_rank, {}=={}'.format(len(self._imgsPath),self._eta)
    #     """Create the buttons."""
    #     self.buttons    = dict()
    #     buttonsLayout = QGridLayout()
    #     # buttonsLayout   = self.imagesLayout
    #     # Button text | position on the QGridLayout
    
    #     buttons         = {'Query': (0,0)}
    #     for i in range(1, len(self._imgsPath[:self._numShowImags])+1):
    #         # buttons['Rank_{:02d}'.format(i)] = (1,i)
    #         buttons['Rank_{:02d}'.format(i)] = (0,i)

    #     # Create the buttons and add them to the grid layout
    #     for btnText, pos in buttons.items():
    #         self.buttons[btnText] = QPushButton(btnText)
    #         # self.buttons[btnText].setFixedSize(80, 40)
    #         self.buttons[btnText].setCheckable(True)
    #         # self.buttons[btnText].toggle()
    #         self.buttons[btnText].setStyleSheet("background-color : lightgrey")

    #         if btnText == 'Query':
    #             self.buttons[btnText].setEnabled(False)
    #         buttonsLayout.addWidget(self.buttons[btnText], pos[0], pos[1])
    #     # Add buttonsLayout to the general layout
    #     self._generalLayout.addLayout(buttonsLayout)




    def _createButtonFail(self):
        """Create the buttons."""
        # self.buttonFail = self.buttons
        self.buttonFail          = dict()
        buttonsLayout            = QGridLayout()
        # Button text | position on the QGridLayout
        btnText                  = 'fail'
        self.buttonFail[btnText] = QPushButton('Re-identification &Fail')
        self.buttonFail[btnText].setCheckable(True)
        # self.buttonFail[btnText].toggle()
        self.buttonFail[btnText].setStyleSheet("background-color : lightgrey")

        self.buttonFail[btnText].setEnabled(True)
        buttonsLayout.addWidget(self.buttonFail[btnText], 0, 0)
        self._generalLayout.addLayout(buttonsLayout)



    def _createButtonClear(self):
        """Create the buttons."""
        # self.buttonClear = self.buttons
        self.buttonClear          = dict()
        buttonsLayout             = QGridLayout()
        # Button text | position on the QGridLayout
        btnText                   = 'clear'
        self.buttonClear[btnText] = QPushButton('&Clear_all')
        self.buttonClear[btnText].setStyleSheet("background-color : lightgrey")
        self.buttonClear[btnText].setEnabled(False)
        # self.buttonClear[btnText].setCheckable(True)
        # self.buttonFail[btnText].toggle()


        buttonsLayout.addWidget(self.buttonClear[btnText], 0, 0)
        self._generalLayout.addLayout(buttonsLayout)




    def _createButtonSaveExit(self):
        """Create the buttons."""
        self.buttonSave          = dict()
        buttonsLayout            = QGridLayout()
        # Button text | position on the QGridLayout
        tmp                      = QLabel()
        btnText                  = 'save'
        self.buttonSave[btnText] = QPushButton('&Save and exit')
        self.buttonSave[btnText].setEnabled(False)
        buttonsLayout.addWidget(tmp, 0, 0)
        buttonsLayout.addWidget(tmp, 1, 0)
        buttonsLayout.addWidget(tmp, 2, 0)
        buttonsLayout.addWidget(self.buttonSave[btnText], 4, 0)

        self._generalLayout.addLayout(buttonsLayout)



    def exitButton(self):
        for _btnText,_ in self.buttonSave.items(): 
            self.buttonSave[_btnText].setEnabled(False)

