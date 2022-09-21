



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
    def __init__(self, query_path, imgs_path, numShowImgs, beta, eta, tau, title=None, addtext=None):
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
        self._addtext      = addtext
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
        
        title0 = QLabel()
        title0.setAlignment(Qt.AlignCenter)
        # title0.setStyleSheet("border: 1px solid black;")
        title0.setStyleSheet("color : red;")
        title0.setText('Beta = {:.3f}  --   Tau = {:04d}  --   Eta = {:04d}'.format(self._beta,  self._tau, self._eta,))
        title0.setTextInteractionFlags(Qt.TextSelectableByMouse);
        # title0.setFixedSize(50, 10)
        imagesLayout.addWidget(title0, 0, 0)  

        title = QLabel()
        title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("border: 1px solid black;")
        title.setStyleSheet("color : black;")
        title.setText(self._addtext)
        title.setTextInteractionFlags(Qt.TextSelectableByMouse);

        # title.setFixedSize(50, 10)
        imagesLayout.addWidget(title, 1, 0)  

        title2 = QLabel()
        title2.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet("border: 1px solid black;")
        title2.setStyleSheet("color : Green;")
        title2.setText(self._title)
        title2.setTextInteractionFlags(Qt.TextSelectableByMouse);
        # title.setFixedSize(50, 10)
        imagesLayout.addWidget(title2, 2, 0)        

        self._generalLayout.addLayout(imagesLayout)



    def _createImages(self):
      
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


        ### text_buttons = {'Query': (1,0), 'TrueCa&ll':(1,1)}

        btnQueryText = 'Query'
        self.buttonQuery = QPushButton(btnQueryText)
        self.buttonQuery.setCheckable(True)
        self.buttonQuery.setStyleSheet("background-color : lightgrey")
        self.buttonQuery.setEnabled(False)
        imagesLayout.addWidget(self.buttonQuery, 1, 0)
        

        # Create the buttons and add them to the grid layout
        btnText = 'TrueC&all'
        self.buttonTrueCall = QPushButton(btnText)
        self.buttonTrueCall.setCheckable(True)
        self.buttonTrueCall.setStyleSheet("background-color : lightgrey")
        imagesLayout.addWidget(self.buttonTrueCall, 1, 1, 1, self._numShowImags)


        # Add buttonsLayout to the general layout
        self._generalLayout.addLayout(imagesLayout)




    def _createButtonFail(self):
        """Create the buttons."""
        buttonsLayout            = QGridLayout()
        # Button text | position on the QGridLayout
        btnFailText              = 'Re-identification &Fail'
        self.buttonFail = QPushButton(btnFailText)
        self.buttonFail.setCheckable(True)
        # self.buttonFail[btnText].toggle()
        self.buttonFail.setStyleSheet("background-color : lightgrey")

        self.buttonFail.setEnabled(True)
        buttonsLayout.addWidget(self.buttonFail, 0, 0)
        self._generalLayout.addLayout(buttonsLayout)


