# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:25:28 2019

@author: ChiarelG
"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from skimage import io
from tkinter import Tk, filedialog
from skimage.feature import peak_local_max
import pyqtgraph.exporters

from scipy import ndimage as ndi
from scipy import optimize

from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from SMAnalyzer_Ro_GERMAN import MyPopup_histogram

class Frontend(QtGui.QFrame):

    import_image_signal = pyqtSignal(bool)
    make_histo_signal = pyqtSignal(bool)
    

    shutter1_signal = pyqtSignal(bool)
    shutter2_signal = pyqtSignal(bool)
    shutter3_signal = pyqtSignal(bool)
    flipper_signal = pyqtSignal(bool)
    flipper_notch532_signal = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        self.setUpGUI()

    def setUpGUI(self):       
 
        # Create ImageView
        self.imv = pg.ImageView()

        # Create buttons
        self.btn1 = QtGui.QPushButton('Load Image')
        self.btn2 = QtGui.QPushButton('Create ROI')
        self.btn3 = QtGui.QPushButton('Delete ROI')
        self.btn4 = QtGui.QPushButton('Get ROI mean (Traces)')
        self.btn5 = QtGui.QPushButton('Go to Video')
        self.btn6 = QtGui.QPushButton('Detect Molecules')
        self.btn7 = QtGui.QPushButton('Export Traces')

        # Can put colors in the buttons
        self.btn99_clearall = QtGui.QPushButton('Clear all')
        self.btn99_clearall.setStyleSheet(
                "QPushButton { background-color: rgb(210, 30, 100); }"
                "QPushButton:pressed { background-color: red; }")

        self.btn_images = QtGui.QPushButton('image analysis')
        self.btn_images.setStyleSheet(
                "QPushButton { background-color: rgb(200, 200, 10); }"
                "QPushButton:pressed { background-color: blue; }")
        
        self.btn_small_roi = QtGui.QPushButton('New small ROI')
        self.btn_gauss_fit = QtGui.QPushButton('Gaussian Fit')
        self.btn_filter_bg = QtGui.QPushButton('Filter bg')
        
        # labels with a fixed width
        self.gauss_fit_label = QtGui.QLabel('sigma_X / sigma_Y ><')
        self.gauss_fit_edit = QtGui.QLineEdit('1.2')
        self.gauss_fit_edit.setFixedWidth(30)
        
        self.btn_histogram = QtGui.QPushButton('Make Histogram')

        self.crazyStepButton = QtGui.QPushButton('Crazy go')
        self.crazyStepEdit = QtGui.QLineEdit('10')
        self.crazyStepEdit.setFixedWidth(30)

        # Create parameter fields with labels
        self.meanStartLabel = QtGui.QLabel('Start frame:')
        self.meanStartEdit = QtGui.QLineEdit('5')
        self.meanEndLabel = QtGui.QLabel('End frame:')
        self.meanEndEdit = QtGui.QLineEdit('15')
        self.maxDistLabel = QtGui.QLabel('Minimum distance:')
        self.maxDistEdit = QtGui.QLineEdit('6')
        self.maxThreshLabel = QtGui.QLabel('Threshold:')
        self.maxThreshEdit = QtGui.QLineEdit('0')
        self.moleculeSizeLabel = QtGui.QLabel('Size (pix):')
        self.moleculeSizeEdit = QtGui.QLineEdit('9')
        self.channelDifferenceLabel = QtGui.QLabel('Channel height difference (pixels):')
        self.channelDifferenceEdit = QtGui.QLineEdit('0')
        self.channelCorrectionLabel = QtGui.QLabel('Secondary Channel Correction:')
        self.channelCorrectionEdit = QtGui.QLineEdit('0')

        self.BgSizeLabel = QtGui.QLabel('BackGround (size + 2N)')
        self.BgSizeEdit = QtGui.QLineEdit('2')
        self.BgSizeEdit.setFixedWidth(30)

        self.time_adquisitionLabel = QtGui.QLabel('Adquisition time (ms)')
        self.time_adquisitionEdit = QtGui.QLineEdit('100')

        # Create a grid layout to manage the widgets size and position
        self.layout = QtGui.QGridLayout()


        # Add widgets to the layout in their proper positions 
        #                                       (-Y, X, Y_width ,X_width)
#        self.layout.addWidget(QtGui.QLabel(" "),       0, 0, 1, 3)
        self.layout.addWidget(self.time_adquisitionLabel,     0, 0, 1, 1)
        self.layout.addWidget(self.time_adquisitionEdit,      0, 1, 1, 2)

        self.layout.addWidget(self.btn1,               1, 0, 1, 3)
        self.layout.addWidget(self.btn2,               2, 0, 1, 3)
        self.layout.addWidget(self.btn3,               3, 0, 1, 3)

        self.layout.addWidget(self.meanStartLabel,     4, 0, 1, 1)
        self.layout.addWidget(self.meanStartEdit,      4, 1, 1, 2)
        self.layout.addWidget(self.meanEndLabel,       5, 0, 1, 1)
        self.layout.addWidget(self.meanEndEdit,        5, 1, 1, 2)


        self.layout.addWidget(self.btn4,               6, 0, 1, 1)
        self.layout.addWidget(self.btn_images,         6, 2, 1, 1)
        
        self.layout.addWidget(self.btn5,               7, 0, 1, 3)
        self.layout.addWidget(QtGui.QLabel(" "),       8, 0, 1, 3)
        
        self.layout.addWidget(self.maxDistLabel,       9, 0, 1, 1)
        self.layout.addWidget(self.maxDistEdit,        9, 1, 1, 2)
        self.layout.addWidget(self.maxThreshLabel,    10, 0, 1, 1)
        self.layout.addWidget(self.maxThreshEdit,     10, 1, 1, 2)
        self.layout.addWidget(self.moleculeSizeLabel, 11, 0, 1, 1)
        self.layout.addWidget(self.moleculeSizeEdit,  11, 1, 1, 2)
        self.layout.addWidget(self.BgSizeLabel,       12, 0, 1, 1)
        self.layout.addWidget(self.BgSizeEdit,        12, 1, 1, 2)
        
#        self.layout.addWidget(self.channelDifferenceLabel, 11, 0, 1, 1)
#        self.layout.addWidget(self.channelDifferenceEdit, 11, 1, 1, 2)
#        self.layout.addWidget(self.channelCorrectionLabel, 12, 0, 1, 1)
#        self.layout.addWidget(self.channelCorrectionEdit, 12, 1, 1, 2)

        self.layout.addWidget(self.btn6,              13, 0, 1, 3)

        self.layout.addWidget(self.btn99_clearall,    14, 2, 1, 1)

        self.layout.addWidget(self.btn7,              15, 0, 1, 3)
        self.layout.addWidget(self.imv,              0, 4, 16, 16)
        
        self.layout.addWidget(self.btn_small_roi,     2, 25, 1, 2)
        self.layout.addWidget(self.gauss_fit_label,   5, 25, 1, 1)
        self.layout.addWidget(self.gauss_fit_edit,    5, 26, 1, 1)
        self.layout.addWidget(self.btn_gauss_fit,     6, 25, 1, 2)
        self.layout.addWidget(self.btn_filter_bg,     9, 25, 1, 2)
        self.layout.addWidget(self.btn_histogram,    11, 25, 1, 2)
        self.layout.addWidget(self.crazyStepEdit,    15, 26, 1, 1)
        self.layout.addWidget(self.crazyStepButton,  15, 25, 1, 1)

        # Define a top-level widget to hold everything
        self.w = QtGui.QWidget()
        self.w.setWindowTitle('SMAnalyzer - Video')
#        self.w.resize(1500, 800)
        self.w.setLayout(self.layout)
        self.setGeometry(10, 40, 1600, 800)  # (PosX, PosY, SizeX, SizeY)

      # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)    
        grid.addWidget(self.w)

        # button actions
        self.btn1.clicked.connect(self.import_image_button)
        self.btn_histogram.clicked.connect(self.make_histo_button)

#        self.btn2.clicked.connect(self.createROI)
#        self.btn3.clicked.connect(self.deleteROI)
#        self.btn4.clicked.connect(self.ROImean)
#        self.btn5.clicked.connect(self.showVideo)
#        self.btn6.clicked.connect(self.detectMaxima)
#        self.btn7.clicked.connect(self.exportTraces_or_images)
#        
#        self.btn_images.clicked.connect(self.image_analysis)
#        self.btn_small_roi.clicked.connect(self.create_small_ROI)
#        self.btn_gauss_fit.clicked.connect(self.gaussian_fit_ROI)
#        self.btn_filter_bg.clicked.connect(self.filter_bg)
#        self.btn_histogram.clicked.connect(self.make_histogram)
#        self.crazyStepButton.clicked.connect(self.automatic_crazy_start)
#        
#        self.btn99_clearall.clicked.connect(self.clear_all)
#
#        # automatic action when you edit the number 
#        self.meanStartEdit.textEdited.connect(self.update_image)
#
#        # a Python timer that call a function with a specific clock (later)
#        self.automatic_crazytimer = QtCore.QTimer()
#        self.automatic_crazytimer.timeout.connect(self.automatic_crazy)

        


    # initialize  parameters. Remember, this is Just at start, never come here again.
        # Create empty ROI
        self.roi = None
        self.smallroi = None

        # Molecule ROI dictionary
        self.molRoi = dict()
        self.bgRoi = dict()
        self.new_roi = dict()
        self.new_roi_bg = dict()
        
        self.removerois = []

        # ROI label dictionary
        self.label = dict()
        
        # Initial number of maximums detected
        self.maxnumber = 0
        self.maxnumber_new_gauss = 0
        self.fixing_number = 0

        # Save file number
        self.n = 0

        self.is_image = False
        self.histo_data = False

    def import_image_button(self):
        self.import_image_signal.emit(True)

    def make_histo_button(self):
        self.make_histo_signal.emit(True)

#    @pyqtSlot(bool)
#    def is_JPG_function(self, JPG):
#        self.JPG = JPG
#        print("JPG", self.JPG, JPG)

    @pyqtSlot(str, bool)
    def plot_image_imported(self, file, JPG):
        print("plot image imported")

        f = file
        if JPG:  # in case I want one picture
            print("is jpg 2")
            self.data = np.mean(io.imread(f), axis=2)                
            self.axes = (0,1)  # axe 2 is the coloms of RGB
#            print("WORKING ON THIS \n","JPG =", self.JPG,)
            self.meanStartLabel.setStyleSheet(" color: red; ")
            self.meanEndLabel.setStyleSheet(" color: red; ")
            self.meanStartEdit.setStyleSheet(" background-color: red; ")
            self.meanEndEdit.setStyleSheet(" background-color: red; ")
            self.btn7.setText("Export Intensities")
            self.btn4.setStyleSheet(
                "QPushButton { background-color: rgb(10, 30, 10); }")
            self.total_size = [self.data.shape[1], self.data.shape[0]]

            self.maxDistEdit.setText("60")
            self.moleculeSizeEdit.setText("90")
            self.maxThreshEdit.setText(str(np.mean(self.data[:,:])))
            self.mean = self.data

            
        else:
            print("is tiff 2")
            # Import selected image
            self.data = io.imread(f)
            self.axes = (1,2)  # axe 0 are the frames
            self.total_size = [self.data.shape[2], self.data.shape[1]]

            self.maxDistEdit.setText("6")
            self.moleculeSizeEdit.setText("9")
            self.maxThreshEdit.setText(str(np.mean(self.data[1,:,:])))

            plot_with_colorbar(self.imv, self.data)
    
#            self.w.setWindowTitle('SMAnalyzer - Video - ' + self.f)
            self.imv.sigTimeChanged.connect(self.index_changed)
    
            self.validator = QtGui.QIntValidator(0, self.data.shape[0])
            self.meanStartEdit.setValidator(self.validator)
            self.meanEndEdit.setValidator(self.validator)

    def index_changed(self):
        """ change the numbers of start and endig frame  when move the slide"""
        self.meanStartEdit.setText(str((self.imv.currentIndex)))
        self.meanEndEdit.setText(str(int(self.imv.currentIndex)+15))


    def make_connection(self, backend):
        backend.image_to_plot_signal.connect(self.plot_image_imported)
#        backend.read_pos_signal.connect(self.read_pos_list)
#        backend.is_JPG_signal.connect(self.is_JPG_function)


class Backend(QtCore.QObject):

    image_to_plot_signal = pyqtSignal(str, bool)
#    is_JPG_signal = pyqtSignal(bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @pyqtSlot(bool)
    def import_image(self):  # connected to Load Image (btn1)
        """Select a file to analyse, can be a tif or jpg(on progres)
        the tiff data comes in a shape=(Frames, x, y) """

        JPG = False
        # Remove annoying empty window
        root = Tk()
        root.withdraw()

        # Select image from file
        f = filedialog.askopenfilename(filetypes=[("All", '*.tiff;*.tif;*.jpg'),
                                                       ("Videos", '*.tiff;*.tif'),
                                                       ("Pictures", "*.jpg")])

        if not f:
            print("No elegiste nada")
        else:
            file_path = f
            print("direccion elegida: \n", file_path, "\n")   

            # Delete existing ROIs
            self.delete_ROI()
            self.clear_all()
    
            if f[-4:] == ".jpg":  # in case I want one picture
                print("is jpg")
                JPG = True
#                data = np.mean(io.imread(f), axis=2)                
            else:
                print("is tiff")
                # Import selected image
#                data = io.imread(f)


#            self.is_JPG_signal.emit(JPG)
            print("Se manda la señal")
            self.image_to_plot_signal.emit(f, JPG)

    @pyqtSlot(bool)
    def make_histogram(self):  # connected to make histogram (btn_histogram)
        """Prepare to make the histogram with all the spots detected.
        It opens a new window to run in another thread and  make it easy.
        If the new window is not closed, it add the new data to the histogram
        and can save all of this.
        When closed, starts over, AND CANNOT SAVE ALL THIS"""
        print("make hist")
#        self.calculate_images()
        self.intensitys = np.linspace(0,10,10)
        self.intensitys2 = self.intensitys
        try:
            self.intensitys2 = np.concatenate((self.intensitys,
                                               self.intensitys2))
        except:
            self.intensitys2 = self.intensitys
        self.doit()

        self.histo_data = True
#        self.btn7.setText("Export all histogram data ({})".format(len(self.intensitys2)))

    def doit(self):  # from make_histogram
        """start the new popup window. its run independenty of the Main win"""
        self.time_adquisitionEdit = QtGui.QLineEdit('10')
        self.w2 = MyPopup_histogram(self)
        self.w2.setGeometry(QtCore.QRect(750, 50, 450, 600))
        self.w2.show()

    def delete_ROI(self):
        print("delete_ROI")


    def clear_all(self):
        print("clear_all")

    @pyqtSlot(bool)
    def shutter1(self, shutterbool):  # 642
        if shutterbool:
            openShutter(shutters[1])
        else:
            closeShutter(shutters[1])


    def make_connection(self, frontend):
        frontend.import_image_signal.connect(self.import_image)
        frontend.make_histo_signal.connect(self.make_histogram)
#        frontend.shutter1_signal.connect(self.shutter1)
#        frontend.shutter2_signal.connect(self.shutter2)
#        frontend.shutter3_signal.connect(self.shutter3)
#        frontend.flipper_signal.connect(self.power_change)
#        frontend.flipper_notch532_signal.connect(self.notch532_change)

def plot_with_colorbar(imv,data):
    print("plot with colorbar start")
    # Display the data and assign each frame a number
    x = np.linspace(1., data.shape[0], data.shape[0])

    # Load array as an image
    imv.setImage(data, xvals=x)

    # Set a custom color map
    colors = [
            (0, 0, 0),
            (45, 5, 61),
            (84, 42, 55),
            (150, 87, 60),
            (208, 171, 141),
            (255, 255, 255)
            ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    imv.setColorMap(cmap)

if __name__ == '__main__':

    app = QtGui.QApplication([])

    gui = Frontend()   
    worker = Backend()

    worker.make_connection(gui)
    gui.make_connection(worker)

    shuttersThread = QtCore.QThread()
    worker.moveToThread(shuttersThread)
    shuttersThread.start()

    gui.show()
    app.exec_()
                