"""
@author: Rodrigo, the one who never doment his code
@Remastered by: German, the non-serious commenter


User interface to detect particles in a movie
and extract its traces in a .txt file
First, A custom sized Region Of Interest(ROI) of the hole image have to be created.
-choose a start and end frame to average frames within that ROI 
and have a sharper image where to detect the particles
-"get ROI mean" returns the image of the averaged ROI, where
then the maximums will be detected
.
For that you need to put (try by hand for the best ones):
- minimum distance between the max it will gonna find.
- threshold if you want. Only detect numbers grater that's this.
automaticaly change to the mean of the first frame when the file is loaded
-Size: the estimated size of your psf,
 Actually you have to put a good size to don't loose any photon.
- Detect Molecules: search all the local maximums in the ROI
averaged, at distances greater than minimun distance. Draw a
square size*size on each point detected.
subtract the calculated bakground from a square 1 pixel more than the periphery
(size + 1)*(size + 1) - size*size. Normalized by size

-Export Traces: Save the .txt file with 1 column per particle.

- Choose a Minimun distance between maximum
- Threshold if necessary. Only detect maximums greater than this
- Size: the size of your PSF


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

import time as time

class smAnalyzer(pg.Qt.QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Define a top-level widget to hold everything
        self.w = QtGui.QWidget()
        self.w.setWindowTitle('SMAnalyzer - Video')
        self.w.resize(1500, 800)

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

        self.remove_new_Button = QtGui.QPushButton('Remove added new rois')

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

        self.gauss_fit_label = QtGui.QLabel('threshold to sigma:')
        self.gauss_fit_edit = QtGui.QLineEdit('1.5')
        self.gauss_fit_edit.setFixedWidth(30)
        
        self.btn_histogram = QtGui.QPushButton('Make instogram')

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

        # Create a grid layout to manage the widgets size and position
        self.layout = QtGui.QGridLayout()
        self.w.setLayout(self.layout)

        # Add widgets to the layout in their proper positions
        self.layout.addWidget(QtGui.QLabel(" "),       0, 0, 1, 3)
        self.layout.addWidget(self.btn1,               1, 0, 1, 3)
        self.layout.addWidget(self.btn2,               2, 0, 1, 3)
        self.layout.addWidget(self.btn3,               3, 0, 1, 3)

        self.layout.addWidget(self.meanStartLabel,     4, 0, 1, 1)
        self.layout.addWidget(self.meanStartEdit,      4, 1, 1, 2)
        self.layout.addWidget(self.meanEndLabel,       5, 0, 1, 1)
        self.layout.addWidget(self.meanEndEdit,        5, 1, 1, 2)


        self.layout.addWidget(self.btn4,               6, 0, 1, 1)
        self.layout.addWidget(self.btn_images,         6, 2, 1, 1)
        

        self.layout.addWidget(QtGui.QLabel(" "),       7, 0, 1, 3)
        self.layout.addWidget(self.btn5,               8, 0, 1, 3)
#        self.layout.addWidget(QtGui.QLabel(" "),       9, 0, 1, 3)
        
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
        
        self.layout.addWidget(self.btn_small_roi,     2, 25, 1, 1)
        self.layout.addWidget(self.remove_new_Button, 3, 25, 1, 1)
        self.layout.addWidget(self.gauss_fit_label,   5, 25, 1, 1)
        self.layout.addWidget(self.gauss_fit_edit,    6, 25, 1, 1)
        self.layout.addWidget(self.btn_gauss_fit,     7, 25, 1, 1)
        self.layout.addWidget(self.btn_filter_bg,     9, 25, 1, 1)
        self.layout.addWidget(self.btn_histogram,    11, 25, 1, 1)
        self.layout.addWidget(self.crazyStepEdit,    13, 25, 1, 1)
        self.layout.addWidget(self.crazyStepButton,  14, 25, 1, 1)

        # button actions
        self.btn1.clicked.connect(self.importImage)
        self.btn2.clicked.connect(self.createROI)
        self.btn3.clicked.connect(self.deleteROI)
        self.btn4.clicked.connect(self.ROImean)
        self.btn5.clicked.connect(self.showVideo)
        self.btn6.clicked.connect(self.detectMaxima)
        self.btn7.clicked.connect(self.exportTraces_or_images)
        
        self.btn_images.clicked.connect(self.image_analysis)
        self.btn_small_roi.clicked.connect(self.create_small_ROI)
        self.btn_gauss_fit.clicked.connect(self.gaussian_fit_ROI)
        self.btn_filter_bg.clicked.connect(self.filter_bg)
        self.btn_histogram.clicked.connect(self.make_histogram)
        self.crazyStepButton.clicked.connect(self.automatic_crazy_start)
        
        self.remove_new_Button.clicked.connect(self.remove_all_new)
        self.btn99_clearall.clicked.connect(self.deleteMaxima)

        self.meanStartEdit.textEdited.connect(self.update_image)

        self.automatic_crazytimer = QtCore.QTimer()
        self.automatic_crazytimer.timeout.connect(self.automatic_crazy)
        
        


        # Create empty ROI
        self.roi = None
        self.smallroi = None


        # Molecule ROI dictionary
        self.molRoi = dict()
        self.bgRoi = dict()
        self.new_roi = dict()
        self.new_roi_bg = dict()
        
        self.removerois = []
        self.removed_new_rois = []  # OJO CON TODO ESTO QUE NO SE REINICIA

        # ROI label dictionary
        self.label = dict()
        self.new_label = dict()
        
        # Initial number of maximums detected
        self.maxnumber = 0
        self.maxnumber_new = 0
        self.fixing_number = 0

        # Save file number
        self.n = 0

        self.JPG = False
        self.image_analysis = False
        self.histo_data = False

        self.new_i = 0

    def update_image(self):
        self.imv.setCurrentIndex(int(self.meanStartEdit.text()))

    def exportTraces_or_images(self):
        if self.image_analysis:
            
            self.calculate_images()
            self.export("images")
        else:
            
            self.calculate_traces()
            self.export("trace")

    def image_analysis(self):
        self.image_analysis = True
        if self.roi == None:
            self.start = int(self.meanStartEdit.text())
            self.mean = self.data[self.imv.currentIndex,:,:]
        else:
            self.ROI_no_mean_images()

        self.btn7.setText("Export Intensities from frame={}".format(self.start))
        self.btn7.setStyleSheet(
                "QPushButton { background-color: rgb(200, 200, 10); }")
        self.meanEndEdit.setStyleSheet(" background-color: red; ")
        print("change colores")

    def small_ROI_to_new_ROI(self):
        print("\n YOU CLICK MEE 0_o \n")
        self.roiSize = [int(self.moleculeSizeEdit.text())] * 2
        self.bgroiSize = np.array(self.roiSize) + 2* int(self.BgSizeEdit.text())
        center = int(self.BgSizeEdit.text()) * np.array([1, 1])

#        i = self.new_i
#        self.new_roi[i] = pg.ROI(self.smallroi.pos(), self.roiSize,
#                                                       scaleSnap=True,
#                                                       translateSnap=True,
#                                                       movable=False,
#                                                       removable=True,
#                                                       pen='y') 
#        self.new_roi_bg[i] = pg.ROI((self.smallroi.pos() - center), self.bgroiSize,
#                                                          scaleSnap=True,
#                                                          translateSnap=True,
#                                                          movable=False,
#                                                          removable=True,
#                                                          pen='y') 
#        self.imv.view.addItem(self.new_roi[i])
#        self.imv.view.addItem(self.new_roi_bg[i])
#
#        self.new_roi[i].sigRemoveRequested.connect(self.remove_new_ROI)
#        self.new_roi_bg[i].sigRemoveRequested.connect(self.remove_new_ROI)
#
#        self.new_label[i] = pg.TextItem(text="new_"+str(i))
#        self.new_label[i].setPos(self.new_roi[i].pos())
#        self.imv.view.addItem(self.new_label[i])
#        
#        self.new_i = i + 1
#        self.relabel_new_ROI()
        continue_number = self.fixing_number
        i = continue_number
        self.molRoi[i,0] = pg.ROI(self.smallroi.pos(), self.roiSize,
                                                       scaleSnap=True,
                                                       translateSnap=True,
                                                       movable=False,
                                                       removable=True,
                                                       pen='y') 
        self.bgRoi[i,0] = pg.ROI((self.smallroi.pos() - center), self.bgroiSize,
                                                      scaleSnap=True,
                                                      translateSnap=True,
                                                      movable=False,
                                                      removable=True,
                                                      pen='y') 
        self.imv.view.addItem(self.molRoi[i,0])
        self.imv.view.addItem(self.bgRoi[i,0])

        self.molRoi[i,0].sigRemoveRequested.connect(self.remove_ROI)
        self.bgRoi[i,0].sigRemoveRequested.connect(self.remove_ROI)

        self.label[i,0] = pg.TextItem(text=str(i))
        self.label[i,1] = pg.TextItem(text=str(i))
        self.label[i,0].setPos(self.molRoi[i,0].pos())
        self.imv.view.addItem(self.label[i,0])

        self.fixing_number = i + 1
        print("finxing", self.fixing_number)
        self.relabel_new_ROI()

    def remove_all_new(self):
        print("new_i",self.new_i)
        for i in range(self.new_i):
            self.imv.view.removeItem(self.new_roi[i])
            self.imv.view.removeItem(self.new_roi_bg[i])
            self.imv.view.removeItem(self.new_label[i])
            del self.new_roi[i]
            del self.new_roi_bg[i]
            del self.new_label[i]
    
        self.new_roi = dict()
        self.new_roi_bg = dict()
        self.new_label = dict()
        self.new_i = 0

    def remove_new_ROI(self,evt):
        print("Remove_NEW_roi")
        for i in range(len(self.new_roi)):
            if self.new_roi[i] == evt or self.new_roi_bg[i] == evt:
                index = i
                self.removed_new_rois.append(index)
        
        self.imv.view.scene().removeItem(self.new_roi[index])
        self.imv.view.scene().removeItem(self.new_roi_bg[index])
        self.imv.view.scene().removeItem(self.new_label[index])


        self.relabel_new_ROI()

    def create_small_ROI(self):
        
        if self.smallroi is not None:
            self.meanEndEdit.setStyleSheet(" background-color: ; ")
            self.imv.view.scene().removeItem(self.smallroi)
            self.smallroi = None
            print("good bye old roi, Hello new Roi")

        try:
            roisize = int(self.moleculeSizeEdit.text())
            self.smallroi = pg.ROI([0, 0], [roisize, roisize],
                                   scaleSnap=True, translateSnap=True,
                                   movable=True, removable=True, pen='g')
            self.imv.view.addItem(self.smallroi)
            self.smallroi.sigRemoveRequested.connect(self.remove_small_ROI)
            self.smallroi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            self.smallroi.sigClicked.connect(self.small_ROI_to_new_ROI)
            
        except:
            pass

    def createROI(self):
        if self.roi is None:
            self.roi = pg.ROI([0, 0], [self.data.shape[2], self.data.shape[1]],
                              scaleSnap=True, translateSnap=True)  # [70, 70]
            self.roi.addScaleHandle([1, 1], [0, 0])
            self.roi.addScaleHandle([0, 0], [1, 1])
            self.imv.view.addItem(self.roi)
        else:
            pass

    def deleteROI(self):
        if self.roi is None:
            pass
        else:
            self.imv.view.removeItem(self.roi)
            self.roi = None

    def importImage(self):

        # Remove annoying empty window
        root = Tk()
        root.withdraw()

        # Select image from file
        self.f = filedialog.askopenfilename(filetypes=[("Videos", '*.tiff;*.tif;*.jpg'),
                                                       ("Pictures", "*.jpg")])
        if self.f[-4:] == ".jpg":  # in case I want one picture

            self.JPG = True
            self.axes = (0,1)  # axe 2 are the 3 coloms of RGB
            print("JPG =", self.JPG)
            self.data = np.mean(io.imread(self.f),axis=2)
            self.meanStartLabel.setStyleSheet(" color: red; ")
            self.meanEndLabel.setStyleSheet(" color: red; ")
            self.meanStartEdit.setStyleSheet(" background-color: red; ")
            self.meanEndEdit.setStyleSheet(" background-color: red; ")
            self.btn7.setText("Export Intensities")
            
        else:
            # Import selected image
            self.data = io.imread(self.f)
            self.axes = (1,2)  # axe 1 are the frames

        
        # Delete existing ROIs
        self.deleteROI()
        self.deleteMaxima()



        # Display the data and assign each frame a number
        x = np.linspace(1., self.data.shape[0], self.data.shape[0])

        # Load array as an image
        self.imv.setImage(self.data, xvals=x)

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
        self.imv.setColorMap(cmap)
        self.w.setWindowTitle('SMAnalyzer - Video - ' + self.f)
        self.imv.sigTimeChanged.connect(self.indexChanged)

        self.validator = QtGui.QIntValidator(0, self.data.shape[0])
        self.meanStartEdit.setValidator(self.validator)
        self.meanEndEdit.setValidator(self.validator)
        self.maxThreshEdit.setText(str(np.mean(self.data[0,:,:])))

    def indexChanged(self):
        self.meanStartEdit.setText(str((self.imv.currentIndex)))
        self.meanEndEdit.setText(str(int(self.imv.currentIndex)+15))


    def ROI_no_mean_images(self):
        z = self.roi.getArrayRegion(self.data, self.imv.imageItem, axes=self.axes)
        if  self.JPG:
            self.mean = z 
        else:
            self.start = int(self.meanStartEdit.text())
            self.mean = z[self.start,:,:]

    def ROImean(self):
        self.image_analysis = False

        self.btn7.setText("Export Traces")
        self.btn7.setStyleSheet(
                "QPushButton { background-color: ; }")
        self.meanEndEdit.setStyleSheet(" background-color: ; ")
        print("change colors back to normal")

        z = self.roi.getArrayRegion(self.data, self.imv.imageItem, axes=self.axes)

        self.start = int(self.meanStartEdit.text())
        self.end = int(self.meanEndEdit.text())
        print("z", z[self.start:self.start+self.end, :, :].shape)
        z = z[self.start:self.start+self.end, :, :]
            
        self.mean = np.mean(z, axis=0)
        print("mean",self.mean.shape)

        # Display the data and assign each frame a number
        x = np.linspace(1., self.data.shape[0], self.data.shape[0])


        # Load Mean Image
        self.imv.setImage(self.mean, xvals=x)

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
        self.imv.setColorMap(cmap)
        self.w.setWindowTitle('SMAnalyzer - ROI Mean - ' + self.f)
        self.imv.view.removeItem(self.roi)

    def showVideo(self):
        
        # Display the data and assign each frame a number
        x = np.linspace(1., self.data.shape[0], self.data.shape[0])

        # Load array as an image
        self.imv.setImage(self.data, xvals=x)

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
        self.imv.setColorMap(cmap)
        self.w.setWindowTitle('SMAnalyzer - Video - ' + self.f)
        self.meanEndEdit.setStyleSheet(" background-color: ; ")

        try:
            self.translateMaxima()
            self.imv.view.addItem(self.roi)
        except:
            pass

    def detectMaxima(self):
#        self.deleteMaxima()
        self.dist = int(self.maxDistEdit.text())
        self.threshold = float(self.maxThreshEdit.text())
        
        # set roi Dimension array
        self.roiSize = [int(self.moleculeSizeEdit.text())] * 2
        self.bgroiSize = np.array(self.roiSize) + 2* int(self.BgSizeEdit.text())  # one pixel each side
        center = int(self.BgSizeEdit.text()) * np.array([1, 1])
        if self.roi == None:
            if not self.image_analysis:
                self.mean = self.data[self.imv.currentIndex,:,:]

        self.maximacoord = peak_local_max(self.mean, min_distance=self.dist, threshold_abs=self.threshold)


        maxvalues = []
        for i in range(len(self.maximacoord[:,0])):
            maxvalues.append(self.mean[self.maximacoord[i,0],self.maximacoord[i,1]])
        
        nomaxlow = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]
        
        aux = np.arange(len(maxvalues))
        goodmax = np.delete(aux,nomaxlow)
        
#        nomaxhigh = np.where(np.array(maxvalues) > 1.5*np.mean(np.array(maxvalues)[goodmax]))
#        toerase = np.sort(np.append(nomaxlow, nomaxhigh))
        maxindex = goodmax  # np.delete(aux,toerase)   NOT Nice for now

        print(len(goodmax), "points finded")

        self.maxnumber = np.size(self.maximacoord[maxindex], 0)
#        print("finxing", self.fixing_number)
#        print("arange + fixing", np.arange(0,self.maxnumber)+self.fixing_number)
        p = 0
        for i in np.arange(0, self.maxnumber)+self.fixing_number:
    
            # Translates molRoi to particle center
            corrMaxima = np.flip(self.maximacoord[maxindex[p]], 0) - 0.5*np.array(self.roiSize) + [0.5, 0.5]
            self.molRoi[i,0] = pg.ROI(corrMaxima, self.roiSize,
                                                           scaleSnap=True,
                                                           translateSnap=True,
                                                           movable=False,
                                                           removable=True)
            self.bgRoi[i,0] = pg.ROI((corrMaxima - center), self.bgroiSize,
                                                          scaleSnap=True,
                                                          translateSnap=True,
                                                          movable=False,
                                                          removable=True)
#            self.molRoi[i,1] = pg.ROI(corrMaxima - [0, int(self.channelDifferenceEdit.text())], self.roiSize, scaleSnap=True, translateSnap=True, movable=False, removable=True)
#            self.bgRoi[i,1] = pg.ROI(corrMaxima - [1, 1] - [0, int(self.channelDifferenceEdit.text())], self.bgroiSize, scaleSnap=True, translateSnap=True, movable=False, removable=True)
            self.imv.view.addItem(self.molRoi[i,0])
#            self.imv.view.addItem(self.molRoi[i,1])
            self.imv.view.addItem(self.bgRoi[i,0])
#            self.imv.view.addItem(self.bgRoi[i,1])


            self.molRoi[i,0].sigRemoveRequested.connect(self.remove_ROI)
            self.bgRoi[i,0].sigRemoveRequested.connect(self.remove_ROI)
#            self.molRoi[i,1].sigRemoveRequested.connect(self.remove_ROI)
#            self.bgRoi[i,1].sigRemoveRequested.connect(self.remove_ROI)

            # Create ROI label
            self.label[i,0] = pg.TextItem(text=str(i))
            self.label[i,1] = pg.TextItem(text=str(i))
            self.label[i,0].setPos(self.molRoi[i,0].pos())
#            self.label[i,1].setPos(self.molRoi[i,1].pos())
            self.imv.view.addItem(self.label[i,0])
#            self.imv.view.addItem(self.label[i,1])
            p+=1
#        self.Nparticles = self.maxnumber
        self.fixing_number = i + 1
#        print("finxing", self.fixing_number)
        self.relabel_new_ROI()
        if not self.image_analysis:
            self.btn7.setText("Intensities from frame={}".format(int(self.meanStartEdit.text())))


    def filter_bg(self):
#        self.suma = dict()
        molArray = dict()
        bgArray = dict()
        bg = dict()
        bgNorm = dict()
        bgArray = dict()
        j = 0
        p = 0
#        suma = []
        a = 0
        bgsize = int(self.BgSizeEdit.text())
        for i in range(len(self.molRoi)): #np.arange(0, self.maxnumber):
            if i not in self.removerois:

                # get molecule array
                molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.mean, self.imv.imageItem)

                # get background plus molecule array
                bgArray[i,j] = self.bgRoi[i,j].getArrayRegion(self.mean, self.imv.imageItem)

                # get background array
                bg[i,j] = np.sum(bgArray[i,j]) - np.sum(molArray[i,j])

                # get total background to substract from molecule traces
                bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(bg[i,j])/(((2* int(self.BgSizeEdit.text()))**2)*(int(self.moleculeSizeEdit.text())+1))

#                suma.append(bgNorm[i,j])  # np.sum(molArray[i,j]) - 

                p +=1 # I have to use this to have order because of removerois

#        a = 0
#        for i in range(len(suma)):
#            if suma[i] > np.mean(suma):
#                self.bgRoi[i,j].setPen('r')
#                a+=1
                b = True
#                print("removerois",self.removerois)
                for l in np.arange(-bgsize,bgsize):
                    if b:
    #                    print("l", l)
    #                    print("mean", np.mean(bgArray[i,j][:,l]))
    #                    print("threshold", float(self.maxThreshEdit.text()))
                        if np.mean(bgArray[i,j][:,l]) > float(self.maxThreshEdit.text()) or np.mean(bgArray[i,j][l,:]) > float(self.maxThreshEdit.text()):
                            print("bad #", a)
                            b = False
                            self.bgRoi[i,j].setPen('r')
                            self.removerois.append(i)
                            a+=1
#                        if np.mean(bgArray[i,j][l,:]) > float(self.maxThreshEdit.text()):
#                            print("casi 2")
#                            b = False
#                            self.bgRoi[i,j].setPen('r')
#                            a+=1
#    #                        self.removerois.append(i)
    
        print("badParticles/total=", a,"/", self.maxnumber-len(self.removerois))

    def gaussian_fit_ROI(self):
        self.remove_gauss_ROI()

        molArray = dict()
        self.gauss_roi = dict()
        j=0
        i=1
        print("start gaussian")
        print(len(self.molRoi))
        for i in range(len(self.molRoi)):
            if i not in self.removerois:
                molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.mean, self.imv.imageItem)
                print("i= ", i)
                data = np.transpose(molArray[i,j])
    #            params = fitgaussian(data)
    #            fit = gaussian(*params)
                try:
                    new_params = fitgaussian(data) #molArray[i,j])
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    print("who knows \o/")  
        #        all_params[j] = new_params
                (height, x, y, width_x, width_y) = new_params
                print("\n new_params \n",
                                         "[amplitude, x, y, Sigma_x, sigma_y] \n",
                                         new_params)
                print(self.roiSize, np.round(x),np.round(y))
                newx = x-self.roiSize[0]//2 + 0.5
                newy = y-self.roiSize[1]//2 + 0.5
                print(self.molRoi[i,j].pos())
                originx =  self.molRoi[i,j].pos()[0]
                originy =  self.molRoi[i,j].pos()[1]

                self.gauss_roi[i] = pg.ROI([originx,originy], self.roiSize, pen=(100,50,200,200),
                                                               scaleSnap=True,
                                                               translateSnap=True,
                                                               movable=False,
                                                               removable=False)

#                self.molRoi[i,j].setPos((originx+newx, originy+newy))
                self.molRoi[i,j].setPen('m')
                self.molRoi[i,j].translate([newx, newy])
                self.bgRoi[i,j].translate([newx, newy])

                self.imv.view.addItem(self.gauss_roi[i])
                print("Created new roi",i, "to", [newy, newx],"\n")
                threshold_sigma = float(self.gauss_fit_edit.text())
                if width_x > threshold_sigma*width_y or width_y > threshold_sigma*width_x:
#                    self.gauss_roi[i].setPen('r')
                    self.molRoi[i,j].setPen('r')
                    self.removerois.append(i)

        self.maxnumber_new = len(self.molRoi)

    def remove_gauss_ROI(self):
        for i in range(self.maxnumber_new):
            try:
                self.imv.view.removeItem(self.gauss_roi[i])
                del self.gauss_roi[i]
            except:
                pass

    def remove_ROI(self,evt):
        print("Remove_ROI")
        for i in np.arange(0, self.fixing_number):
            if self.bgRoi[i,0] == evt or self.molRoi[i,0] == evt:
#                print("Removed Roi",i)
                index = i
                self.removerois.append(index)
        
#        self.imv.view.scene().removeItem(evt)
        self.imv.view.scene().removeItem(self.molRoi[index,0])
        self.imv.view.scene().removeItem(self.bgRoi[index,0])
        self.imv.view.scene().removeItem(self.label[index,0])
#        print("\n")
#        print(self.molRoi[0,0],"molRoi0")
#        print(self.bgRoi[0,0],"bgRoi0")

        self.relabel_new_ROI()

    def relabel_ROI(self):
        p = 0
        for i in np.arange(0, self.fixing_number):
            if i not in self.removerois:
#                print("i",i)
#                self.imv.view.removeItem(self.label[i,0])
#                self.imv.view.removeItem(self.label[i,1])
#                self.label[i,0] = pg.TextItem(text=str(p))
#                self.label[i,0].setPos(self.molRoi[i,0].pos())
#                self.imv.view.addItem(self.label[i,0])
                self.label[i,0].setText(text=str(p))
                p+=1
        self.Nparticles = p-1

    def relabel_new_ROI(self):
        self.relabel_ROI()

        for i in self.removerois:
            self.imv.view.removeItem(self.molRoi[i,0])
            self.imv.view.removeItem(self.bgRoi[i,0])
            self.imv.view.removeItem(self.label[i,0])
            self.imv.view.removeItem(self.label[i,1])

#        p = self.Nparticles + 1
#        for i in range(len(self.new_roi)):
#            if i not in self.removed_new_rois:
##                print(i,self.removerois)
#                self.imv.view.removeItem(self.new_label[i])
#                self.new_label[i] = pg.TextItem(text=str(p))
#                self.new_label[i].setPos(self.new_roi[i].pos())
#                self.imv.view.addItem(self.new_label[i])
#                p+=1
#        self.N_newparticles = p-1

    def remove_small_ROI(self, evt):
        self.imv.view.scene().removeItem(evt)

    def deleteMaxima(self):
        self.remove_gauss_ROI()
        for i in np.arange(0, self.fixing_number):
            try:
                self.imv.view.removeItem(self.molRoi[i,0])
#                self.imv.view.removeItem(self.molRoi[i,1])
                self.imv.view.removeItem(self.bgRoi[i,0])
#                self.imv.view.removeItem(self.bgRoi[i,1])
                self.imv.view.removeItem(self.label[i,0])
                self.imv.view.removeItem(self.label[i,1])
                del self.molRoi[i,0]
#                del self.molRoi[i,1]
                del self.bgRoi[i,0]
#                del self.bgRoi[i,1]
                del self.label[i,0]
#                del self.label[i,1]
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
                print("ya estaba borrado")  

        self.molRoi = dict()
        self.bgRoi = dict()
        self.label = dict()
        self.maxnumber = 0
        self.removerois = []
        self.fixing_number = 0
        self.image_analysis = False

    def translateMaxima(self):
        for i in range(len(self.molRoi)):  # np.arange(0, self.maxnumber):
            self.molRoi[i,0].translate(self.roi.pos())
#            self.molRoi[i,1].translate(self.roi.pos())
            self.bgRoi[i,0].translate(self.roi.pos())
#            self.bgRoi[i,1].translate(self.roi.pos())
            self.label[i,0].setPos(self.molRoi[i,0].pos())
#            self.label[i,1].setPos(self.molRoi[i,1].pos())
            try:
#                if i in self.removerois:
#                    self.imv.view.removeItem(self.gauss_roi[i])
#                else:
                self.gauss_roi[i].translate(self.roi.pos())
            except:
                pass

                
        
    def calculate_traces(self):
        # Create dict with traces
        self.trace = dict()
        molArray = dict()
        bgArray = dict()
        bg = dict()
        bgNorm = dict()
        
        j=0  # I justo kill the second trace

        p=0
        for i in range(len(self.molRoi)):  #2 np.arange(0, self.maxnumber):
            if i not in self.removerois:

                
                # get molecule array
                molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)

                # get background plus molecule array
                bgArray[i,j] = self.bgRoi[i,j].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)

                # get background array
                bg[i,j] = np.sum(bgArray[i,j], axis=self.axes) - np.sum(molArray[i,j], axis=self.axes)

                # get total background to substract from molecule traces
#                bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(bg[i,j])/(4*(int(self.moleculeSizeEdit.text())+1))
                bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(bg[i,j])/(((2* int(self.BgSizeEdit.text()))**2)*(int(self.moleculeSizeEdit.text())+1))

#bgNorm = [ Bg / ( m*m - n*n ) ] * n*n ==> m = n + s ==> [bg / s*s(n+1) ] n*n

                self.trace[p,j] = np.sum(molArray[i,j], axis=self.axes) - bgNorm[i,j]
                p +=1 # I have to use this to have order because of removerois

# =============================================================================
#                 # Correct second channel by channel correction input
#                 if j == 0:
#                     self.trace[i,j] = np.sum(molArray[i,j], axis=(1,2)) - bgNorm[i,j]
#                 else:
#                     self.trace[i,j] = float(self.channelCorrectionEdit.text())*(np.sum(molArray[i,j], axis=(1,2)) - bgNorm[i,j])
# =============================================================================
        
        # Save traces as an array
#        a = []        
#        for i in self.trace.keys():  # NO TIENE ORDEN!!!! CAMBIAR
#            a.append(self.trace[i])
        self.new_trace = dict()
        new_molArray = dict()
        new_bgArray = dict()
        new_bg = dict()
        new_bgNorm = dict()
        p=0
        print("new_i",self.new_i)
        for i in range(self.new_i):
            if i not in self.removed_new_rois:             
                # get molecule array
                new_molArray[i,j] = self.new_roi[i].getArrayRegion(self.mean, self.imv.imageItem, axis=self.axes, returnMappedCoords=False)

                # get background plus molecule array
                new_bgArray[i,j] = self.new_roi_bg[i].getArrayRegion(self.mean, self.imv.imageItem, axis=self.axes, returnMappedCoords=False)

                # get background array
                new_bg[i,j] = np.sum(new_bgArray[i,j], axis=self.axes) - np.sum(new_molArray[i,j], axis=self.axes)

                # get total background to substract from molecule traces
                new_bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(new_bg[i,j])/(((2* int(self.BgSizeEdit.text()))**2)*(int(self.moleculeSizeEdit.text())+1))

                self.new_trace[p,j] = np.sum(new_molArray[i,j], axis=self.axes) - new_bgNorm[i,j]
                p +=1 # I have to use this to have order because of removerois

        # Save traces as an array
#        a = []        
#        for i in self.trace.keys():  # NO TIENE ORDEN!!!! CAMBIAR
#            a.append(self.trace[i])

        a = []
        
        for p in range(len(self.trace)):
            a.append(self.trace[p,j])
        
        for p in range(len(self.new_trace)):
            a.append(self.new_trace[p,j])

        b = np.array(a).T        
        print("len traces", len(b))
        self.traces = b

    def calculate_images(self):
        print(" Calculate Images")
        # Create dict with traces
        self.sum_spot = dict()
        molArray = dict()
        bgArray = dict()
        bg = dict()
        bgNorm = dict()
        
        j=0  # I just do not use the second trace
        p=0
        for i in range(len(self.molRoi)):  # np.arange(0, self.maxnumber):
            if i not in self.removerois:             
                # get molecule array
                molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.mean, self.imv.imageItem)

                # get background plus molecule array
                bgArray[i,j] = self.bgRoi[i,j].getArrayRegion(self.mean, self.imv.imageItem)

                # get background array
                bg[i,j] = np.sum(bgArray[i,j]) - np.sum(molArray[i,j])

                # get total background to substract from molecule traces
                bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(bg[i,j])/(((2* int(self.BgSizeEdit.text()))**2)*(int(self.moleculeSizeEdit.text())+1))

                self.sum_spot[p,j] = np.sum(molArray[i,j]) - bgNorm[i,j]
                p +=1 # I have to use this to have order because of removerois

# =============================================================================
#                 # Correct second channel by channel correction input
#                 if j == 0:
#                     self.trace[i,j] = np.sum(molArray[i,j], axis=(1,2)) - bgNorm[i,j]
#                 else:
#                     self.trace[i,j] = float(self.channelCorrectionEdit.text())*(np.sum(molArray[i,j], axis=(1,2)) - bgNorm[i,j])
# =============================================================================
        self.new_sum_spot = dict()
        new_molArray = dict()
        new_bgArray = dict()
        new_bg = dict()
        new_bgNorm = dict()
        p=0
        print("new_i",self.new_i)
        for i in range(self.new_i):
            if i not in self.removed_new_rois:             
                # get molecule array
                new_molArray[i,j] = self.new_roi[i].getArrayRegion(self.mean, self.imv.imageItem)

                # get background plus molecule array
                new_bgArray[i,j] = self.new_roi_bg[i].getArrayRegion(self.mean, self.imv.imageItem)

                # get background array
                new_bg[i,j] = np.sum(new_bgArray[i,j]) - np.sum(new_molArray[i,j])

                # get total background to substract from molecule traces
                new_bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(new_bg[i,j])/(((2* int(self.BgSizeEdit.text()))**2)*(int(self.moleculeSizeEdit.text())+1))

                self.new_sum_spot[p,j] = np.sum(new_molArray[i,j]) - new_bgNorm[i,j]
                p +=1 # I have to use this to have order because of removerois
        # Save traces as an array
#        a = []        
#        for i in self.trace.keys():  # NO TIENE ORDEN!!!! CAMBIAR
#            a.append(self.trace[i])

        a = []

        for p in range(len(self.sum_spot)):
            a.append(self.sum_spot[p,j])
        
        for p in range(len(self.new_sum_spot)):
            a.append(self.new_sum_spot[p,j])
        
        b = np.array(a).T
        print("len cuentas", len(b))
        self.intensitys = b

    def export(self, what):
        print(" Export ", what)
        if what == "trace":
            b = self.traces
            trace_name = 'traces-'+ str(b.shape[1])+"("+ str(self.n)+")" + '.txt'
            np.savetxt(trace_name, b, delimiter="    ", newline='\r\n')
            print(b.shape[1],"Traces exported as", trace_name)
    
    
            exporter = pg.exporters.ImageExporter(self.imv.imageItem)
    #        # set export parameters if needed
    #        exporter.parameters()['width'] = 100   # (note this also affects height parameter)
            # save to file
            png_name = 'Image_traces-'+ str(b.shape[1]) +"(" + str(self.n)+")" + '.png'
            exporter.export(png_name)
            print( "Picture exported as", png_name)
    
            self.n += 1

        if what == "images":
            if self.histo_data:
                b = self.intensitys2
            else:
                b = self.intensitys
            intensities_name = 'intensities' + str(len(b))+"(" + str(self.n)+")"+ '.txt'
            np.savetxt(intensities_name, b, delimiter="    ", newline='\r\n')
            print(len(b), "intensities exported as", intensities_name)
    
    
            exporter = pg.exporters.ImageExporter(self.imv.imageItem)
    #        # set export parameters if needed
    #        exporter.parameters()['width'] = 100   # (note this also affects height parameter)
            # save to file
            png_name = 'Image_intensities'+ str(len(b))+"(" + str(self.n)+")" + '.png'
            exporter.export(png_name)
            print( "Picture exported as", png_name)
            
            self.n += 1

#    def make_histogram(self):
#        self.calculate_images()
#        print("create histogram")
#        import matplotlib.pyplot as plt
#        plt.hist(self.intensitys)
#        plt.grid()
#        plt.show()
#        vals = self.intensitys
#        self.plt1 = self.imv.addPlot()
#        y,x = np.histogram(vals)
#        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
#        self.plt1.showGrid(x = True, y = True, alpha = 0.5)

    def automatic_crazy_start(self):
        self.timing = 0
        self.automatic_crazytimer.start(50)  # imput in ms
        print("number of images to se :", int(self.crazyStepEdit.text()))

    def automatic_crazy(self):
#        N = int(self.crazyStepEdit.text())
#        for i in range(N):
        self.mean = self.data[int(self.timing*self.data.shape[0]//int(self.crazyStepEdit.text())),:,:]
        self.deleteMaxima()
        self.image_analysis = True
        self.detectMaxima()
        self.imv.setCurrentIndex(int(self.timing*self.data.shape[0]//int(self.crazyStepEdit.text())))

        self.gaussian_fit_ROI()
        self.filter_bg()

        self.make_histogram()
        print("step #", self.timing,"frame :", int(self.timing*self.data.shape[0]//int(self.crazyStepEdit.text())))
        self.timing +=1
        if self.timing == int(self.crazyStepEdit.text()):
            self.automatic_crazytimer.stop()
            print(" automatic analysis finished")

    def make_histogram(self):
        self.calculate_images()
        try:
#            print("concateno")
            self.intensitys2 = np.concatenate((self.intensitys,
                                               self.intensitys2))
        except:
#            print("1ra")
            self.intensitys2 = self.intensitys
        self.doit()
#        print("Histograming...")
        self.histo_data = True
        self.btn7.setText("Export all histogram data ({})".format(len(self.intensitys2)))


    def doit(self):
#        print("Opening a new popup window...")
        self.w2 = MyPopup_histogram(self)
        self.w2.setGeometry(QtCore.QRect(750, 50, 450, 600))
        self.w2.show()


class MyPopup_histogram(QtGui.QWidget):
    """ new class to create a new window for the trace menu"""

    def closeEvent(self, event):
        self.main.intensitys2 = None
        self.main.histo_data = False
        self.main.btn7.setText("Export only last ({}) points".format(len(self.main.intensitys)))
#        print("Close and clear the points")
        
    def __init__(self, main, *args, **kwargs):
        QtGui.QWidget.__init__(self)
        super().__init__(*args, **kwargs)
        self.main = main
#        self.ScanWidget = ScanWidget  # call ScanWidget
        self.traza_Widget2 = pg.GraphicsLayoutWidget()
        self.running = False
        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        self.p6 = self.traza_Widget2.addPlot(row=2, col=1, title="Traza")
        self.p6.showGrid(x=True, y=True)

        intensitys = self.main.intensitys2
        self.intensitys = intensitys

        grid.addWidget(self.traza_Widget2,      0, 0, 1, 7)
#        grid.addWidget(self.play_pause_Button,  1, 0)

        self.setWindowTitle("Histogram. (ESC key, close it.)")

        vals = self.intensitys
#        self.plt1 = self.traza_Widget2.addPlot()
        y,x = np.histogram(vals)
        self.p6.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        self.p6.showGrid(x = True, y = True, alpha = 0.5)

        self.close_Action = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('ESC'), self, self.close_win)

    def close_win(self):
        self.main.intensitys2 = None
        self.main.histo_data = False
        self.main.btn7.setText("Export only last ({}) points".format(len(self.main.intensitys)))

        self.close()

# %% Functions to make the Gauss fit
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):

    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

# %% END... Its a neverending story ♪♫
if __name__ == '__main__':

    app = pg.Qt.QtGui.QApplication([])
    exe = smAnalyzer()
    exe.w.show()
    app.exec_()



"""
Poner un slider para cambiar de frame al analizar imagenes
hacer que aprete el boton cuando se mueve

Por ahora conecte el slider que ya esta (pero se va cuando toco el boton, logico)
No necesito tocar el boton para hacer todo el analisis!

circularity
"""


