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

from pyqtgraph.dockarea import Dock, DockArea

import time as time

class smAnalyzer(pg.Qt.QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Define a top-level widget to hold everything
        self.w = QtGui.QWidget()
#        self.setCentralWidget(self.w)
        self.w.setWindowTitle('SMAnalyzer - Video')
        self.w.resize(20, 100)

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

        self.see_labels_button = QtGui.QCheckBox('Labels?')
        self.see_labels_button.setChecked(True)


        self.label_save = QtGui.QLabel('File Name')
        self.label_save.resize(self.label_save.sizeHint())
        self.edit_save = QtGui.QLineEdit('Test')
        self.edit_save.resize(self.edit_save.sizeHint())
        self.edit_save.setToolTip('Selec a name to save the data.\
              The name automatically changes to not replace the previous one')

        # Create a grid layout to manage the widgets size and position
#        self.layout = QtGui.QGridLayout()
#        self.w.setLayout(self.layout)

        self.viewer_grid = QtGui.QGridLayout()
        self.viewer_wid = QtGui.QWidget()
        self.viewer_wid.setLayout(self.viewer_grid)

        self.options_grid = QtGui.QGridLayout()
        self.optios_wid = QtGui.QWidget()
        self.optios_wid.setLayout(self.options_grid)

        self.post_grid = QtGui.QGridLayout()
        self.post_wid = QtGui.QWidget()
        self.post_wid.setLayout(self.post_grid)

        # Add widgets to the layout in their proper positions 
        #                                       (-Y, X, Y_width ,X_width)
#        self.layout.addWidget(QtGui.QLabel(" "),       0, 0, 1, 3)
        self.options_grid.addWidget(self.time_adquisitionLabel,     0, 0, 1, 1)
        self.options_grid.addWidget(self.time_adquisitionEdit,      0, 1, 1, 2)

        self.options_grid.addWidget(self.btn1,               1, 0, 1, 3)
        self.options_grid.addWidget(self.btn2,               2, 0, 1, 3)
        self.options_grid.addWidget(self.btn3,               3, 0, 1, 3)

        self.options_grid.addWidget(self.meanStartLabel,     4, 0, 1, 1)
        self.options_grid.addWidget(self.meanStartEdit,      4, 1, 1, 2)
        self.options_grid.addWidget(self.meanEndLabel,       5, 0, 1, 1)
        self.options_grid.addWidget(self.meanEndEdit,        5, 1, 1, 2)

        self.options_grid.addWidget(self.btn4,               6, 0, 1, 1)
        self.options_grid.addWidget(self.btn_images,         6, 2, 1, 1)
        
        self.options_grid.addWidget(self.btn5,               7, 0, 1, 3)
#        self.options_grid.addWidget(QtGui.QLabel(" "),       8, 0, 1, 3)
        
        self.options_grid.addWidget(self.maxDistLabel,       9, 0, 1, 1)
        self.options_grid.addWidget(self.maxDistEdit,        9, 1, 1, 2)
        self.options_grid.addWidget(self.maxThreshLabel,    10, 0, 1, 1)
        self.options_grid.addWidget(self.maxThreshEdit,     10, 1, 1, 2)
        self.options_grid.addWidget(self.moleculeSizeLabel, 11, 0, 1, 1)
        self.options_grid.addWidget(self.moleculeSizeEdit,  11, 1, 1, 2)
        self.options_grid.addWidget(self.BgSizeLabel,       12, 0, 1, 1)
        self.options_grid.addWidget(self.BgSizeEdit,        12, 1, 1, 2)
        
        self.options_grid.addWidget(self.btn6,              13, 0, 1, 3)
        self.options_grid.addWidget(self.btn99_clearall,    14, 2, 1, 1)
        self.options_grid.addWidget(self.btn7,              15, 0, 1, 3)


        self.viewer_grid.addWidget(self.label_save,         0, 4, 1, 5)
        self.viewer_grid.addWidget(self.edit_save,          0, 9, 1, 10)

        self.viewer_grid.addWidget(self.imv,              1, 4, 16, 16)
        

        self.post_grid.addWidget(self.see_labels_button, 3, 25, 1, 2)
        self.post_grid.addWidget(self.btn_small_roi,     4, 25, 1, 2)
        self.post_grid.addWidget(self.gauss_fit_label,   5, 25, 1, 1)
        self.post_grid.addWidget(self.gauss_fit_edit,    5, 26, 1, 1)
        self.post_grid.addWidget(self.btn_gauss_fit,     6, 25, 1, 2)
        self.post_grid.addWidget(self.btn_filter_bg,     8, 25, 1, 2)
        self.post_grid.addWidget(self.btn_histogram,    10, 25, 1, 2)
        self.post_grid.addWidget(self.crazyStepEdit,    12, 26, 1, 1)
        self.post_grid.addWidget(self.crazyStepButton,  12, 25, 1, 1)

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
        
        self.btn99_clearall.clicked.connect(self.clear_all)

        # automatic action when you edit the number 
        self.meanStartEdit.textEdited.connect(self.update_image)

        # a Python timer that call a function with a specific clock (later)
        self.automatic_crazytimer = QtCore.QTimer()
        self.automatic_crazytimer.timeout.connect(self.automatic_crazy)

        self.see_labels_button.clicked.connect(self.see_labels)

#        # DOCK cosas, mas comodo!
        self.state = None  # defines the docks state (personalize your oun UI!)

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        dockArea = DockArea()
        self.dockArea = dockArea
        grid.addWidget(self.dockArea)

        viewDock = Dock('viewbox', size=(300, 50))
        viewDock.addWidget(self.viewer_wid)
#        viewDock.hideTitleBar()
        self.dockArea.addDock(viewDock)

        postDock = Dock('posDetection', size=(1, 1))
        postDock.addWidget(self.post_wid)
        self.dockArea.addDock(postDock, "right", viewDock)

        optionsDock = Dock('Load options', size=(1, 1))
        optionsDock.addWidget(self.optios_wid)
        self.dockArea.addDock(optionsDock, "left", viewDock)

        self.setWindowTitle("Single Molecule Analizer")  # Nombre de la ventana
        self.setGeometry(10, 40, 1600, 800)  # (PosX, PosY, SizeX, SizeY)

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
        self.fixing_number = 0

        # Save file number
        self.n = 0

        self.is_image = False
        self.histo_data = False
        self.is_trace = False

    def importImage(self):  # connected to Load Image (btn1)
        """Select a file to analyse, can be a tif or jpg(on progres)
        the tiff data comes in a shape=(Frames, x, y) """

        self.JPG = False
        # Remove annoying empty window
        root = Tk()
        root.withdraw()

        # Select image from file
        self.f = filedialog.askopenfilename(filetypes=[("Videos", '*.tiff;*.tif;*.jpg'),
                                                       ("Pictures", "*.jpg")])
        if not self.f:
            print("No elegiste nada")
        else:
            self.file_path = self.f
            print("direccion elegida: \n", self.file_path, "\n")

        if self.f[-4:] == ".jpg":  # in case I want one picture

            self.JPG = True
            self.axes = (0,1)  # axe 2 are the 3 coloms of RGB
            print("WORKING ON THIS \n","JPG =", self.JPG,)
            self.data = np.mean(io.imread(self.f), axis=2)
            print(self.data.shape)
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

            
        else:
            # Import selected image
            self.data = io.imread(self.f)
            self.axes = (1,2)  # axe 0 are the frames
            self.total_size = [self.data.shape[2], self.data.shape[1]]

            self.maxDistEdit.setText("6")
            self.moleculeSizeEdit.setText("9")
            self.maxThreshEdit.setText(str(np.mean(self.data[1,:,:])))


        # Delete existing ROIs
        self.deleteROI()
        self.clear_all()
        
        plot_with_colorbar(self.imv, self.data)

        self.w.setWindowTitle('SMAnalyzer - Video - ' + self.f)
        self.imv.sigTimeChanged.connect(self.indexChanged)

        self.validator = QtGui.QIntValidator(0, self.data.shape[0])
        self.meanStartEdit.setValidator(self.validator)
        self.meanEndEdit.setValidator(self.validator)
#        try:
#            self.maxThreshEdit.setText(str(np.mean(self.data[1,:,:])))
#        except:
#            pass

    def update_image(self):  # Put the start frame in the image when change the number
        self.imv.setCurrentIndex(int(self.meanStartEdit.text()))

    def indexChanged(self):  #connected to the slide in the img
        """ change the numbers of start and endig frame  when move the slide"""
        self.meanStartEdit.setText(str((self.imv.currentIndex)))
        self.meanEndEdit.setText(str(int(self.imv.currentIndex)+15))

    def createROI(self):  # connected to Create ROI (btn2)
        """ create a big ROI to select the area to make the analysis
        default is the size of the picture"""
        if self.roi is None:
            self.roi = pg.ROI([0, 0], self.total_size,
                              scaleSnap=True, translateSnap=True,
                              removable=True)  # [70, 70]
            self.roi.addScaleHandle([1, 1], [0, 0])
            self.roi.addScaleHandle([0, 0], [1, 1])
            self.imv.view.addItem(self.roi)
            self.roi.sigRemoveRequested.connect(self.deleteROI)
        else:
            pass

    def deleteROI(self):  # connected to Delete ROI (btn3)
        """ delete the big ROI"""
        if self.roi is None:
            pass
        else:
            self.imv.view.removeItem(self.roi)
            self.roi = None

    def ROImean(self):  # connected to Get ROI mean (traces) (btn4)
        """ get the mean in the big ROI area, between the start and ending
        selected frames. Here self.mean is really a mean"""
        
        self.is_image = False
        self.is_trace = True

        # if you comes from images, it get the colors back to normal
        self.btn7.setText("Export Traces")
        self.btn7.setStyleSheet(
                "QPushButton { background-color: ; }")
        self.meanEndEdit.setStyleSheet(" background-color: ; ")

        z = self.roi.getArrayRegion(self.data, self.imv.imageItem, axes=self.axes)

        self.start = int(self.meanStartEdit.text())
        self.end = int(self.meanEndEdit.text())
        z = z[self.start:self.start+self.end, :, :]
            
        self.mean = np.mean(z, axis=0)  # axis=0 is the frames axis

        plot_with_colorbar(self.imv, self.mean)

        self.w.setWindowTitle('SMAnalyzer - ROI Mean - ' + self.f)
        self.imv.view.removeItem(self.roi)


    def showVideo(self):  # connected to Go to vide (btn5)
        """Get the original image back. Use this to came back from the 
        ROImean image.
        If you have rois in the mean image, they are moved with translatemaxima
        to the good positions in the origianl image. So you can follow them"""

        plot_with_colorbar(self.imv, self.data)

        self.w.setWindowTitle('SMAnalyzer - Video - ' + self.f)
        self.meanEndEdit.setStyleSheet(" background-color: ; ")

        try:
            self.translateMaxima()
            self.imv.view.addItem(self.roi)
        except:
            pass

        try:
            del self.mean2
        except:
            pass

    def translateMaxima(self):  # go to video call this function
        """ translate the position from the big ROI in to the video again"""
        for i in range(len(self.molRoi)):  # np.arange(0, self.maxnumber):
            self.molRoi[i].translate(self.roi.pos())
            self.bgRoi[i].translate(self.roi.pos())
            self.label[i].setPos(self.molRoi[i].pos())
            try:
                self.gauss_roi[i].translate(self.roi.pos())
            except:
                pass
        self.relabel_new_ROI()

    def detectMaxima(self):  # connected to Detect Molecules (btn6)
        """here is where the magic begins...
        if you did not had already created self.mean (that's not a mean for images) 
        it use the actual frame to see spots (Standar = image mode).
        Then, for each spot, creates a square roi of side size(=imput)
        and another roi bigger (also imput) to the background """

#        self.clear_all()
        self.dist = int(self.maxDistEdit.text())
        self.threshold = float(self.maxThreshEdit.text())
        
        # set roi Dimension array
        self.roiSize = [int(self.moleculeSizeEdit.text())] * 2
        self.bgroiSize = np.array(self.roiSize) + 2* int(self.BgSizeEdit.text())  # one pixel each side
        center = int(self.BgSizeEdit.text()) * np.array([1, 1])

        self.start = int(self.meanStartEdit.text())

        if self.JPG:
            if self.roi == None:
                self.mean = self.data
            else:
                self.mean = self.mean2
        else:
            if not self.is_image:
                try: 
                    self.mean = self.mean2[self.imv.currentIndex,:,:]
                except:
                    self.mean = self.data[self.imv.currentIndex,:,:]

        # find the local peaks
        self.maximacoord = peak_local_max(self.mean, min_distance=self.dist, threshold_abs=self.threshold)

        maxvalues = []
        for i in range(len(self.maximacoord[:,0])):
            maxvalues.append(self.mean[self.maximacoord[i,0],self.maximacoord[i,1]])

        # filter spurious peaks taking only the brighters ones
        nomaxlow = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]
        
        aux = np.arange(len(maxvalues))
        goodmax = np.delete(aux,nomaxlow)

        # Can do the same for the very brighter spots, but don't wanna
#        nomaxhigh = np.where(np.array(maxvalues) > 1.5*np.mean(np.array(maxvalues)[goodmax]))
#        toerase = np.sort(np.append(nomaxlow, nomaxhigh))

        maxindex = goodmax  # np.delete(aux,toerase)   NOT Nice for now

        print(len(goodmax), "good points finded")

        self.maxnumber = np.size(self.maximacoord[maxindex], 0)

        p = 0  
        print(self.fixing_number, "old points added")
        # I move my start because of the fixing number, so need to use p=0
        for i in np.arange(0, self.maxnumber)+self.fixing_number:

            # Translates molRoi to particle center
            corrMaxima = np.flip(self.maximacoord[maxindex[p]], 0) - 0.5*np.array(self.roiSize) + [0.5, 0.5]
            self.molRoi[i] = pg.ROI(corrMaxima, self.roiSize,scaleSnap=True,
                           translateSnap=True, movable=False, removable=True)
            self.bgRoi[i] = pg.ROI((corrMaxima - center), self.bgroiSize,
                                          scaleSnap=True, translateSnap=True,
                                          movable=False, removable=True)
            self.imv.view.addItem(self.molRoi[i])
            self.imv.view.addItem(self.bgRoi[i])

            self.molRoi[i].sigRemoveRequested.connect(self.remove_ROI)
            self.bgRoi[i].sigRemoveRequested.connect(self.remove_ROI)

            # Create ROI label
            self.label[i] = pg.TextItem(text=str(i))
            self.label[i].setPos(self.molRoi[i].pos())
            self.imv.view.addItem(self.label[i])
            p+=1
        try:
            self.fixing_number = i + 1
        except:
            print("ZERO points finded. ZERO!!")
        self.relabel_new_ROI()

        if not self.is_image:
            self.btn7.setText("Intensities from frame={}".format(int(self.meanStartEdit.text())))
        
        self.see_labels_button.setChecked(True)

    def exportTraces_or_images(self):  # connected to export traces button (btn7)
        if self.is_trace:
            self.calculate_traces()
            self.export("trace")            
        else:
            self.calculate_images()
            self.export("images")          


    def image_analysis(self):  # connected to image analysis button (btn_images)
        """Choose the image analysis mode, if there is a big ROI use this 
        area at the selected start number. If its not, use the actual one.
        The name "mean" is historical (and useful to no change code later),
        there is no mean in the images"""

        self.start = int(self.meanStartEdit.text())

        if self.roi == None and self.JPG == False :
            self.mean = self.data[self.start,:,:]
            self.is_image = True
        else:
            self.ROI_no_mean_images()

        # change the name and color of the buttos to notice the way you choose
        self.btn7.setText("Export Intensities from frame={}".format(self.start))
        self.btn7.setStyleSheet(
                "QPushButton { background-color: rgb(200, 200, 10); }")
        self.meanEndEdit.setStyleSheet(" background-color: red; ")

    def ROI_no_mean_images(self):  # Commes from image_analysis 
        """Choose the place to do the analysis, and "zoom in".
        if its a JPG only have one option"""

        z = self.roi.getArrayRegion(self.data, self.imv.imageItem, axes=self.axes)
        self.mean2 = z
        
        plot_with_colorbar(self.imv, self.mean2)

        self.w.setWindowTitle('SMAnalyzer - ROI Mean - ' + self.f)
        self.imv.view.removeItem(self.roi)


    def create_small_ROI(self):  # connected to New small Roi (btn_small_roi)
        """ creates a new green roi at (0,0) position. then you move it
        and click on it to add rois at your analysis"""
        
        if self.smallroi is not None:  # only want one
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

    def remove_small_ROI(self, evt):  # rigth click to delete the new small roi.
        self.imv.view.scene().removeItem(evt)

    def small_ROI_to_new_ROI(self): # connected to click the new small green roi
        """ it create a yellow new roi with backgroun and add it to the
        list of rois that want to be analysed"""

        print("\n YOU CLICK MEE 0_o \n")

        self.roiSize = [int(self.moleculeSizeEdit.text())] * 2
        self.bgroiSize = np.array(self.roiSize) + 2* int(self.BgSizeEdit.text())
        center = int(self.BgSizeEdit.text()) * np.array([1, 1])

        # fixing_number exist because you can put new rois before OR after the normal ones.
        i = self.fixing_number

        self.molRoi[i] = pg.ROI(self.smallroi.pos(), self.roiSize,
                                                       scaleSnap=True,
                                                       translateSnap=True,
                                                       movable=False,
                                                       removable=True,
                                                       pen='y') 
        self.bgRoi[i] = pg.ROI((self.smallroi.pos() - center), self.bgroiSize,
                                                      scaleSnap=True,
                                                      translateSnap=True,
                                                      movable=False,
                                                      removable=True,
                                                      pen='y') 
        self.imv.view.addItem(self.molRoi[i])
        self.imv.view.addItem(self.bgRoi[i])

        self.molRoi[i].sigRemoveRequested.connect(self.remove_ROI)
        self.bgRoi[i].sigRemoveRequested.connect(self.remove_ROI)

        self.label[i] = pg.TextItem(text=str(i))
        self.label[i].setPos(self.molRoi[i].pos())
        self.imv.view.addItem(self.label[i])

        self.fixing_number = i + 1

        self.relabel_new_ROI()

    def relabel_new_ROI(self):  # everiwhere you create or delete a roi
        """ fix the numeration and showing rois when you add or remove them
        in this case, call the normal one, and remove the manually yellow"""
        self.relabel_ROI()

        for i in self.removerois:
            self.imv.view.removeItem(self.molRoi[i])
            self.imv.view.removeItem(self.bgRoi[i])
            self.imv.view.removeItem(self.label[i])
            self.imv.view.removeItem(self.label[i])

    def relabel_ROI(self):  # from the new version (relabel_new_ROI)
        """ fix the numeration and showing rois when you add or remove them"""
        p = 0
        for i in np.arange(0, self.fixing_number):
            if i not in self.removerois:
                self.label[i].setText(text=str(p))
                p+=1

    def see_labels(self):
        if self.see_labels_button.isChecked():
            for i in range(len(self.molRoi)):
                if i not in self.removerois:
                    try:
                        self.imv.view.addItem(self.label[i])
                    except:
                        pass
        else:
            for i in range(len(self.molRoi)):
                try:
                    self.imv.view.removeItem(self.label[i])
                except:
                    pass

    def filter_bg(self):  # connected to filter bg (btn_filter_bg)
        """ Check at the counts in the background zone, if they are above
        the threshold level (user imput), discards the spot.
        discards the spots meas to add them the the black list (removerois)
        and then do not use them from the total molRoi list
        It check means in the columns and files of the bg Roi"""

        bgArray = dict()
        a = 0
        bgsize = int(self.BgSizeEdit.text())
        for i in range(len(self.molRoi)): #np.arange(0, self.maxnumber):
            if i not in self.removerois:
                # get background plus molecule array
                bgArray[i] = self.bgRoi[i].getArrayRegion(self.mean, self.imv.imageItem)                
                b = True
                for l in np.arange(-bgsize,bgsize):
                    if b:
                        if np.mean(bgArray[i][:,l]) > float(self.maxThreshEdit.text()) or \
                        np.mean(bgArray[i][l,:]) > float(self.maxThreshEdit.text()):
                            b = False  # just to stop if already find this.
                            self.bgRoi[i].setPen('r')
                            self.removerois.append(i)
                            a+=1


        print("badBg/total=", a,"/", len(self.molRoi))

    def gaussian_fit_ROI(self):  # connect to gaussian fit (btn_gauss_fit)
        """For each not discarted roi in molRoi, make a 2D gaussian fit
        and move to this new position x, y. Use the ratio (input) between 
        sigma_x and sigma_y to discard bad shapes.
        Also left a Blue semitransparent roi in the old position just to
        see what happened"""

        # first clear the old gauss fited
        self.remove_gauss_ROI()

        molArray = dict()
        self.gauss_roi = dict()
        print("Gauss fit for",len(self.molRoi)-len(self.removerois), "spots")
        a = 0
        for i in range(len(self.molRoi)):
            if i not in self.removerois:
                molArray[i] = self.molRoi[i].getArrayRegion(self.mean, self.imv.imageItem)
                data = np.transpose(molArray[i])  # not sure why, but is works

                try:  # if the fit fails, print error and continue with the next
                    new_params = fitgaussian(data)
                except:
                    continue

                (height, x, y, width_x, width_y) = new_params
                newx = x-self.roiSize[0]//2 + 0.5  # that is not pixel fixed
                newy = y-self.roiSize[1]//2 + 0.5
                originx =  self.molRoi[i].pos()[0]
                originy =  self.molRoi[i].pos()[1]
                
                # new name for the old roi position
                self.gauss_roi[i] = pg.ROI([originx,originy], self.roiSize, pen=(100,50,200,200),
                                                               scaleSnap=True,
                                                               translateSnap=True,
                                                               movable=False,
                                                               removable=False)

                # moves the original set of rois to the new fited coordinate
                self.molRoi[i].setPen('m')
                self.molRoi[i].translate([newx, newy])
                self.bgRoi[i].translate([newx, newy])

                self.imv.view.addItem(self.gauss_roi[i])

                threshold_sigma = float(self.gauss_fit_edit.text())
                sigma_ratio = width_x / width_y

                # take away the not so circular fits (imput gauss_fit_edit)
                if sigma_ratio > threshold_sigma or sigma_ratio < (1/threshold_sigma):
                    self.molRoi[i].setPen('r')
                    self.removerois.append(i)
                    a += 1

        print("badGauss/total=", a,"/", len(self.molRoi))
#        self.maxnumber_new_gauss = len(self.molRoi)
    def remove_gauss_ROI(self):
        """removes the gauss rois. They are not useful for anything,
        only to mark the spot"""

        for i in range(len(self.molRoi)):
            try:
                self.imv.view.removeItem(self.gauss_roi[i])
                del self.gauss_roi[i]
            except:
                pass

    def remove_ROI(self,evt):
        """ Remove the clicked roi, his background and label"""

        for i in np.arange(0, self.fixing_number):
            if self.bgRoi[i] == evt or self.molRoi[i] == evt:

                index = i
                self.removerois.append(index)

        print("Removed ROI", index)

        self.imv.view.scene().removeItem(self.molRoi[index])
        self.imv.view.scene().removeItem(self.bgRoi[index])
        self.imv.view.scene().removeItem(self.label[index])

        self.relabel_new_ROI()

    def clear_all(self):  # connected to button Clear all
        """ clear all the thing in the view, and initialize the variables"""

        self.remove_gauss_ROI()
        for i in np.arange(0, self.fixing_number):
            try:
                self.imv.view.removeItem(self.molRoi[i])
                self.imv.view.removeItem(self.bgRoi[i])
                self.imv.view.removeItem(self.label[i])
                self.imv.view.removeItem(self.label[i])
                del self.molRoi[i]
                del self.bgRoi[i]
                del self.label[i]
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
                print("ya estaba borrado")  

        self.molRoi = dict()
        self.bgRoi = dict()
        self.label = dict()
        self.maxnumber = 0
        self.removerois = []
        self.fixing_number = 0
        self.is_image = False

    def calculate_traces(self): # from exportTraces_or_images (<- btn7 call it)
        """ calculate the traces to save.
        For each spot take the counts of the molecular roi, and 
        substract the counts from the backgruond. Normalized for god.
        DO NOT FIX DRIFT"""

        # Create dict with traces
        self.trace = dict()
        molArray = dict()
#        bgArray = dict()
#        bg = dict()
        bgNorm = dict()

#        s = (2*int(self.BgSizeEdit.text()))  # bgsize = molsize + s
        p=0
        for i in range(len(self.molRoi)):  #2 np.arange(0, self.maxnumber):
            if i not in self.removerois:
                # get molecule array
                molArray[i] = self.molRoi[i].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)
                # get normalized background
                bgNorm[i] = get_counts_bgNorm(self.imv, self.data,
                                          self.molRoi[i], self.bgRoi[i],
                                          int(self.moleculeSizeEdit.text()),
                                          int(self.BgSizeEdit.text()))

                self.trace[p] = np.sum(molArray[i], axis=self.axes) - bgNorm[i]
                p +=1 # I have to use this to have order because of removerois

        # Save traces as an array
        a = []
        for p in range(len(self.trace)):
            a.append(self.trace[p])

        b = np.array(a).T        
        print("len traces", len(b))
        self.traces = b

    def calculate_images(self):# from exportTraces_or_images (<- btn7 call it)
        """ calculate the traces to save.
        For each spot take the counts of the molecular roi, and 
        substract the counts from the normalized backgruond """

        print("Calculate Images")
        
        # Create dict with spots
        self.sum_spot = dict()
        molArray = dict()
        bgArray = dict()
#        bg = dict()
        bgNorm = dict()
        morgane = []
#        s = (2*int(self.BgSizeEdit.text()))  # bgsize = molsize + s
        p=0
        for i in range(len(self.molRoi)):  # np.arange(0, self.maxnumber):
            if i not in self.removerois:             
                # get molecule array
                molArray[i] = self.molRoi[i].getArrayRegion(self.mean, self.imv.imageItem)

                # get background plus molecule array
                bgArray[i] = self.bgRoi[i].getArrayRegion(self.mean, self.imv.imageItem)

                # get normalized background
                bgNorm[i] = get_counts_bgNorm(self.imv, self.mean,
                                          self.molRoi[i], self.bgRoi[i],
                                          int(self.moleculeSizeEdit.text()),
                                          int(self.BgSizeEdit.text()))

                self.sum_spot[p] = np.sum(molArray[i]) - bgNorm[i]
                p +=1 # I have to use this to have order because of removerois
                morgane.append((np.sum(molArray[i]), np.sum(bgArray[i])))

        # Save sums as an array
        a = []
        for p in range(len(self.sum_spot)):
            a.append(self.sum_spot[p])
        
        b = np.array(a).T
        print("len spots", len(b))
        self.intensitys = b

        self.morgane = np.array(morgane)

    def export(self, what):  # after analysis, you want to save the data
        """ to save the data.
        Makes a diference betweeen the video part (traces)
        and the spot detection function (image analysis).
        Gives you back a .txt with 1 trace per column
        or only 1 column with all the intensities.
        Also a .png with the image with the rois as you see in the UI
        Save Customname+"trace/image"+#ofdetecctions+self.n;
        this las self.n change every click to avoid lost data.
        Now, alse sabes another file with the molArray and background
        without substracting anything, as morgane aks for.
        """

        Custom_name  = str(self.edit_save.text()) + "_"
        if what == "trace":
            b = self.traces
            trace_name = Custom_name  + 'traces-'+ str(b.shape[1])+"("+ str(self.n)+")" + '.txt'
            np.savetxt(trace_name, b, delimiter="    ", newline='\r\n')
            print("\n", b.shape[1],"Traces exported as", trace_name)
    
            exporter = pg.exporters.ImageExporter(self.imv.imageItem)
            png_name = Custom_name  + 'Image_traces-'+ str(b.shape[1]) +"(" + str(self.n)+")" + '.png'
            exporter.export(png_name)
            print( "\n Picture exported as", png_name)
    
            self.n += 1

        if what == "images":
            if self.histo_data:
                b = self.intensitys2
            else:
                b = self.intensitys
            intensities_name = Custom_name  + 'intensities' + str(len(b))+"(" + str(self.n)+")"+ '.txt'
            np.savetxt(intensities_name, b, delimiter="    ", newline='\r\n')
            print("\n", len(b), "Intensities exported as", intensities_name)

            c = self.morgane
            intensities_morgane_name = Custom_name  + 'intensities_morgane' + str(len(c))+"(" + str(self.n)+")"+ '.txt'
            np.savetxt(intensities_morgane_name, c, delimiter="    ", newline='\r\n')
            print("\n", len(c), "Intensities exported as", intensities_morgane_name)

            ratio = self.mean.shape[1]/self.mean.shape[0]
            height = int(1920)
            width = int(1920*ratio)
            exporter = pg.exporters.ImageExporter(self.imv.imageItem)
            exporter.params.param('width').setValue(width, blockSignal=exporter.widthChanged)
            exporter.params.param('height').setValue(height, blockSignal=exporter.heightChanged)

            exporter = pg.exporters.ImageExporter(self.imv.imageItem)
            png_name = Custom_name  + 'Image_intensities'+ str(len(b))+"(" + str(self.n)+")" + '.png'
            exporter.export(png_name)
            print( "\n Picture exported as", png_name)
            
            self.n += 1

# %% out of program
    def automatic_crazy_start(self): # connected to crazy go (crazyStepButton)
        """it start the timer with a specific time in ms to call again
        the function connected (automatic_crazy). 
        It stop when timer.spot is called (automatic_crazy function)"""
        self.timing = 0
        self.automatic_crazytimer.start(50)  # imput in ms
        print("number of images to see :", int(self.crazyStepEdit.text()))

    def automatic_crazy(self):  # from the timer of automatic_crazy_start
        """ to make a huge histogram looking all the video in small parts
        this is useful to have and idea or the results expected. 
        BUT as this is not perfec, it take a lot of bad spots.
        Makes (imput) steps"""

        self.mean = self.data[int(self.timing*self.data.shape[0]//int(self.crazyStepEdit.text())),:,:]
        self.clear_all()
        self.is_image = True
        self.detectMaxima()
        self.imv.setCurrentIndex(int(self.timing*self.data.shape[0]//int(self.crazyStepEdit.text())))
        self.filter_bg()
        self.gaussian_fit_ROI()
        self.filter_bg()
        self.make_histogram()
        print("step #", self.timing,"frame :", int(self.timing*self.data.shape[0]//int(self.crazyStepEdit.text())))
        self.timing +=1
        if self.timing == int(self.crazyStepEdit.text()):
            self.automatic_crazytimer.stop()
            print(" automatic analysis finished")

# %% for the new windows.
    def make_histogram(self):  # connected to make histogram (btn_histogram)
        """Prepare to make the histogram with all the spots detected.
        It opens a new window to run in another thread and  make it easy.
        If the new window is not closed, it add the new data to the histogram
        and can save all of this.
        When closed, starts over, AND CANNOT SAVE ALL THIS"""

        self.calculate_images()
        try:
            self.intensitys2 = np.concatenate((self.intensitys,
                                               self.intensitys2))
        except:
            self.intensitys2 = self.intensitys
        self.doit()
        self.histo_data = True
        self.btn7.setText("Export all histogram data ({})".format(len(self.intensitys2)))


    def doit(self):  # from make_histogram
        """start the new popup window. its run independenty of the Main win"""
        self.w2 = MyPopup_histogram(self)
        self.w2.setGeometry(QtCore.QRect(750, 50, 450, 600))
        self.w2.show()

class MyPopup_histogram(QtGui.QWidget):
    """ new class to create a new window for the histogram menu
    Can put a lot of thigs to play with the histogram, but for now is only a graph"""

    def closeEvent(self, event):  # when the win is closed
        """ things to do when the windows is closed: erase all the acumulated
        data; change the text on the butto to make clear what you have"""
        self.main.intensitys2 = None
        self.main.histo_data = False
        self.main.btn7.setText("Export only last ({}) points".format(len(self.main.intensitys)))
        
    def __init__(self, main, *args, **kwargs):  # when doit do w2.show()
        """this is the initialize of this new windows. Make the new
        windows appear and put the histogram inside. Can have mor things
        For now, it can be closed with ESC as a shortcut"""

        QtGui.QWidget.__init__(self)
        super().__init__(*args, **kwargs)

        self.main = main
        self.Histo_widget = pg.GraphicsLayoutWidget()
        grid = QtGui.QGridLayout()  # overkill for one item...
        self.setLayout(grid)

        self.p1 = self.Histo_widget.addPlot(row=2, col=1, title="Histogram (kHz)")
        self.p1.showGrid(x=True, y=True)

        intensitys = self.main.intensitys2
        self.intensitys = intensitys

        grid.addWidget(self.Histo_widget,      0, 0, 1, 7)
#        grid.addWidget(self.play_pause_Button,  1, 0)

        self.setWindowTitle("Histogram. (ESC key, close it.)")

        vals = self.intensitys / float(self.main.time_adquisitionEdit.text())
        y,x = np.histogram(vals)
        self.p1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        self.p1.showGrid(x = True, y = True, alpha = 0.5)

        # Shortcut. ESC ==> close_win
        self.close_Action = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('ESC'), self, self.close_win)

    def close_win(self):  # called pressing ESC
        """Close all when press ESC. Same changes es the other closeEvent
        (not sure about how to trigger both at the same time)"""
        
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

# %% functions to meka the life easier

def plot_with_colorbar(imv,data):
    
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

def get_counts_bgNorm(imv, image, molRoi, bgRoi, moleculeSize, BgSize):
    mean = image
    # get molecule array
    molArray = molRoi.getArrayRegion(mean, imv.imageItem)

    # get background plus molecule array
    bgArray = bgRoi.getArrayRegion(mean, imv.imageItem)

    # get background array
    bg = np.sum(bgArray) - np.sum(molArray)
    
    # get total background to substract from molecule traces
    n = moleculeSize
    m = (2*BgSize) + moleculeSize
    bgNorm = (n*n)*(bg) / (m*m - n*n)
    return bgNorm

# %% END... Its a neverending story ♪♫
if __name__ == '__main__':

    app = pg.Qt.QtGui.QApplication([])
    exe = smAnalyzer()
#    exe.w.show()
    exe.show()
    app.exec_()



"""
circularity---

maxnumber ?
"""


