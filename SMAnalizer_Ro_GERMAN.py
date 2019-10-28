"""
@author: Rodrigo el que no documenta su codigo
@Enchulado por: German, el que comenta cosas poco serias

Interfaz usuario para detectar las particulas en una pelicula
y extraer sus trazas en un archivo .txt
- Se crea un Roi de tamaño a eleccion (muy grande tarda mucho)
- Se eligen Start frame y End frame para promediar dentro de ese roi
y tener una imagen mas nitida donde detectar las particulas
- Get ROI mean devuelve la imagen del ROI promediada, donde 
luego se van a detectar los maximos.
- Se elije una Minimun distance entre maximos
- Threshold de ser necesario. Solo detecta maximos mayores a esto
- Size: el tamaño de tu PSF
 COSAS PARA FRET QUE VOY A SACAR
     - Channel Height difference (pixels): 
     - Secondary Channel Correction:
- Detect Molecules: busca todos los maximos locales en el ROI
promediado, a distancias mayores a minimun distance. Dibuja un
cuadrado tamaño size*size sobre cada punto detectado.
resta el bakground calculado de un cuadrado 1 pixel mas de la periferia
(size+1)*(size+1) - size*size.

-Export Traces: Guarda el archivo .txt con 1 columna por particula.

"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from skimage import io
from tkinter import Tk, filedialog
from skimage.feature import peak_local_max
import pyqtgraph.exporters

class smAnalyzer(pg.Qt.QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Define a top-level widget to hold everything
        self.w = QtGui.QWidget()
        self.w.setWindowTitle('SMAnalyzer - Video')
        self.w.resize(1300, 800)

        # Create ImageView
        self.imv = pg.ImageView()

        # Create buttons
        self.btn1 = QtGui.QPushButton('Load Image')
        self.btn2 = QtGui.QPushButton('Create ROI')
        self.btn3 = QtGui.QPushButton('Delete ROI')
        self.btn4 = QtGui.QPushButton('Get ROI mean')
        self.btn5 = QtGui.QPushButton('Go to Video')
        self.btn6 = QtGui.QPushButton('Detect Molecules')
        self.btn7 = QtGui.QPushButton('Export Traces')

        # Create parameter fields with labels
        self.meanStartLabel = QtGui.QLabel('Start frame:')
        self.meanStartEdit = QtGui.QLineEdit()
        self.meanEndLabel = QtGui.QLabel('End frame:')
        self.meanEndEdit = QtGui.QLineEdit()
        self.maxDistLabel = QtGui.QLabel('Minimum distance:')
        self.maxDistEdit = QtGui.QLineEdit()
        self.maxThreshLabel = QtGui.QLabel('Threshold:')
        self.maxThreshEdit = QtGui.QLineEdit()
        self.moleculeSizeLabel = QtGui.QLabel('Size (pix):')
        self.moleculeSizeEdit = QtGui.QLineEdit()
        self.channelDifferenceLabel = QtGui.QLabel('Channel height difference (pixels):')
        self.channelDifferenceEdit = QtGui.QLineEdit()
        self.channelCorrectionLabel = QtGui.QLabel('Secondary Channel Correction:')
        self.channelCorrectionEdit = QtGui.QLineEdit()

        # Create a grid layout to manage the widgets size and position
        self.layout = QtGui.QGridLayout()
        self.w.setLayout(self.layout)

        # Add widgets to the layout in their proper positions
        self.layout.addWidget(self.btn1, 0, 0, 1, 3)
        self.layout.addWidget(self.btn2, 1, 0, 1, 3)
        self.layout.addWidget(self.btn3, 2, 0, 1, 3)

        self.layout.addWidget(self.meanStartLabel, 3, 0, 1, 1)
        self.layout.addWidget(self.meanStartEdit,  3, 1, 1, 2)
        self.layout.addWidget(self.meanEndLabel,   4, 0, 1, 1)
        self.layout.addWidget(self.meanEndEdit,    4, 1, 1, 2)


        self.layout.addWidget(self.btn4, 5, 0, 1, 3)

        self.layout.addWidget(QtGui.QLabel(" "), 6, 0, 1, 3)
        self.layout.addWidget(self.btn5, 7, 0, 1, 3)
#        self.layout.addWidget(QtGui.QLabel(" "), 8, 0, 1, 3)
        
        self.layout.addWidget(self.maxDistLabel, 8, 0, 1, 1)
        self.layout.addWidget(self.maxDistEdit, 8, 1, 1, 2)
        self.layout.addWidget(self.maxThreshLabel, 9, 0, 1, 1)
        self.layout.addWidget(self.maxThreshEdit, 9, 1, 1, 2)
        self.layout.addWidget(self.moleculeSizeLabel, 10, 0, 1, 1)
        self.layout.addWidget(self.moleculeSizeEdit, 10, 1, 1, 2)
        self.layout.addWidget(self.channelDifferenceLabel, 11, 0, 1, 1)
        self.layout.addWidget(self.channelDifferenceEdit, 11, 1, 1, 2)
        self.layout.addWidget(self.channelCorrectionLabel, 12, 0, 1, 1)
        self.layout.addWidget(self.channelCorrectionEdit, 12, 1, 1, 2)

        self.layout.addWidget(self.btn6, 13, 0, 1, 3)

        self.layout.addWidget(QtGui.QLabel(" "), 14, 0, 1, 3)
        self.layout.addWidget(self.btn7, 15, 0, 1, 3)
        self.layout.addWidget(self.imv, 0, 4, 16, 16)

        # button actions
        self.btn1.clicked.connect(self.importImage)
        self.btn2.clicked.connect(self.createROI)
        self.btn3.clicked.connect(self.deleteROI)
        self.btn4.clicked.connect(self.ROImean)
        self.btn5.clicked.connect(self.showVideo)
        self.btn6.clicked.connect(self.detectMaxima)
        self.btn7.clicked.connect(self.exportTraces)

        # Create empty ROI
        self.roi = None

        # Molecule ROI dictionary
        self.molRoi = dict()
        self.bgRoi = dict()
        
        # ROI label dictionary
        self.label = dict()

        # Initial number of maximums detected
        self.maxnumber = 0
        
        # Save file number
        self.n = 0

        self.JPG = False
    def createROI(self):
        if self.roi is None:
            self.roi = pg.ROI([0, 0], [10, 10], scaleSnap=True, translateSnap=True)
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
        self.f = filedialog.askopenfilename(filetypes=[("Videos", '*.tiff;*tif'),
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

    def ROImean(self):
        z = self.roi.getArrayRegion(self.data, self.imv.imageItem, axes=self.axes)
        if not self.JPG:

            self.start = int(self.meanStartEdit.text())
            self.end = int(self.meanEndEdit.text())
            z = z[self.start:self.start+self.end, :, :]
            self.mean = np.mean(z, axis=0)
        else:
            self.mean = z 

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
        self.translateMaxima()
        self.imv.view.addItem(self.roi)

    def detectMaxima(self):
        self.deleteMaxima()
        self.dist = int(self.maxDistEdit.text())
        self.threshold = int(self.maxThreshEdit.text())
        
        # set roi Dimension array
        self.roiSize = [int(self.moleculeSizeEdit.text())] * 2
        self.bgroiSize = np.array(self.roiSize) + 2  # one pixel each side
        
        self.maxima = peak_local_max(self.mean, min_distance=self.dist, threshold_abs=self.threshold)

        self.maxnumber = np.size(self.maxima, 0)
        for i in np.arange(0, self.maxnumber):
            
            # Translates molRoi to particle center
            corrMaxima = np.flip(self.maxima[i], 0) - 0.5*np.array(self.roiSize) + [0.5, 0.5]
            self.molRoi[i,0] = pg.ROI(corrMaxima, self.roiSize, scaleSnap=True, translateSnap=True, movable=False)
            self.bgRoi[i,0] = pg.ROI((corrMaxima - [1, 1]), self.bgroiSize, scaleSnap=True, translateSnap=True, movable=False)
            self.molRoi[i,1] = pg.ROI(corrMaxima - [0, int(self.channelDifferenceEdit.text())], self.roiSize, scaleSnap=True, translateSnap=True, movable=False)
            self.bgRoi[i,1] = pg.ROI(corrMaxima - [1, 1] - [0, int(self.channelDifferenceEdit.text())], self.bgroiSize, scaleSnap=True, translateSnap=True, movable=False)
            self.imv.view.addItem(self.molRoi[i,0])
            self.imv.view.addItem(self.molRoi[i,1])
            self.imv.view.addItem(self.bgRoi[i,0])
            self.imv.view.addItem(self.bgRoi[i,1])
            
            # Create ROI label
            self.label[i,0] = pg.TextItem(text=str(i))
            self.label[i,1] = pg.TextItem(text=str(i))
            self.label[i,0].setPos(self.molRoi[i,0].pos())
            self.label[i,1].setPos(self.molRoi[i,1].pos())
            self.imv.view.addItem(self.label[i,0])
            self.imv.view.addItem(self.label[i,1])

    def deleteMaxima(self):
        for i in np.arange(0, self.maxnumber):
            self.imv.view.removeItem(self.molRoi[i,0])
            self.imv.view.removeItem(self.molRoi[i,1])
            self.imv.view.removeItem(self.bgRoi[i,0])
            self.imv.view.removeItem(self.bgRoi[i,1])
            self.imv.view.removeItem(self.label[i,0])
            self.imv.view.removeItem(self.label[i,1])
            del self.molRoi[i,0]
            del self.molRoi[i,1]
            del self.bgRoi[i,0]
            del self.bgRoi[i,1]
            del self.label[i,0]
            del self.label[i,1]
        self.molRoi = dict()
        self.bgRoi = dict()
        self.label = dict()
        self.maxnumber = 0

    def translateMaxima(self):
        for i in np.arange(0, self.maxnumber):
            self.molRoi[i,0].translate(self.roi.pos())
            self.molRoi[i,1].translate(self.roi.pos())
            self.bgRoi[i,0].translate(self.roi.pos())
            self.bgRoi[i,1].translate(self.roi.pos())
            self.label[i,0].setPos(self.molRoi[i,0].pos())
            self.label[i,1].setPos(self.molRoi[i,1].pos())
        
    def exportTraces(self):
        self.trace = dict()
        molArray = dict()
        bgArray = dict()
        bg = dict()
        bgNorm = dict()
        
        # Create dict with traces
        for i in np.arange(0, self.maxnumber):
            for j in np.arange(1):  # I justo kill the second trace
                
                # get molecule array
                molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)

                # get background plus molecule array
                bgArray[i,j] = self.bgRoi[i,j].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)

                # get background array
                bg[i,j] = np.sum(bgArray[i,j], axis=self.axes) - np.sum(molArray[i,j], axis=self.axes)

                # get total background to substract from molecule traces
                bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(bg[i,j])/(4*(int(self.moleculeSizeEdit.text())+1))

                self.trace[i,j] = np.sum(molArray[i,j], axis=self.axes) - bgNorm[i,j]

# =============================================================================
#                 # Correct second channel by channel correction input
#                 if j == 0:
#                     self.trace[i,j] = np.sum(molArray[i,j], axis=(1,2)) - bgNorm[i,j]
#                 else:
#                     self.trace[i,j] = float(self.channelCorrectionEdit.text())*(np.sum(molArray[i,j], axis=(1,2)) - bgNorm[i,j])
# =============================================================================
        
        # Save traces as an array
        a = []        
        for i in self.trace.keys():
            a.append(self.trace[i])

        b = np.array(a).T
        np.savetxt('traces' + str(self.n) + '.txt', b, delimiter="    ", newline='\r\n')
        print( "Trace exported as", 'traces' + str(self.n) + '.txt')


        exporter = pg.exporters.ImageExporter(self.imv.imageItem)
#        # set export parameters if needed
#        exporter.parameters()['width'] = 100   # (note this also affects height parameter)
        # save to file
        exporter.export('Image' + str(self.n) + '.png')

        self.n += 1
# =============================================================================
# images = np.random.random((3,128,128))
# 
# imagewindow = pg.image()
# 
# for i in xrange(images.shape[0]):
#     img = pg.ImageItem(images[i])
#     imagewindow.clear()
#     imagewindow.addItem(img)
#     exporter = pg.exporters.ImageExporter(imagewindow.view)
#     exporter.export('image_'+str(i)+'.png')
#         
# =============================================================================
if __name__ == '__main__':

    app = pg.Qt.QtGui.QApplication([])
    exe = smAnalyzer()
    exe.w.show()
    app.exec_()