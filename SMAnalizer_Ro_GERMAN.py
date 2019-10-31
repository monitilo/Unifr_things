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

        self.btn_small_roi = QtGui.QPushButton('New small ROI')

        # Create parameter fields with labels
        self.meanStartLabel = QtGui.QLabel('Start frame:')
        self.meanStartEdit = QtGui.QLineEdit('5')
        self.meanEndLabel = QtGui.QLabel('End frame:')
        self.meanEndEdit = QtGui.QLineEdit('15')
        self.maxDistLabel = QtGui.QLabel('Minimum distance:')
        self.maxDistEdit = QtGui.QLineEdit('3')
        self.maxThreshLabel = QtGui.QLabel('Threshold:')
        self.maxThreshEdit = QtGui.QLineEdit('0')
        self.moleculeSizeLabel = QtGui.QLabel('Size (pix):')
        self.moleculeSizeEdit = QtGui.QLineEdit('8')
        self.channelDifferenceLabel = QtGui.QLabel('Channel height difference (pixels):')
        self.channelDifferenceEdit = QtGui.QLineEdit('0')
        self.channelCorrectionLabel = QtGui.QLabel('Secondary Channel Correction:')
        self.channelCorrectionEdit = QtGui.QLineEdit('0')

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
        
        self.layout.addWidget(self.btn_small_roi, 2, 25, 1, 1)

        # button actions
        self.btn1.clicked.connect(self.importImage)
        self.btn2.clicked.connect(self.createROI)
        self.btn3.clicked.connect(self.deleteROI)
        self.btn4.clicked.connect(self.ROImean)
        self.btn5.clicked.connect(self.showVideo)
        self.btn6.clicked.connect(self.detectMaxima)
        self.btn7.clicked.connect(self.exportTraces)
        
#        self.btn_small_roi.clicked.connect(self.create_small_ROI)
        self.btn_small_roi.clicked.connect(self.gaussian_fit_ROI)

        # Create empty ROI
        self.roi = None
        self.smallroi = None

        # Molecule ROI dictionary
        self.molRoi = dict()
        self.bgRoi = dict()
        
        self.removerois = []
        # ROI label dictionary
        self.label = dict()

        # Initial number of maximums detected
        self.maxnumber = 0
        
        # Save file number
        self.n = 0

        self.JPG = False

    def algo(self):
        print("a")

    def create_small_ROI(self):
#        if self.smallroi is None:
        try:
            roisize = int(self.moleculeSizeEdit.text())
            self.smallroi = pg.ROI([0, 0], [roisize, roisize],
                                   scaleSnap=True, translateSnap=True,
                                   movable=True, removable=True)
            self.imv.view.addItem(self.smallroi)
            self.smallroi.sigRemoveRequested.connect(self.remove_small_ROI)
            self.smallroi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            self.smallroi.sigClicked.connect(self.algo)
            
        except:
            pass

    def createROI(self):
        if self.roi is None:
            self.roi = pg.ROI([0, 0], [70, 70] , scaleSnap=True, translateSnap=True)  # [self.data.shape[2], self.data.shape[1]]
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
        
        self.maximacoord = peak_local_max(self.mean, min_distance=self.dist, threshold_abs=self.threshold)

        maxvalues = []
        for i in range(len(self.maximacoord[:,0])):
            maxvalues.append(self.mean[self.maximacoord[i,0],self.maximacoord[i,1]])
        
        nomaxlow = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]
        
        aux = np.arange(len(maxvalues))
        goodmax = np.delete(aux,nomaxlow)
        
        nomaxhigh = np.where(np.array(maxvalues) > 1.5*np.mean(np.array(maxvalues)[goodmax]))
        
        toerase = np.sort(np.append(nomaxlow, nomaxhigh))
        maxindex = np.delete(aux,toerase)

        print(len(goodmax), "points finded")

        self.maxnumber = np.size(self.maximacoord[maxindex], 0)
        for i in np.arange(0, self.maxnumber):
            
            # Translates molRoi to particle center
            corrMaxima = np.flip(self.maximacoord[maxindex[i]], 0) - 0.5*np.array(self.roiSize) + [0.5, 0.5]
            self.molRoi[i,0] = pg.ROI(corrMaxima, self.roiSize,
                                                           scaleSnap=True,
                                                           translateSnap=True,
                                                           movable=False,
                                                           removable=True)
            self.bgRoi[i,0] = pg.ROI((corrMaxima - [1, 1]), self.bgroiSize,
                                                          scaleSnap=True,
                                                          translateSnap=True,
                                                          movable=False,
                                                          removable=True)
            self.molRoi[i,1] = pg.ROI(corrMaxima - [0, int(self.channelDifferenceEdit.text())], self.roiSize, scaleSnap=True, translateSnap=True, movable=False, removable=True)
            self.bgRoi[i,1] = pg.ROI(corrMaxima - [1, 1] - [0, int(self.channelDifferenceEdit.text())], self.bgroiSize, scaleSnap=True, translateSnap=True, movable=False, removable=True)
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
            self.label[i,1].setPos(self.molRoi[i,1].pos())
            self.imv.view.addItem(self.label[i,0])
#            self.imv.view.addItem(self.label[i,1])
        self.Nparticles = self.maxnumber

    def gaussian_fit_ROI(self):
        # get molecule array
        molArray = dict()
        self.newRoi = dict()
        j=0
        i=1
        print("start gaussian")
        print(len(self.molRoi)//2)
        for i in range(len(self.molRoi)//2):
            molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.mean, self.imv.imageItem)
            print("i= ", i)
            data = np.transpose(molArray[i,j])
            params = fitgaussian(data)
            fit = gaussian(*params)
            new_params = fitgaussian(molArray[i,j])
    #        all_params[j] = new_params
            (height, x, y, width_x, width_y) = new_params
            print("\n new_params \n",
                                     "[amplitude, x, y, Sigma_x, sigma_y] \n",
                                     new_params)
            print(self.roiSize, np.round(x),np.round(y))
            newx = np.round(x)-self.roiSize[0]//2
            newy = np.round(y)-self.roiSize[1]//2
            print(self.molRoi[i,j].pos())
            originx =  self.molRoi[i,j].pos()[0]
            originy =  self.molRoi[i,j].pos()[1]
#            self.molRoi[i,0].translate([newy, newx])
            self.newRoi[i] = pg.ROI([originx+newx,originy+newy], self.roiSize, pen='m',
                                                           scaleSnap=True,
                                                           translateSnap=True,
                                                           movable=False,
                                                           removable=True)
            self.imv.view.addItem(self.newRoi[i])
            print("Created new roi",i, "to", [newy, newx],"\n")
            if width_x > 6 or width_y > 6:
                self.newRoi[i].setPen('r')

    def relabel_ROI(self):
        p = 0
        for i in np.arange(0, self.maxnumber):
            if i not in self.removerois:
#                print(i,self.removerois)
                self.imv.view.removeItem(self.label[i,0])
                self.imv.view.removeItem(self.label[i,1])
                self.label[i,0] = pg.TextItem(text=str(p))
                self.label[i,0].setPos(self.molRoi[i,0].pos())
                self.imv.view.addItem(self.label[i,0])
                p+=1
        self.Nparticles = p-1

    def remove_ROI(self,evt):
        print("Remove_ROI")
        for i in np.arange(0, self.maxnumber):
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

        self.relabel_ROI()

    def remove_small_ROI(self, evt):
        self.imv.view.scene().removeItem(evt)

    def deleteMaxima(self):
        for i in np.arange(0, self.maxnumber):
            try:
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
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
                print("ya estaba borrado")
        self.molRoi = dict()
        self.bgRoi = dict()
        self.label = dict()
        self.maxnumber = 0
        self.removerois = []

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
        
        j=0  # I justo kill the second trace
        # Create dict with traces
        p=0
        for i in np.arange(0, self.maxnumber):
            if i not in self.removerois:

                
                # get molecule array
                molArray[i,j] = self.molRoi[i,j].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)

                # get background plus molecule array
                bgArray[i,j] = self.bgRoi[i,j].getArrayRegion(self.data, self.imv.imageItem, axes=self.axes, returnMappedCoords=False)

                # get background array
                bg[i,j] = np.sum(bgArray[i,j], axis=self.axes) - np.sum(molArray[i,j], axis=self.axes)

                # get total background to substract from molecule traces
                bgNorm[i,j] = (int(self.moleculeSizeEdit.text())**2)*(bg[i,j])/(4*(int(self.moleculeSizeEdit.text())+1))

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

        a = []
        print("len", len(self.trace))
        for p in range(len(self.trace)):
            a.append(self.trace[p,j])

        b = np.array(a).T
        np.savetxt('traces' + str(self.n) + '.txt', b, delimiter="    ", newline='\r\n')
        print( "Trace exported as", 'traces' + str(self.n) + '.txt')


        exporter = pg.exporters.ImageExporter(self.imv.imageItem)
#        # set export parameters if needed
#        exporter.parameters()['width'] = 100   # (note this also affects height parameter)
        # save to file
        exporter.export('Image' + str(self.n) + '.png')
        print( "Picture exported as", 'Image' + str(self.n) + '.png')

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