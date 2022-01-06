# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:03:40 2019

@author: Cecilia Zaza

Este programa es una interfaz usuario que permite inspeccionar y seleccionar 
trazas, o alguna señal 1D desde un archivo .txt.
Cada columna del archivo txt, corresponde a una traza o set de datos
-Campo para poner el tiempo de exposicion o tiempo caracteristico del detector usado para
calculo de tiempos
-Tiene un Slider para inspeccionar toda la data
-Tiene un slider para tocar el umbral para separar entre eventos on y off. Si no se 
cambia, se guarda el que viene por default (ajuste de gaussiana del fondo + 10sigmas)
-Un campo editable para mirar una traza o set de datos especifíco
-Los botones good Trace y bad Trace, sirven para decidir si se quiere trabajar 
con esa traza o set de datos que se encuentra graficada
-Al exportar, se va a indicar el porcentaje de trazas buenas seleccionadas y
se guarda un archivo .txt con las trazas "good traces" cuyo nombre empieza con
"FILTERED_" y sigue con el nombre del archivo de trazas seleccionado. Dos archivos .txt
con los tiempos On y Off calculados de las "good traces".
-En la matriz self.selection se guarda lo siguiente: la primera columna
es el indice de la traza, en la segunda columna se va a guardar un 1 si la
traza es buena y un -1 si la traza es mala. En la tercera columna inicialmente, 
se carga un umbral determinado como la moda (fondo) + std de cada traza que luego se
ira cambiando. En la cuarta y quinta columna se guardan la seleccion de frames realizada 
con el programa para hacer un post analisis. 
OJO: Si no ponen el exposure time, el programa no va a calcular los tiempos On y Off

Bugs detectados:
  - No muestra correctamente la traza cuando se carga el archivo, es necesario mover el slider para actualizar
  - Las zonas verde y roja esta siempre compactadas de un lado.


## ENGLISH

This program is a user interface that allows you to inspect and select 
traces, or some 1D signal from a .txt file.
Each column in the txt file corresponds to a trace or dataset
-Field for setting the exposure time or characteristic time of the detector used for
time calculation
-It has a slider to inspect all the data
-It has a slider to touch the threshold to separate between on and off events. If you don't 
changes, the default is saved (background gaussian setting + 10sigma)
-An editable field to look at a specific trace or data set
-The good Trace and bad Trace buttons are used to decide if you want to work 
with that trace or dataset that is graphed
-When exporting, the percentage of selected good traces will be indicated and
a .txt file with the "good traces" is saved, the name of which starts with
"FILTERED_" and continues with the name of the selected trace file. Two .txt files
with the calculated On and Off times of the good traces(if the calculate ON OFF times buttos was pressed).
-The following is stored in the self.selection matrix: 
    -the first column is the index of the trace, 
    -in the second column a 1 will be saved if the trace is good and a -1 if the trace is bad. 
    -In the third column initially, a certain threshold is loaded as the mode (background) + std 
       of each trace which is then will be changing. 
    -In the fourth and fifth columns, the selection of frames made with the program to do a post analysis. (deprecated)
    - The final colum (6th) is the "step intensity". Green zone mean - red zone mean.

WARNING: If you do not set the exposure time, the program will not calculate the On and Off times


Detected  Bugs:
  - The graph do not load correctly with the file, moving the slider is needed to update it.
  - The green / red zone are colapsed into a side.    

"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import os
from tkinter import Tk, filedialog
#from skimage.feature import peak_local_max
from scipy import stats
from pyqtgraph.dockarea import Dock, DockArea

class Trace_Inspector(pg.Qt.QtGui.QMainWindow):  # pg.Qt.QtGui.QMainWindow

    def closeEvent(self, event):
        print("close Event")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Interpret image data as row-major instead of col-major
#        pg.setConfigOptions(imageAxisOrder='row-major')

        # General Configuration
        self.testData = False
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)
        
        # Define a top-level widget to hold everything
        self.w = QtGui.QWidget()
        self.w.setWindowTitle('Trace Inspector')
#        self.w.resize(500, 800)
#        self.w.setGeometry(10, 40, 300, 300)  # (PosX, PosY, SizeX, SizeY)

        # Create ImagePlot
        self.graph = pg.PlotWidget()
        self.BinaryTrace = pg.PlotWidget()
#        self.histo_window = pg.GraphicsWindow()

        # Create buttons
        self.btnLoad = QtGui.QPushButton('Load Traces')
        self.btnShow = QtGui.QPushButton('Show Trace')
        self.btnGoodTrace = QtGui.QPushButton('Good Trace (g)')
        self.btnBadTrace = QtGui.QPushButton('Bad Trace (b)')
        self.btnTonTimes = QtGui.QPushButton('Calculate Ton and Toff')
        self.btnExport = QtGui.QPushButton('Export Trace Selection and T')
        self.btnautomatic_detect = QtGui.QPushButton('Automatic find threshold ')
        self.btn_import_selection = QtGui.QPushButton('Choose and existing selection matrix')

        # Create parameter fields with labels
        self.traceindex = QtGui.QLabel('Show Trace:')
        self.traceindexEdit = QtGui.QLineEdit('0')
        self.ExposureTime = QtGui.QLabel('Exposure Time [ms]')
        self.ExposureTimeEdit = QtGui.QLineEdit("1")
#        self.ThesholdIndex= QtGui.QLabel('Threshold:')
#        self.ThesholdIndexEdit = QtGui.QLineEdit()
#        self.StartFrame = QtGui.QLabel('Start frame:')
#        self.StartFrameEdit = QtGui.QLineEdit('0')
#        self.EndFrame = QtGui.QLabel('End frame:')
#        self.EndFrameEdit = QtGui.QLineEdit()
     
        # Create Slider Trace Index
        self.traceSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.traceSlider.setMinimum(0 )
        self.traceSlider.setMaximum(1)
        self.traceSlider.setValue(0)
        self.traceSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.traceSlider.setTickInterval(1)
       
        Trace_index_Slider = QtGui.QLabel('Trace index:')
        self.Trace_index_Slider_Edit = QtGui.QLabel('0')
        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value())))
        
        # Create Slicer for the threshold
        self.thresholdSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.thresholdSlider.setMinimum(0)
        self.thresholdSlider.setMaximum(1)
        self.thresholdSlider.setValue(0)
        self.thresholdSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.thresholdSlider.setTickInterval(0.1)
#        self.thresholdSlider.setSingleStep(1)

        threshold_index_Slider = QtGui.QLabel('Threshold Slider:')
        self.threshold_index_Slider_Edit = QtGui.QLabel('0')
        self.threshold_index_Slider_Edit.setText(format(int(self.thresholdSlider.value())))

        # Final message after saving data
        self.Amount_goodTraces_text = QtGui.QLabel('Good traces selection:')
        self.Amount_goodTraces = QtGui.QLabel() 

        # Labels to know the means to save
        self.labelmax = QtGui.QLabel("Left")
        self.labelmin = QtGui.QLabel("Rigth")
        self.labelstep = QtGui.QLabel("substract L-R")
        self.labelmax.setFixedWidth(300)
        self.labelmin.setFixedWidth(300)
        self.labelstep.setFixedWidth(300)
        # Button to print it (and save)
        self.btnmaxmin = QtGui.QPushButton('Calculate RigthMean - LeftMean')

        self.labelstep2 = QtGui.QLabel("Usign threshold")
        self.labelstep2.setFixedWidth(300)

        # Create a grid layout to manage the widgets size and position
#        layout = QtGui.QGridLayout()
#        self.w.setLayout(layout)

        self.Trace_grid = QtGui.QGridLayout()
        self.Trace_wid = QtGui.QWidget()
        self.Trace_wid.setLayout(self.Trace_grid)

        # Add widgets to the layout in their proper positions
        self.Trace_grid.addWidget(self.btnLoad,                     0, 0, 1, 3)
        self.Trace_grid.addWidget(self.ExposureTime,                1, 0, 1, 1)
        self.Trace_grid.addWidget(self.ExposureTimeEdit,            1, 1, 1, 2)
        self.Trace_grid.addWidget(Trace_index_Slider,               2.5, 0, 1, 3)
        self.Trace_grid.addWidget(self.Trace_index_Slider_Edit,     2.5, 1, 1, 3)
        self.Trace_grid.addWidget(self.traceSlider,                 3, 0, 1, 3)        
        self.Trace_grid.addWidget(self.traceindex,                  4.5, 0, 1, 1)
        self.Trace_grid.addWidget(self.traceindexEdit,              4.5, 1, 1, 2)
        self.Trace_grid.addWidget(self.btnShow,                     5, 0, 1, 3)        
        self.Trace_grid.addWidget(self.btnautomatic_detect,         6, 0, 1, 3)
        self.Trace_grid.addWidget(self.btn_import_selection,        7, 0, 1, 3)

#        self.Trace_grid.addWidget(self.StartFrame,                  6, 0, 1, 1)
#        self.Trace_grid.addWidget(self.StartFrameEdit,              6, 1, 1, 2)
#        self.Trace_grid.addWidget(self.EndFrame,                    7, 0, 1, 1)
#        self.Trace_grid.addWidget(self.EndFrameEdit,                7, 1, 1, 2)


        self.Trace_grid.addWidget(threshold_index_Slider,           8, 0, 1, 3)
        self.Trace_grid.addWidget(self.threshold_index_Slider_Edit, 8, 1, 1, 3)
        self.Trace_grid.addWidget(self.thresholdSlider,             9, 0, 1, 3)       
        self.Trace_grid.addWidget(self.btnGoodTrace,                10, 0, 1, 1)
        self.Trace_grid.addWidget(self.btnBadTrace,                 10, 2, 1, 1)
        self.Trace_grid.addWidget(self.btnTonTimes,                 11, 0, 1, 3)
        self.Trace_grid.addWidget(self.btnExport,                   12, 0, 1, 3)
        self.Trace_grid.addWidget(self.graph,                     0, 4, 13, 50)
        self.Trace_grid.addWidget(self.BinaryTrace,               14,0,500, 104)
#        self.Trace_grid.addWidget(self.histo_window,             14,0,40, 50)

        
        self.Trace_grid.addWidget(self.labelmax,                   0,  15, 1, 3)
        self.Trace_grid.addWidget(self.labelstep,                   0, 30, 1, 3)
        self.Trace_grid.addWidget(self.labelmin,                    0, 45, 1, 3)
#        self.Trace_grid.addWidget(self.btnmaxmin,                   11, 0, 1, 3)
        self.Trace_grid.addWidget(self.labelstep2,                  1, 30, 1, 3)


#        # DOCK cosas, mas comodo!
        self.state = None  # defines the docks state (personalize your oun UI!)

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
#
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        dockArea = DockArea()
        self.dockArea = dockArea
        grid.addWidget(self.dockArea)

        TraceDock = Dock('TraceDock', size=(300, 50))
        TraceDock.addWidget(self.Trace_wid)
#        viewDock.hideTitleBar()
        self.dockArea.addDock(TraceDock)

        self.setWindowTitle("Trazer Molecuzer")  # windows name
        self.setGeometry(10, 40, 1600, 800)  # (PosX, PosY, SizeX, SizeY)

#        layout.setColumnStretch(0,1)
#        layout.setColumnStretch(1,10)
     
        # Button actions
        self.btnLoad.clicked.connect(self.importTrace)
        self.btnShow.clicked.connect(self.showTrace)
        self.btnGoodTrace.clicked.connect(self.save_goodSelection_traces)
        self.btnBadTrace.clicked.connect(self.save_badSelection_traces)
        self.btnTonTimes.clicked.connect(self.Calculate_TON_times)
        self.btnExport.clicked.connect(self.exportTraces)

        self.btnautomatic_detect.clicked.connect(self.calculate_threshold)  #  self.step_detection)
        self.btn_import_selection.clicked.connect(self.importselection)
        
#        self.btnmaxmin.clicked.connect(self.calculate_max_min)
#        self.btnmaxmin.clicked.connect(self.make_histogram)

        self.btnmaxmin.clicked.connect(self.Quickanddirtystart)

        # Slider Action
        self.traceSlider.valueChanged.connect(self.update_trace)
        self.thresholdSlider.valueChanged.connect(self.update_threshold)

#        self.traceindexEdit.textEdited.connect(self.showTrace)  # I like it more

        self.Quickanddirtytimer = QtCore.QTimer()
        self.Quickanddirtytimer.timeout.connect(self.Quickanddirty)


        # Shortcut. ESC ==> close_win
        self.close_Action = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('ESC'), self, self.close_win)

        self.good_selection_Action = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('g'), self, self.save_goodSelection_traces)

        self.good_selection_Action2 = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('1'), self, self.save_goodSelection_traces)

        self.bad_selection_Action = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('b'), self, self.save_badSelection_traces)

        self.bad_selection_Action2 = QtGui.QAction(self)
        QtGui.QShortcut(
            QtGui.QKeySequence('0'), self, self.save_badSelection_traces)

    def close_win(self):  # called pressing ESC
        """Close all when press ESC"""
        print("Close win")
        self.close()

    def Quickanddirtystart(self):
        self.timing = 0
        self.Quickanddirtytimer.start(10)  # imput in ms
        print("number of traces to check :", self.data.shape[1])

    def Quickanddirty(self):
        self.step_detection()
        self.calculate_max_min()
        print(self.step_intensity, self.stepintensity)
#            self.make_histogram()
        self.next_trace()
        self.update_trace()
        self.update_threshold()
        self.timing+=1
        if self.timing == self.data.shape[1]-1:
            self.Quickanddirtytimer.stop()
            print(" automatic analysis finished")
        
    def updatelr(self):
        self.lrmax.setZValue(10)
        minX, maxX = self.lrmax.getRegion()
        self.lrmin.setZValue(10)
        minX2, maxX2 = self.lrmin.getRegion()
        self.avgmax = np.nanmean(self.data[int(minX):int(maxX), (int(self.traceSlider.value()))])
        self.avgmin = np.nanmean(self.data[int(minX2):int(maxX2), (int(self.traceSlider.value()))])

        self.selection[int(self.traceSlider.value()), 3] = minX
        self.selection[int(self.traceSlider.value()), 4] = maxX
        self.selection[int(self.traceSlider.value()), 5] = minX2
        self.selection[int(self.traceSlider.value()), 6] = maxX2

        self.stepintensity = (self.avgmax-self.avgmin)
        self.labelmax.setText("<span style='font-size: 12pt'> <span style='color: green'>LeftMean=%0.1f</span>" % (self.avgmax))
        self.labelmin.setText("<span style='font-size: 12pt'> <span style='color: red'>RigthMean=%0.1f</span>" % (self.avgmin))
        self.labelstep.setText("<span style='font-size: 12pt'> <span style='color: blue'>Step=%0.1f</span>" % self.stepintensity)

    def calculate_max_min(self):
#        self.lrmax.setZValue(10)
#        minX, maxX = self.lrmax.getRegion()
#        self.lrmin.setZValue(10)
#        minX2, maxX2 = self.lrmin.getRegion()
#        self.avgmax = np.nanmean(self.data[int(minX):int(maxX), (int(self.traceSlider.value()))])
#        self.avgmin = np.nanmean(self.data[int(minX2):int(maxX2), (int(self.traceSlider.value()))])
#        
#        print(self.avgmax - self.avgmin)
#        print(self.stepintensity)
        self.selection[int(self.traceSlider.value()), 5] = self.step_intensity  # stepintensity


    def importselection(self):
        # Select image from file
        self.selection_file_name = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.txt")))
        self.selection = np.loadtxt(self.selection_file_name)


    # Define Actions    
    def importTrace(self):
        """ data.shape[0] = amount of frames (i.e. 1000 or 10k)
        data.shape[1] = amount of traces (the spot you picked)
"""
        # Remove annoying empty window
        root = Tk()
        root.withdraw()

        # Select image from file
        self.file_name = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.txt")))
        self.data = np.loadtxt(self.file_name)
        self.traceSlider.setMaximum(self.data.shape[1]-1)

        
        self.sum_on_trace = np.zeros(self.data.shape[1])
        self.signal2noise = np.zeros(self.data.shape[1])
        self.intensity = np.zeros(self.data.shape[1])

        self.graph.clear()
        self.selection = np.zeros((self.data.shape[1], 9))  # +color + Steps columns (5 ==> 9)
        self.selection[:,0] = np.arange(0,self.data.shape[1])
#        self.selection[:,4] = self.data.shape[0]
        self.colorgraph = (100, 150, 255)
# =============================================================================
#         self.lr = pg.LinearRegionItem([0,int(self.selection[0,4])], brush=None)
#         self.lr.setZValue(10)
#         self.graph.addItem(self.lr)
# =============================================================================
        
#        starting = int(self.selection[(int(self.traceSlider.value())),3])
#        ending = int(self.selection[(int(self.traceSlider.value())),4])
        starting = 0
        ending = self.data.shape[0]

#        self.selection[:,3] = np.zeros(len(self.selection[:,3]) 
        self.selection[:,4] = np.ones(len(self.selection[:,4]))*((starting+ending)//8)
        self.selection[:,5] =  np.ones(len(self.selection[:,5]))*(ending - ((starting+ending)//4))
        self.selection[:,6] =  np.ones(len(self.selection[:,6]))*ending
        
        self.lrmax = pg.LinearRegionItem([starting,(starting+ending)//8], pen='g',
                                          bounds=[0, self.data.shape[0]],
                                          brush=(5,200,5,25),
                                          hoverBrush=(50,200,50,50))
        self.lrmax.setZValue(10)
        self.graph.addItem(self.lrmax, ignoreBounds=True)
        self.lrmin = pg.LinearRegionItem([ending - ((starting+ending)//4), ending], pen='r',
                                          bounds=[0, self.data.shape[0]],
                                          brush=(200,50,50,25),
                                          hoverBrush=(200,50,50,50))
        self.lrmin.setZValue(10)
        self.graph.addItem(self.lrmin, ignoreBounds=True)
        
        self.graph.plot(self.data[:, 0], pen=pg.mkPen(color=self.colorgraph, width=1))

        # Define initial Threshold
        print('[Initial Threshold Calculation]')
        for i in range(0, self.data.shape[1]):
            short = int(len(self.data[:, i])//4)
            initial_threshold = stats.mode(self.data[-short:, i]) + 5*np.std(self.data[-short:, i])  # there is the initial Threshold
            self.selection[i, 2] = initial_threshold[0]


            fake_trace = self.data[:,i]
            self.sum_on_trace[i] =  np.sum(fake_trace[np.where(fake_trace > initial_threshold[0])])

#        self.another_threshold = self.selection[:,2]

#        self.thresholdSlider.setMaximum(((np.max(self.data[:,int(self.traceindexEdit.text())]))))
#        self.thresholdSlider.setMinimun(((0.5*np.min(self.data[:,int(self.traceindexEdit.text())]))))
        minimo = np.min(self.data[:,int(self.traceindexEdit.text())])
        maximo = np.max(self.data[:,int(self.traceindexEdit.text())])
        self.thresholdSlider.setRange(minimo*10, maximo*10)
#        self.thresholdSlider.setMaximum(50)
        self.thresholdSlider.setValue(int(self.selection[0, 2])*10)
#        self.EndFrameEdit.setText(str(self.data.shape[0]))
        print('[End of Initial Threshold Calculation]')
        print('[File name: ' + self.file_name + ']')

        self.updatelr()

        self.lrmax.sigRegionChanged.connect(self.updatelr)
        self.lrmin.sigRegionChanged.connect(self.updatelr)
    
    # Select a trace to plot    
    def showTrace(self):

        self.graph.clear()
        if self.selection[int(self.traceSlider.value()), 1] == 1:
            self.colorgraph = (120, 220, 50)
        elif self.selection[int(self.traceSlider.value()), 1] == -1:
            self.colorgraph = (250, 150, 50)
        else:
            self.colorgraph = (100, 150, 255)
        self.graph.plot(self.data[:, int(self.traceindexEdit.text())], pen=pg.mkPen(color=self.colorgraph, width=1))
        self.Trace_index_Slider_Edit.setText(format(int(self.traceindexEdit.text())))
        self.traceSlider.setValue(int(self.traceindexEdit.text()))


    # Define update plot with slider    

    # Plot the binary trace of the selected trace    
    def PlotBinaryTrace(self):
        self.BinaryTrace.clear()
        mode = stats.mode(self.data[:, int(self.traceSlider.value())])[0]
#        print("mode=", mode)
        mode=0
        self.BinaryTrace.setLabel('left', "Normalized Intensity")
        self.BinaryTrace.setLabel('bottom', "Frame")
        trace = self.data[:,(int(self.traceSlider.value()))]
#        binary_trace = np.ones(self.data.shape[0])*np.min(trace)*10
        threshold = int(self.thresholdSlider.value())
        threshold_vector = threshold*np.ones(self.data.shape[0])

        self.sum_on_trace[int(self.traceSlider.value())]  = np.sum(trace[np.where(trace > 0.1*threshold)])
#        print("SUM ON TRACE",self.sum_on_trace )

#        binary_trace = np.where(trace < threshold, binary_trace, 1)
#        self.BinaryTrace.plot(binary_trace, pen=pg.mkPen(color=(125 ,50, 150), width=1))
#        self.BinaryTrace.plot((self.data[:,(int(self.traceSlider.value()))]-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=self.colorgraph, width=1))
#        self.BinaryTrace.plot((threshold_vector-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=(255,60,60), width=3))

        binary_trace = np.where(trace*10 < threshold, np.min(trace)*10, np.max(trace)*10)
        self.BinaryTrace.plot(binary_trace, pen=pg.mkPen(color=(125,50,150), width=1))
        self.BinaryTrace.plot(trace*10, pen=pg.mkPen(color=self.colorgraph, width=1))
        self.BinaryTrace.plot((threshold_vector), pen=pg.mkPen(color=(255,60,60), width=3))

    def update_trace(self):
        self.graph.clear()
        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value())))
        if self.selection[int(self.traceSlider.value()), 1] == 1:
            self.colorgraph = (120, 220, 50)
        elif self.selection[int(self.traceSlider.value()), 1] == -1:
            self.colorgraph = (250, 150, 50)
        else:
            self.colorgraph = (100, 150, 255)
        self.graph.plot(self.data[:, (int(self.traceSlider.value()))], pen=pg.mkPen(color=self.colorgraph, width=1))
        self.threshold_index_Slider_Edit.setText(format(int(self.selection[int(self.traceSlider.value()), 2])))
#        self.thresholdSlider.setMaximum(10*(np.max(self.data[:, int(self.traceSlider.value())])))
#        self.thresholdSlider.setMaximum(50)
        
        minimo = np.min(self.data[:, int(self.traceSlider.value())])
        maximo = np.max(self.data[:, int(self.traceSlider.value())])
        self.thresholdSlider.setRange(minimo*10, maximo*10)
#        self.thresholdSlider.setMaximum(50)
        self.thresholdSlider.setValue(10*float(self.selection[int(self.traceSlider.value()), 2]))

#        print("1threshold = ", self.selection[int(self.traceSlider.value()),2])
        self.step_detection()
        self.PlotBinaryTrace()
# =============================================================================
#         self.lr = pg.LinearRegionItem([int(self.selection[(int(self.traceSlider.value())),3]),int(self.selection[(int(self.traceSlider.value())),4])])
#         self.lr.setZValue(10)
#         self.graph.addItem(self.lr)
# =============================================================================
        
        starting = self.selection[int(self.traceSlider.value()), 3]
        ending = self.selection[int(self.traceSlider.value()), 4]


        self.lrmax = pg.LinearRegionItem([starting, ending], pen='g',
                                          bounds=[0, self.data.shape[0]],
                                          brush=(5,200,5,25),
                                          hoverBrush=(50,200,50,50))
        self.lrmax.setZValue(10)
        self.graph.addItem(self.lrmax)

        starting2 = self.selection[int(self.traceSlider.value()), 5]
        ending2 = self.selection[int(self.traceSlider.value()), 6]

        self.lrmin = pg.LinearRegionItem([starting2, ending2], pen='r',
                                          bounds=[0, self.data.shape[0]],
                                          brush=(200,5,5,25),
                                          hoverBrush=(200,50,50,50))
        self.lrmin.setZValue(10)
        self.graph.addItem(self.lrmin)
        
        self.updatelr()
        
        self.lrmax.sigRegionChanged.connect(self.updatelr)
        self.lrmin.sigRegionChanged.connect(self.updatelr)
        
    # Define update Binary Trace plot with slider    
    def update_threshold(self):
        self.BinaryTrace.clear()
        self.threshold_index_Slider_Edit.setText(format(int(self.thresholdSlider.value())))
        mode = stats.mode(self.data[:, int(self.traceSlider.value())])[0]
        mode=0
        trace = self.data[:, (int(self.traceSlider.value()))]
#        new_binary_trace = np.ones(self.data.shape[0])*np.min(trace)
        new_threshold = float(self.thresholdSlider.value())
        new_threshold_vector = new_threshold*np.ones(self.data.shape[0])
#        new_binary_trace = np.where(trace < new_threshold, new_binary_trace, 1)
#        self.BinaryTrace.plot(new_binary_trace,pen=pg.mkPen(color=(125,50,150), width=1))
#        self.BinaryTrace.plot((self.data[:,(int(self.traceSlider.value()))]-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=self.colorgraph, width=1))
#        self.BinaryTrace.plot((new_threshold_vector-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=(255,60,60), width=3))

        new_binary_trace = np.where(trace*10 < new_threshold, np.min(trace)*10, np.max(trace)*10)
        self.BinaryTrace.plot(new_binary_trace, pen=pg.mkPen(color=(125,50,150), width=1))
        self.BinaryTrace.plot(trace*10, pen=pg.mkPen(color=self.colorgraph, width=1))
        self.BinaryTrace.plot((new_threshold_vector), pen=pg.mkPen(color=(255,60,60), width=3))

        self.sum_on_trace[int(self.traceSlider.value())] = np.sum(trace[np.where(trace > 0.1*new_threshold)])
#        print("SUM ON TRACE", self.sum_on_trace[int(self.traceSlider.value())] )
#        print("2threshold = ", self.selection[int(self.traceSlider.value()),2])

        meanON = np.mean(trace[np.where(trace > 0.1*new_threshold)])
        self.signal2noise[int(self.traceSlider.value())] = meanON / np.std(trace[np.where(trace > 0.1*new_threshold)])
#        a = np.asanyarray(trace[np.where(trace > 0.1*new_threshold)])
#        m = a.mean(0)
#        sd = a.std(axis=0, ddof=0)  # TODO: usar std no la varianza

#        self.signal2noise[int(self.traceSlider.value())] = np.where(sd == 0, 0, m/sd)
        self.intensity[int(self.traceSlider.value())] = meanON

    def step_detection(self):

#        threshold = self.another_threshold[int(self.traceSlider.value())]
        threshold = int(self.thresholdSlider.value())*0.1
#        print("another_threshold", threshold)
        self.threshold_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(color=(255,60,60), width=2))
        self.threshold_line.setPos(threshold)
        self.graph.addItem(self.threshold_line)
        aux = self.data[:,(int(self.traceSlider.value()))]
        self.step_intensity = np.mean(aux[np.where(aux>threshold)]) - np.mean(aux[np.where(aux<threshold)])
        self.labelstep2.setText("<span style='font-size: 12pt'> <span style='color: black'>Step2=%0.1f</span>" % self.step_intensity)

        self.threshold_line.sigPositionChangeFinished.connect(self.moving_threshold)

    def moving_threshold(self):
        print("pos", self.threshold_line.pos()[1])
        self.thresholdSlider.setValue(int(self.threshold_line.pos()[1]*10))
        threshold = int(self.thresholdSlider.value())*0.1
        aux = self.data[:,(int(self.traceSlider.value()))]
        self.step_intensity = np.mean(aux[np.where(aux>threshold)]) - np.mean(aux[np.where(aux<threshold)])

#        self.selection[int(self.traceSlider.value()), 2] = int(self.thresholdSlider.value())
        self.labelstep2.setText("<span style='font-size: 12pt'> <span style='color: black'>Step2=%0.1f</span>" % self.step_intensity)


    # Next trace when you touch good or bad trace button    
    def next_trace(self):
#        self.make_histogram()

        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value()) + 1))
        self.traceSlider.setValue(int(self.traceSlider.value()) + 1)
        
    # Good Trace Button Action    
    def save_goodSelection_traces(self):
        self.selection[int(self.traceSlider.value()), 1] = 1
        self.selection[int(self.traceSlider.value()), 2] = 0.1*int(self.thresholdSlider.value())

        self.selection[int(self.traceSlider.value()), 3] = self.signal2noise[int(self.traceSlider.value())]
        self.selection[int(self.traceSlider.value()), 4] = self.intensity[int(self.traceSlider.value())]
# =============================================================================
#         self.selection[int(self.traceSlider.value()), 3] = int(self.lr.getRegion()[0])
#         self.selection[int(self.traceSlider.value()), 4] = int(self.lr.getRegion()[1])
# =============================================================================

        self.selection[int(self.traceSlider.value()), 5] = self.stepintensity

        print("GOODselection traces")
#        print(self.selection)
#        print("\n")
#        print(self.selection[int(self.traceSlider.value())-1:int(self.traceSlider.value())+2])

        self.next_trace()
        self.update_trace()
        self.update_threshold()


    # Bad Trace Button Action   
    def save_badSelection_traces(self):
        self.selection[int(self.traceSlider.value()), 1] = -1
        self.colorgraph = (250, 150, 50)

        self.selection[int(self.traceSlider.value()), 2] = 0
        self.selection[int(self.traceSlider.value()), 5] = 0
        print("BADselection traces")
#        print(self.selection[int(self.traceSlider.value())-1:int(self.traceSlider.value())+2])
        self.next_trace()
        self.update_trace()
        self.update_threshold()


    # Calculate On times
    def Calculate_TON_times(self):

        
        
        self.times_frames_total_on = np.zeros(1, dtype = int)
        self.times_frames_total_off = np.zeros(1, dtype = int)
        Exposure_time = int(self.ExposureTimeEdit.text())
        
        for j in range(0, int(self.selection.shape[0])):
            if int(self.selection[j, 1]) == 1:
                trace = self.data[:, int(self.selection[j, 0])]
                threshold = self.selection[j, 2]
                binary_trace = np.zeros(int(self.data.shape[0]), dtype = int)
                binary_trace = np.where(trace < threshold, binary_trace, 1)
                diff_binary_trace = np.diff(binary_trace)
                indexes = np.argwhere(np.diff(binary_trace)).squeeze()
                print("indexes:", indexes)
                print("\n sum", np.sum(diff_binary_trace))
                print("\n ==1?", diff_binary_trace[indexes])
                try:
                    number = int(len(indexes))
                    print("\n number",number)
                    times_frames_on = np.zeros(number//2, dtype = int)
                    times_frames_off = np.zeros(number//2, dtype = int)
                    c_on = 0 #to count
                    c_off = 0 
                    for n in np.arange(0,number,2):
                        if np.sum(diff_binary_trace) == 0: #case 1
                            if diff_binary_trace[indexes[0]] == 1:
                                times_frames_on[c_on] = indexes[n+1] - indexes[n]
                                c_on += 1
                                if n > 0:
                                    times_frames_off[c_off] = indexes[n] - indexes[n-1]
                                    c_off += 1
                            else: #case 2
                                times_frames_off[c_off] = indexes[n+1] - indexes[n]
                                c_off += 1
                                if n > 0:
                                    times_frames_on[c_on] = indexes[n] - indexes[n-1]
                                    c_on += 1
                        else: #case 3
                            if diff_binary_trace[indexes[0]] == 1:
                                if n > 0:
                                    times_frames_off[c_off] = indexes[n] - indexes[n-1]
                                    c_off += 1
                                if n != number - 1: #porque tira error al final1
                                    times_frames_on[c_on] = indexes[n+1] - indexes[n]
                                    c_on += 1
                            else: #case 4
                                if n != number - 1: #porque tira error al final1
                                    times_frames_off[c_off] = indexes[n+1] - indexes[n]
                                    c_off += 1
                                if n > 0: 
                                    times_frames_on[c_on] = indexes[n] - indexes[n-1]
                                    c_on += 1
                    times_frames_on = np.trim_zeros(times_frames_on)
                    times_frames_off = np.trim_zeros(times_frames_off)
                except TypeError:
                    print("Excepticon")
                    times_frames_on = 0
                    times_frames_off = 0
                    if diff_binary_trace[indexes] == -1:
                        times_frames_on = indexes
                    else:
                        times_frames_off = indexes

#                self.times_frames_total_on = np.append(self.times_frames_total_on, times_frames_on)
#                self.times_frames_total_off = np.append(self.times_frames_total_off, times_frames_off)
                self.times_frames_total_on = np.append(self.times_frames_total_on, np.sum(times_frames_on))
                self.times_frames_total_off = np.append(self.times_frames_total_off, np.sum(times_frames_off))

        self.times_frames_total_on = np.trim_zeros(self.times_frames_total_on)
        self.times_frames_total_off = np.trim_zeros(self.times_frames_total_off)
        self.times_frames_total_on = self.times_frames_total_on*Exposure_time
        self.times_frames_total_off = self.times_frames_total_off*Exposure_time
        print('[Ton and Toff Calculation finished]')

    def calculate_threshold(self):

        for i in range(len(self.selection[:, 2])):
            distance = []
            step = 2
            trace = self.data[:, i]*10
            new_binary_trace = np.zeros(self.data.shape[0])
            old_threshold = 10 + float(self.selection[i, 2])

#            converge = False
#            while converge == False:
            for j in range(int(np.max(trace)//step)):
#                mean_off_trace = np.mean(trace[np.where(trace < old_threshold)])
                new_binary_trace = np.where(trace < old_threshold, new_binary_trace, np.max(trace))
                indexes = np.argwhere(np.diff(new_binary_trace)).squeeze()
#                if indexes.size > 10:  # PUEDE SER UNA VARIABLE EN LA UI
#                    if mean_off_trace < np.mean(trace[-20:]):
#                        old_threshold +=1
#                    elif mean_off_trace > np.mean(trace[-20:]):
#                        old_threshold -=1
#                else:
#                    converge = True
#                    print("CONVERGE at ", old_threshold)
#                    break
#                j+=1
                distance.append(np.mean(np.diff(indexes)))
                old_threshold += step
                
            the_i = np.where(np.array(distance)==np.max(np.nan_to_num(distance)))
            print((the_i), "the_i")
            old_threshold = 10 + step*int(np.mean(the_i)-2)
            new_binary_trace = np.where(trace< old_threshold, new_binary_trace, np.max(trace))

            self.selection[i, 2] = float(old_threshold/10)




#    def make_histogram(self):
#        try:
#            self.histo_window.removeItem(self.plt1)
#        except:
#            print("Create the histogram")
#
#        vals = self.selection[:, 5] / float(self.ExposureTimeEdit.text())
#        self.plt1 = self.histo_window.addPlot(title="Histogram (kHz)")
#        y,x = np.histogram(vals[np.nonzero(vals)])
#        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
#        self.plt1.showGrid(x = True, y = True, alpha = 0.5)
#        self.plt1.setLabel(axis="bottom",
#                    text='kHz, {} points'.format(len(vals[np.nonzero(vals)])))

    # Define export selection of traces       
    def exportTraces(self):
        count_good_traces = np.count_nonzero(self.selection[:, 1] == 1)
        filtered_traces = np.zeros((self.data.shape[0], count_good_traces))
        k_good = 0

        filtered_sum_on = []

        for k in range(0, self.selection.shape[0]):
            if self.selection[k, 1] == 1:
                filtered_traces[:, k_good] = self.data[:, k]
                filtered_sum_on.append(self.sum_on_trace[k])
                k_good += 1

        folder = os.path.dirname(self.file_name)
        file_traces_name = os.path.basename(self.file_name)
        np.savetxt(folder+'/FILTERED_'+file_traces_name, filtered_traces)
        amount_goodTraces = (np.count_nonzero(self.selection[:, 1] == 1)/int(self.data.shape[1]))*100
        print('[Filtered Traces Saved]: Amount of Good Traces: '+str(amount_goodTraces)[0:3]+'%')

        header = "Traces"+"    "+"Good(1)_bad(-1)"+"    "+"Threshold"+"    "+"Signal2noise"+"    "+"Intensity"+"    "+"step"

        np.savetxt(folder+'/selection_'+file_traces_name, self.selection, header=header)
        print("[selection saved]", folder+'/selection_'+file_traces_name)
#        print(self.selection)
        try:
            np.savetxt(folder+'/ON_TIMES_'+file_traces_name,self.times_frames_total_on)
            np.savetxt(folder+'/OFF_TIMES_'+file_traces_name,self.times_frames_total_off)
            print('and, [Ton and Toff saved]')
        except:
            pass

        np.savetxt(folder+'/SUM_COUNTS_ON_'+file_traces_name, filtered_sum_on)
        print('Sum_counts_on saved')
        
if __name__ == '__main__':

    app = pg.Qt.QtGui.QApplication([])
    exe = Trace_Inspector()
    exe.show()
    app.exec_()