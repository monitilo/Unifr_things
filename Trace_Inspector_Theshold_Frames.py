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
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import os
from tkinter import Tk, filedialog
#from skimage.feature import peak_local_max
from scipy import stats


class Trace_Inspector(pg.Qt.QtGui.QMainWindow):

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
        self.w.resize(1500, 1000)

        # Create ImagePlot
        self.graph = pg.PlotWidget()
        self.BinaryTrace = pg.PlotWidget()

        # Create buttons
        self.btnLoad = QtGui.QPushButton('Load Traces')
        self.btnShow = QtGui.QPushButton('Show Trace')
        self.btnGoodTrace = QtGui.QPushButton('Good Trace')
        self.btnBadTrace = QtGui.QPushButton('Bad Trace')
        self.btnTonTimes = QtGui.QPushButton('Calculate Ton and Toff')
        self.btnExport = QtGui.QPushButton('Export Trace Selection and T')


        # Create parameter fields with labels
        self.traceindex = QtGui.QLabel('Show Trace:')
        self.traceindexEdit = QtGui.QLineEdit('0')
        self.ExposureTime = QtGui.QLabel('Exposure Time [ms]')
        self.ExposureTimeEdit = QtGui.QLineEdit()
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
        self.Trace_index_Slider_Edit.setText(str(int(self.traceSlider.value())))
        
        # Create Slicer for the threshold
        self.thresholdSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.thresholdSlider.setMinimum(0 )
        self.thresholdSlider.setMaximum(1)
        self.thresholdSlider.setValue(0)
        self.thresholdSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.thresholdSlider.setTickInterval(1)
       
        threshold_index_Slider = QtGui.QLabel('Threshold Slider:')
        self.threshold_index_Slider_Edit = QtGui.QLabel('0')
        self.threshold_index_Slider_Edit.setText(format(int(self.thresholdSlider.value())))
        
        # Final message after saving data
        self.Amount_goodTraces_text = QtGui.QLabel('Good traces selection:')
        self.Amount_goodTraces = QtGui.QLabel() 
        
        # Create a grid layout to manage the widgets size and position
        layout = QtGui.QGridLayout()
        self.w.setLayout(layout)

        # Add widgets to the layout in their proper positions
        layout.addWidget(self.btnLoad,                     0, 0, 1, 3)
        layout.addWidget(self.ExposureTime,                1, 0, 1, 1)
        layout.addWidget(self.ExposureTimeEdit,            1, 1, 1, 2)
        layout.addWidget(Trace_index_Slider,               2.5, 0, 1, 3)
        layout.addWidget(self.Trace_index_Slider_Edit,     2.5, 1, 1, 3)
        layout.addWidget(self.traceSlider,                 3, 0, 1, 3)        
        layout.addWidget(self.traceindex,                  4.5, 0, 1, 1)
        layout.addWidget(self.traceindexEdit,              4.5, 1, 1, 2)
        layout.addWidget(self.btnShow,                     5, 0, 1, 3)        
        
#        layout.addWidget(self.StartFrame,                  6, 0, 1, 1)
#        layout.addWidget(self.StartFrameEdit,              6, 1, 1, 2)
#        layout.addWidget(self.EndFrame,                    7, 0, 1, 1)
#        layout.addWidget(self.EndFrameEdit,                7, 1, 1, 2)


        layout.addWidget(threshold_index_Slider,           8, 0, 1, 3)
        layout.addWidget(self.threshold_index_Slider_Edit, 8, 1, 1, 3)
        layout.addWidget(self.thresholdSlider,             9, 0, 1, 3)       
        layout.addWidget(self.btnGoodTrace,                10, 0, 1, 1)
        layout.addWidget(self.btnBadTrace,                 10, 2, 1, 1)
        layout.addWidget(self.btnTonTimes,                 11, 0, 1, 3)
        layout.addWidget(self.btnExport,                   12, 0, 1, 3)
        layout.addWidget(self.graph,                       0, 4, 13, 100)
        layout.addWidget(self.BinaryTrace,                 14,0,500, 104)
        
#        layout.setColumnStretch(0,1)
#        layout.setColumnStretch(1,10)
     
        # Button actions
        self.btnLoad.clicked.connect(self.importTrace)
        self.btnShow.clicked.connect(self.showTrace)
        self.btnGoodTrace.clicked.connect(self.save_goodSelection_traces)
        self.btnBadTrace.clicked.connect(self.save_badSelection_traces)
        self.btnTonTimes.clicked.connect(self.Calculate_TON_times)
        self.btnExport.clicked.connect(self.exportTraces)
        
        # Slider Action
        self.traceSlider.valueChanged.connect(self.update_trace)
        self.thresholdSlider.valueChanged.connect(self.update_threshold)


    # Define Actions    
    def importTrace(self):

        # Remove annoying empty window
        root = Tk()
        root.withdraw()

        # Select image from file
        self.file_name = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.txt")))
        self.data = np.loadtxt(self.file_name)
        self.traceSlider.setMaximum(self.data.shape[1])
        self.graph.clear()
        self.selection = np.zeros((self.data.shape[1], 5), dtype = int)
        self.selection[:,0] = np.arange(0,self.data.shape[1])
        self.selection[:,4] = self.data.shape[0]
        self.colorgraph = (100, 150, 255)
        self.lr = pg.LinearRegionItem([0,int(self.selection[0,4])],
                                       bounds=[0, int(self.selection[0,4])],
                                       brush=None)
        self.lr.setZValue(10)
        self.graph.addItem(self.lr, ignoreBounds=True)
        self.graph.plot(self.data[:, 0], pen=pg.mkPen(color=self.colorgraph, width=1))
        # Define initial Threshold
        print('[Initial Threshold Calculation]')
        for i in range(0, self.data.shape[1]):
            initial_threshold = stats.mode(self.data[:, i]) + np.std(self.data[:, i])
            self.selection[i, 2] = initial_threshold[0]
        self.thresholdSlider.setMaximum(((np.max(self.data[:,int(self.traceindexEdit.text())]))))
        self.thresholdSlider.setValue(int(self.selection[0, 2]))
#        self.EndFrameEdit.setText(str(self.data.shape[0]))
        print('[End of Initial Threshold Calculation]')
        print('[File name: ' + self.file_name + ']')

    
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
        self.BinaryTrace.setLabel('left', "Normalized Intensity")
        self.BinaryTrace.setLabel('bottom', "Frame")
        binary_trace = np.zeros(self.data.shape[0])
        trace = self.data[:,(int(self.traceSlider.value()))]
        threshold = int(self.thresholdSlider.value())
        threshold_vector = threshold*np.ones(self.data.shape[0])
        binary_trace = np.where(trace < threshold, binary_trace, 1)
        self.BinaryTrace.plot(binary_trace, pen=pg.mkPen(color=(125 ,50, 150), width=1))
        self.BinaryTrace.plot((self.data[:,(int(self.traceSlider.value()))]-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=self.colorgraph, width=1))
        self.BinaryTrace.plot((threshold_vector-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=(255,60,60), width=3))
 
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
        self.thresholdSlider.setMaximum((np.max(self.data[:, int(self.traceSlider.value())])))
        self.thresholdSlider.setValue(int(self.selection[int(self.traceSlider.value()), 2]))
        self.PlotBinaryTrace()
        self.lr = pg.LinearRegionItem([int(self.selection[(int(self.traceSlider.value())),3]),int(self.selection[(int(self.traceSlider.value())),4])],
                                       bounds=[0, int(self.selection[0,4])],
                                       brush=None)
        self.lr.setZValue(10)
        self.graph.addItem(self.lr, ignoreBounds=True)


    # Define update Binary Trace plot with slider    
    def update_threshold(self):
        self.BinaryTrace.clear()
        self.threshold_index_Slider_Edit.setText(format(int(self.thresholdSlider.value())))
        mode = stats.mode(self.data[:, int(self.traceSlider.value())])[0]
        trace = self.data[:, (int(self.traceSlider.value()))]
        new_binary_trace = np.zeros(self.data.shape[0])
        new_threshold = int(self.thresholdSlider.value())
        new_threshold_vector = new_threshold*np.ones(self.data.shape[0])
        new_binary_trace = np.where(trace < new_threshold, new_binary_trace, 1)
        self.BinaryTrace.plot(new_binary_trace,pen=pg.mkPen(color=(125,50,150), width=1))
        self.BinaryTrace.plot((self.data[:,(int(self.traceSlider.value()))]-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=self.colorgraph, width=1))
        self.BinaryTrace.plot((new_threshold_vector-mode)/np.max(self.data[:,(int(self.traceSlider.value()))]-mode), pen=pg.mkPen(color=(255,60,60), width=3))
         
    # Next trace when you touch good or bad trace button    
    def next_trace(self):
        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value()) + 1))
        self.traceSlider.setValue(int(self.traceSlider.value()) + 1)
        
    # Good Trace Button Action    
    def save_goodSelection_traces(self):
        self.selection[int(self.traceSlider.value()), 1] = 1
        self.selection[int(self.traceSlider.value()), 2] = int(self.thresholdSlider.value())
        self.selection[int(self.traceSlider.value()), 3] = int(self.lr.getRegion()[0])
        self.selection[int(self.traceSlider.value()), 4] = int(self.lr.getRegion()[1])
        print(self.selection)
        self.next_trace()
        self.update_trace()
        self.update_threshold()
        
        

    # Bad Trace Button Action   
    def save_badSelection_traces(self):
        self.selection[int(self.traceSlider.value()), 1] = -1
        self.colorgraph = (250, 150, 50)
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
        
    # Define export selection of traces       
    def exportTraces(self):
        count_good_traces = np.count_nonzero(self.selection[:, 1] == 1)
        filtered_traces = np.zeros((self.data.shape[0], count_good_traces))
        k_good = 0

        for k in range(0, self.selection.shape[0]):
            if self.selection[k, 1] == 1:
                filtered_traces[:, k_good] = self.data[:, k]
                k_good += 1
        
        folder = os.path.dirname(self.file_name)
        file_traces_name = os.path.basename(self.file_name)
        np.savetxt(folder+'/FILTERED_'+file_traces_name, filtered_traces)
        amount_goodTraces = (np.count_nonzero(self.selection[:, 1] == 1)/int(self.data.shape[1]))*100
        print('[Filtered Traces Saved]: Amount of Good Traces: '+str(amount_goodTraces)[0:3]+'%')
        np.savetxt(folder+'/ON_TIMES_'+file_traces_name,self.times_frames_total_on)
        np.savetxt(folder+'/OFF_TIMES_'+file_traces_name,self.times_frames_total_off)
        np.savetxt(folder+'/selection_'+file_traces_name,self.selection)
        print(self.selection)
        print('[Ton and Toff saved]')

        
if __name__ == '__main__':

    app = pg.Qt.QtGui.QApplication([])
    exe = Trace_Inspector()
    exe.w.show()
    app.exec_()