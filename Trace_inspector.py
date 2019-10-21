# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:30:48 2019

@author: Cecilia Zaza

Este programa es una interfaz usuario que permite inspeccionar y seleccionar 
trazas, o alguna señal 1D desde un archivo .txt.
Cada columna del archivo txt, corresponde a una traza o set de datos
-Tiene un Slider para inspeccionar toda la data
-Un campo editable para mirar una traza o set de datos especifíco
-Los botones good Trace y bad Trace, sirven para decidir si se quiere trabajar 
con esa traza o set de datos que se encuentra graficada
-Al exportar, se va a indicar el porcentaje de trazas buenas seleccionadas y
se guarda un archivo .txt con las trazas "good traces" cuyo nombre empieza con
"FILTERED_" y sigue con el nombre del archivo de trazas seleccionado.
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import os
from skimage import io
from tkinter import Tk, filedialog
from skimage.feature import peak_local_max


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
#        self.w.resize(1300, 800)

        # Create ImagePlot
        self.graph = pg.PlotWidget()

        # Create buttons
        self.btnLoad = QtGui.QPushButton('Load Traces')
        self.btnShow = QtGui.QPushButton('Show Trace')
        self.btnGoodTrace = QtGui.QPushButton('Good Trace')
        self.btnBadTrace = QtGui.QPushButton('Bad Trace')
        self.btnExport = QtGui.QPushButton('Export Trace Selection')

        # Create parameter fields with labels
        self.traceindex = QtGui.QLabel('Show Trace:')
        self.traceindexEdit = QtGui.QLineEdit('0')
#        self.meanStartLabel = QtGui.QLabel('Start frame:')
#        self.meanStartEdit = QtGui.QLineEdit('0')
#        self.meanEndLabel = QtGui.QLabel('End frame:')
#        self.meanEndEdit = QtGui.QLineEdit()
     
        # Create Slider
        self.traceSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.traceSlider.setMinimum(0 )
        self.traceSlider.setMaximum(1)
        self.traceSlider.setValue(0)
        self.traceSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.traceSlider.setTickInterval(1)
       
        Trace_index_Slider = QtGui.QLabel('Trace index:')
        self.Trace_index_Slider_Edit = QtGui.QLabel('0')
        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value())))
        self.Amount_goodTraces_text = QtGui.QLabel('Good traces selection:')
        self.Amount_goodTraces = QtGui.QLabel()
        
        # Create a grid layout to manage the widgets size and position
        self.layout = QtGui.QGridLayout()
        self.w.setLayout(self.layout)

        # Add widgets to the layout in their proper positions
        self.layout.addWidget(self.btnLoad,                 0, 0, 1, 3)
        self.layout.addWidget(Trace_index_Slider,           1, 0, 1, 3)
        self.layout.addWidget(self.Trace_index_Slider_Edit, 1, 1, 1, 3)
        self.layout.addWidget(self.traceSlider,             2, 0, 1, 3)
        self.layout.addWidget(self.traceindex,              4, 0, 1, 1)
        self.layout.addWidget(self.traceindexEdit,          4, 1, 1, 2)
        self.layout.addWidget(self.btnShow,                 5, 0, 1, 3)
        self.layout.addWidget(self.btnGoodTrace,            7, 0, 1, 1)
        self.layout.addWidget(self.btnBadTrace,             7, 2, 1, 1)
        self.layout.addWidget(self.btnExport,               8, 0, 1, 3)
        self.layout.addWidget(self.graph,                   0, 4, 10, 10)
     
        # Button actions
        self.btnLoad.clicked.connect(self.importTrace)
        self.btnShow.clicked.connect(self.showTrace)
        self.btnGoodTrace.clicked.connect(self.save_goodSelection_traces)
        self.btnBadTrace.clicked.connect(self.save_badSelection_traces)
        self.btnExport.clicked.connect(self.exportTraces)
        
        # Slider Action
        self.traceSlider.valueChanged.connect(self.update_trace)

        
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
        self.selection = np.zeros((self.data.shape[1],4))
        self.selection[:,0] = np.arange(0,self.data.shape[1])
        self.selection[:,3] = self.data.shape[0]
        self.colorgraph = (100,150,255)
        self.graph.plot(self.data[:,0], pen=pg.mkPen(color=self.colorgraph, width=1))
       
    # Select a trace to plot    
    def showTrace(self):

        self.graph.clear()
        if self.selection[int(self.traceSlider.value()), 1] == 1:
            self.colorgraph = (120,220,50)
        elif self.selection[int(self.traceSlider.value()), 1] == -1:
            self.colorgraph = (250,150,50)
        else:
            self.colorgraph = (100,150,255)
        self.graph.plot(self.data[:,int(self.traceindexEdit.text())], pen=pg.mkPen(color=self.colorgraph, width=1))
        self.Trace_index_Slider_Edit.setText(format(int(self.traceindexEdit.text())))
        self.traceSlider.setValue(int(self.traceindexEdit.text()))
        
    # Define update plot with slider    
    def update_trace(self):
        
        self.graph.clear()
        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value())))
        if self.selection[int(self.traceSlider.value()), 1] == 1:
            self.colorgraph = (120,220,50)
        elif self.selection[int(self.traceSlider.value()), 1] == -1:
            self.colorgraph = (250,150,50)
        else:
            self.colorgraph = (100,150,255)
        self.graph.plot(self.data[:,(int(self.traceSlider.value()))], pen=pg.mkPen(color=self.colorgraph, width=1))
        
    # Next trace when you touch good or bad trace button    
    def next_trace(self):
        self.Trace_index_Slider_Edit.setText(format(int(self.traceSlider.value()) + 1))
        self.traceSlider.setValue(int(self.traceSlider.value()) + 1)
        
    # Good Trace Button Action    
    def save_goodSelection_traces(self):
        self.selection[int(self.traceSlider.value()),1] = 1
        self.next_trace()
        self.update_trace()
        
    # Bad Trace Button Action   
    def save_badSelection_traces(self):
        self.selection[int(self.traceSlider.value()),1] = -1
        self.colorgraph = (250,150,50)
        self.next_trace()
        self.update_trace()
        
    # Define export selection of traces       
    def exportTraces(self):
        count_good_traces = np.count_nonzero(self.selection[:,1] == 1)
        filtered_traces = np.zeros((self.data.shape[0],count_good_traces))
        k_good = 0

        for k in range(0,self.selection.shape[0]):
            if self.selection[k,1] == 1:
                filtered_traces[:,k_good] = self.data[:,k]
                k_good += 1
        
        folder = os.path.dirname(self.file_name)
        file_traces_name = os.path.basename(self.file_name)
        np.savetxt(folder+'/FILTERED_'+file_traces_name,filtered_traces)
        amount_goodTraces = (np.count_nonzero(self.selection[:,1] == 1)/int(self.data.shape[1]))*100
        self.layout.addWidget(self.Amount_goodTraces_text,           9, 0, 1, 3)
        self.layout.addWidget(self.Amount_goodTraces,                9, 2, 1, 3)
        self.Amount_goodTraces.setText(str(amount_goodTraces)[0:3]+'%')
        
if __name__ == '__main__':

    app = pg.Qt.QtGui.QApplication([])
    exe = Trace_Inspector()
    exe.w.show()
    app.exec_()