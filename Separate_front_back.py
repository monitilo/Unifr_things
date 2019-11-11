# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:05:47 2019

@author: ChiarelG
"""
#import numpy as np
import os
#import time
#from tkinter import filedialog
#import tkinter as tk
#from datetime import datetime

from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from pyqtgraph.dockarea import DockArea, Dock

import Shutters
import Analizer

class Frontend(QtGui.QMainWindow):
    
    selectDirSignal = pyqtSignal()
    createDirSignal = pyqtSignal()
    openDirSignal = pyqtSignal()
    loadpositionSignal = pyqtSignal()
    closeSignal = pyqtSignal()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('Probanding')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        self.setGeometry(30, 30, 50, 50)

        # Create de file location
        localDirAction = QtGui.QAction('&Select Dir (Ctrl+A)', self)
        localDirAction.setStatusTip('Select the work folder')
        localDirAction.triggered.connect(self.get_selectDir)

        # Create de create daily directory
        dailyAction = QtGui.QAction('&Create daily Dir (Ctrl+S)', self)
        dailyAction.setStatusTip('Create the work folder')
        dailyAction.triggered.connect(self.get_create_daily_directory)
        
        # Open directory
        openAction = QtGui.QAction('&Open Dir (Ctrl+D)', self)
        openAction.setStatusTip('Open document')
        openAction.triggered.connect(self.get_openDir)
        
        # Load las position
        load_position_Action = QtGui.QAction('&Load Last position', self)
        load_position_Action.setStatusTip('Load last position when PyPrinting closed.')
        load_position_Action.triggered.connect(self.load_last_position)
        
        QtGui.QShortcut(
            QtGui.QKeySequence('Ctrl+A'), self, self.get_selectDir)
       
        QtGui.QShortcut(
            QtGui.QKeySequence('Ctrl+S'), self, self.get_create_daily_directory)
         
        QtGui.QShortcut(
            QtGui.QKeySequence('Ctrl+D'), self, self.get_openDir)

    # Create de create daily directory action
        save_docks_Action = QtGui.QAction(QtGui.QIcon('algo.png'), '&Save Docks', self)
        save_docks_Action.setStatusTip('Saves the Actual Docks configuration')
        save_docks_Action.triggered.connect(self.save_docks)

    # Create de create daily directory action
        load_docks_Action = QtGui.QAction(QtGui.QIcon('algo.png'), '&Restore Docks', self)
        load_docks_Action.setStatusTip('Load a previous Docks configuration')
        load_docks_Action.triggered.connect(self.load_docks)
        
        
    # Measurment Printing
        printing_Action = QtGui.QAction('&Do Printing', self)
        printing_Action.triggered.connect(self.measurement_printing)
        
            
    # Measurment Dimers
        dimers_Action = QtGui.QAction('&Do Dimers', self)
        dimers_Action.triggered.connect(self.measurement_dimers)


        # Actions in menubar
    
        menubar = self.menuBar()

        fileMenu = menubar.addMenu('&Files Direction')
        fileMenu.addAction(localDirAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(dailyAction)
        fileMenu.addAction(load_position_Action)

        fileMenu2 = menubar.addMenu('&Measurements')
        fileMenu2.addAction(printing_Action)
        fileMenu2.addAction(dimers_Action)
        
        fileMenu3 = menubar.addMenu('&Docks config')
        fileMenu3.addAction(save_docks_Action)
        fileMenu3.addAction(load_docks_Action)
        
        # GUI layout

        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        # Dock Area

        dockArea = DockArea()
        self.dockArea = dockArea
        grid.addWidget(self.dockArea)

        ## Shutters

        shuttersDock = Dock("Shutters and Flipper")
        self.shuttersWidget = Shutters.Frontend()
        shuttersDock.addWidget(self.shuttersWidget)
        self.dockArea.addDock(shuttersDock)

        ## Shutters

        analizerDock = Dock("SM analizer")
        self.analizerWidget = Analizer.Frontend()
        analizerDock.addWidget(self.analizerWidget)
        self.dockArea.addDock(analizerDock, 'right', shuttersDock)

    def get_openDir(self):
        self.openDirSignal.emit()
        
    def get_selectDir(self):
        self.selectDirSignal.emit()
        
    def get_create_daily_directory(self):
        self.createDirSignal.emit()
        
    def load_last_position(self):
        self.loadpositionSignal.emit()

    def save_docks(self):  # Funciones para acomodar los Docks
        self.state = self.dockArea.saveState()

    def load_docks(self):
        self.dockArea.restoreState(self.state)
        
    def measurement_printing(self):
        
        self.printingWidget.show()
        
    def measurement_dimers(self):
        
        self.dimersWidget.show()
         
#    def make_connection(self, backend):
#        
#        backend.focusWorker.make_connection(self.focusWidget)
#        backend.shuttersWorker.make_connection(self.shuttersWidget)
#        backend.nanopositioningWorker.make_connection(self.nanopositioningWidget)
#        backend.traceWorker.make_connection(self.traceWidget)
#        backend.confocalWorker.make_connection(self.confocalWidget)
#        backend.printingWorker.make_connection(self.printingWidget)
#        backend.dimersWorker.make_connection(self.dimersWidget)
        
#    def closeEvent(self, event):
#
#        reply = QtGui.QMessageBox.question(self, 'Quit', 'Are you sure to quit?',
#                                           QtGui.QMessageBox.No |
#                                           QtGui.QMessageBox.Yes)
#        if reply == QtGui.QMessageBox.Yes:
#            print("PyPrinting Close")
#            self.closeSignal.emit()
#            event.accept()
#            self.close()
#            multipleThread.exit()
#
#        else:
#            event.ignore()
#            print("NO")

class Backend(QtCore.QObject):
    
    fileSignal = pyqtSignal(str)
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.shuttersWorker = Shutters.Backend()
        
        self.file_path = os.path.abspath("C:\Julian\Data_PyPrinting")  #por default, por si se olvida de crear la carpeta del día
        
    @pyqtSlot()    
    def selectDir(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askdirectory()
        if not file_path:
            print("No elegiste nada")
        else:
            self.file_path = file_path
            self.fileSignal.emit(self.file_path)   #Lo reciben los módulos de traza, confocal y printing
             
    @pyqtSlot()  
    def openDir(self):
        os.startfile(self.file_path)
        print('Open: ', self.file_path)
        
    @pyqtSlot()      
    def create_daily_directory(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askdirectory()
        if not file_path:
            print("No elegiste nada ==> No crea la carpeta")
        else:
            timestr = time.strftime("%Y-%m-%d")  # -%H%M%S")

            newpath = file_path + "/" + timestr
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                print("Carpeta creada!")
            else:
                print("Ya existe esa carpeta")

            self.file_path = newpath 
            self.fileSignal.emit(self.file_path) 
            
            
    @pyqtSlot()             
    def load_last_position(self): 
        
        filepath = "C:/Users/CibionPC/Desktop/PyPrinting"
        name = str(filepath  + "/" + "Last_position.txt")
     
        last_position = np.loadtxt(name)
        print(last_position)
        
        targets = list(last_position)
                
        pi_device.MOV(['A', 'B', 'C'], targets)
        time.sleep(0.01)
     
            
    @pyqtSlot()
    def close_all(self):
        
        pos = pi_device.qPOS()
        x_pos = round(pos['A'], 3)
        y_pos = round(pos['B'], 3)
        z_pos = round(pos['C'], 3)
        
        last_position = [x_pos, y_pos, z_pos]
        
        filepath = "C:/Users/CibionPC/Desktop/PyPrinting"
        name = str(filepath  + "/" + "Last_position.txt")
        
        np.savetxt(name, last_position)
        
        print("\n Save Last position on x, y, z:", last_position)

        pi_device.MOV(['A','B','C'], [0, 0, 0])
        pi_device.CloseConnection()

        print(datetime.now(), 'Platina CloseConnection')
        #PDtask.close()
        #shuttertask.close()
        
        Flipper_notch532('down')
          

#    def make_connection(self, frontend):
#        
#        frontend.selectDirSignal.connect(self.selectDir)
#        frontend.openDirSignal.connect(self.openDir)
#        frontend.createDirSignal.connect(self.create_daily_directory)
#        frontend.loadpositionSignal.connect(self.load_last_position)
#        frontend.closeSignal.connect(self.close_all)
#        
#        frontend.focusWidget.make_connection(self.focusWorker)
#        frontend.nanopositioningWidget.make_connection(self.nanopositioningWorker)
#        frontend.traceWidget.make_connection(self.traceWorker)
#        frontend.confocalWidget.make_connection(self.confocalWorker)
#        frontend.printingWidget.make_connection(self.printingWorker)    
#        frontend.dimersWidget.make_connection(self.dimersWorker)
        
    
if __name__ == '__main__':

    app = QtGui.QApplication([])
    
    gui = Frontend()
    worker = Backend()
    
#    gui.make_connection(worker)
#    worker.make_connection(gui)
    
    multipleThread = QtCore.QThread()
    
    #worker.shuttersWorker.moveToThread(multipleThread)
    
    #worker.nanopositioningWorker.moveToThread(multipleThread)
    
    #worker.focusWorker.moveToThread(multipleThread)
    
    #worker.traceWorker.moveToThread(multipleThread)
    #worker.traceWorker.pointtimer.moveToThread(multipleThread)
    
    #worker.confocalWorker.moveToThread(multipleThread)
   # worker.confocalWorker.PDtimer_stepxy.moveToThread(multipleThread)
    #worker.confocalWorker.PDtimer_rampxy.moveToThread(multipleThread)
   # worker.confocalWorker.PDtimer_rampxz.moveToThread(multipleThread)
   # worker.confocalWorker.drifttimer.moveToThread(multipleThread)
    
   # worker.printingWorker.moveToThread(multipleThread)
   # worker.dimersWorker.moveToThread(multipleThread)
   
    gui.moveToThread(multipleThread)
    multipleThread.start()
 
    gui.show()
    app.exec_()
        
        