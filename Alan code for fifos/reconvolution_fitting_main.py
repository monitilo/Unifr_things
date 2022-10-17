# -*- coding: utf-8 -*-
"""
@author: alanszalai
"""
import tkinter as tk
import pandas as pd
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QProgressBar, QPushButton, QMessageBox)

from tkinter.filedialog import askdirectory, askopenfilename
from wavelet_analysis import parse_file, process_entry
from scipy.optimize import curve_fit
from run_analysis_reconvolution import analysis_data, run_analysis
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from aux_functions import shift_IRF, obtain_histogram_values,read_ptu,generate_data, generate_IRF,simulate_data_func
from matplotlib import patches


#%% Screen resolution 

# Our convertion from millimeters to inches
MM_TO_IN = 0.0393700787

# Import the libraries
import ctypes
import math
import tkinter

# Set process DPI awareness
ctypes.windll.shcore.SetProcessDpiAwareness(1)
# Create a tkinter window
root = tkinter.Tk()
# Get a DC from the window's HWND
dc = ctypes.windll.user32.GetDC(root.winfo_id())
# The the monitor phyical width
# (returned in millimeters then converted to inches)
mw = ctypes.windll.gdi32.GetDeviceCaps(dc, 4) * MM_TO_IN
# The the monitor phyical height
mh = ctypes.windll.gdi32.GetDeviceCaps(dc, 6) * MM_TO_IN
# Get the monitor horizontal resolution
dw = ctypes.windll.gdi32.GetDeviceCaps(dc, 8)
# Get the monitor vertical resolution
dh = ctypes.windll.gdi32.GetDeviceCaps(dc, 10)
# Destroy the window
root.destroy()

# Horizontal and vertical DPIs calculated
hdpi, vdpi = dw / mw, dh / mh
# Diagonal DPI calculated using Pythagoras
ddpi = math.hypot(dw, dh) / math.hypot(mw, mh)
# Print the DPIs

# f1 = screensize[0]*2/3840 # adjust resolution of GUI
# f2 = screensize[1]*2/2160 # adjust resolution of GUI

f1 = (1/1436)*0.75*mw*hdpi

f2 = (1/1047)*0.9*mh*vdpi

Font_size = 7

#%%

    
class Ui_MainWindow(object):

    def setupUi(self, MainWindow): # here all the widgets are defined and located in different parts of the GUI
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(round(f1*1436), round(f2*1047))
        
        
       
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setFont(QtGui.QFont("Times", Font_size))
        
        self.browse_dir = QtWidgets.QPushButton(self.centralwidget)
        self.browse_dir.setGeometry(QtCore.QRect(round(f1*80), round(f2*40), round(f1*151), round(f2*30)))
        self.browse_dir.setObjectName("browse_dir")  
        self.browse_dir.setFont(QtGui.QFont("Times", Font_size))
        
        self.file_format = QtWidgets.QComboBox(self.centralwidget)
        self.file_format.setGeometry(QtCore.QRect(round(f1*15), round(f2*40), round(f1*60), round(f2*30)))
        self.file_format.setObjectName("file_format")  
        self.file_format.setFont(QtGui.QFont("Times", Font_size))
        self.file_format.addItem('.fifo')
        self.file_format.addItem('.ptu')
        self.file_format.addItem('.csv')
        
        self.file_format_label = QtWidgets.QLabel(self.centralwidget)
        self.file_format_label.setGeometry(QtCore.QRect(round(f1*12), round(f2*10), round(f1*80), round(f2*30)))
        self.file_format_label.setObjectName("file_format_label")
        self.file_format_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(round(f1*260), round(f2*30), round(f1*261), round(f2*161)))
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setFont(QtGui.QFont("Times", Font_size))


        self.browse_IRF = QtWidgets.QPushButton(self.groupBox)
        self.browse_IRF.setGeometry(QtCore.QRect(round(f1*100), round(f2*40), round(f1*151), round(f2*31)))
        self.browse_IRF.setObjectName("browse_IRF")
        self.browse_IRF.setFont(QtGui.QFont("Times", Font_size))
        
        self.exp_IRF_yn = QtWidgets.QRadioButton(self.groupBox)
        self.exp_IRF_yn.setGeometry(QtCore.QRect(round(f1*10), round(f2*20), round(f1*1.2*111), round(f2*16)))
        self.exp_IRF_yn.setObjectName("exp_IRF_yn")
        self.exp_IRF_yn.setFont(QtGui.QFont("Times", Font_size))
        
        self.IRF_FWHM_label = QtWidgets.QLabel(self.groupBox)
        self.IRF_FWHM_label.setGeometry(QtCore.QRect(round(f1*160), round(f2*110), round(f1*1.2*81), round(f2*16)))
        self.IRF_FWHM_label.setObjectName("IRF_FWHM_label")
        self.IRF_FWHM_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.IRF_fwhm_sim = QtWidgets.QLineEdit(self.groupBox)
        self.IRF_fwhm_sim.setGeometry(QtCore.QRect(round(f1*152), round(f2*130), round(f1*101), round(f2*20)))
        self.IRF_fwhm_sim.setObjectName("IRF_fwhm_sim")
        self.IRF_fwhm_sim.setFont(QtGui.QFont("Times", Font_size))

        self.simulated_IRF_yn = QtWidgets.QRadioButton(self.groupBox)
        self.simulated_IRF_yn.setGeometry(QtCore.QRect(round(f1*10), round(f2*130), round(f1*1.2*101), round(f2*16)))
        self.simulated_IRF_yn.setObjectName("simulated_IRF_yn")
        self.simulated_IRF_yn.setFont(QtGui.QFont("Times", Font_size))

        self.IRF_filename_label = QtWidgets.QLabel(self.groupBox)
        self.IRF_filename_label.setGeometry(QtCore.QRect(round(f1*11), round(f2*60), round(f1*1.25*62), round(f2*16)))
        self.IRF_filename_label.setObjectName("label")
        self.IRF_filename_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_20 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_20.setGeometry(QtCore.QRect(round(f1*570), round(f2*30), round(f1*420), round(f2*161)))
        self.groupBox_20.setObjectName("groupBox_2")
        self.groupBox_20.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_21 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_21.setGeometry(QtCore.QRect(round(f1*570), round(f2*30), round(f1*420), round(f2*161)))
        self.groupBox_21.setObjectName("groupBox_2")
        self.groupBox_21.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(round(f1*570), round(f2*30), round(f1*420), round(f2*161)))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_2.addTab(self.groupBox_20,'Monoexponential simulation')
        self.groupBox_2.addTab(self.groupBox_21,'Biexponential simulation')     

        self.lifetime_ns_sim_label = QtWidgets.QLabel(self.groupBox_20)
        self.lifetime_ns_sim_label.setGeometry(QtCore.QRect(round(f1*10), round(f2*26), round(f1*131), round(f2*16)))
        self.lifetime_ns_sim_label.setObjectName("lifetime_ns_sim_label")
        self.lifetime_ns_sim_label.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime_sim = QtWidgets.QLineEdit(self.groupBox_20)
        self.lifetime_sim.setGeometry(QtCore.QRect(round(f1*10), round(f2*50), round(f1*121), round(f2*20)))
        self.lifetime_sim.setObjectName("lifetime_sim")
        self.lifetime_sim.setFont(QtGui.QFont("Times", Font_size))

        self.num_photons_mono_simulation = QtWidgets.QLineEdit(self.groupBox_20)
        self.num_photons_mono_simulation.setGeometry(QtCore.QRect(round(f1*160), round(f2*50), round(f1*111), round(f2*20)))
        self.num_photons_mono_simulation.setObjectName("num_photons_simulation")
        self.num_photons_mono_simulation.setFont(QtGui.QFont("Times", Font_size))

        self.photons_sim_mono_label = QtWidgets.QLabel(self.groupBox_20)
        self.photons_sim_mono_label.setGeometry(QtCore.QRect(round(f1*160), round(f2*26), round(f1*111), round(f2*16)))
        self.photons_sim_mono_label.setObjectName("photons_sim_label")
        self.photons_sim_mono_label.setFont(QtGui.QFont("Times", Font_size))

        self.simulate_mono_decays = QtWidgets.QPushButton(self.groupBox_20)
        self.simulate_mono_decays.setGeometry(QtCore.QRect(round(f1*300), round(f2*80), round(f1*111), round(f2*31)))
        self.simulate_mono_decays.setObjectName("simulate_decays")
        self.simulate_mono_decays.setFont(QtGui.QFont("Times", Font_size))

        self.repeats_mono_simulation = QtWidgets.QLineEdit(self.groupBox_20)
        self.repeats_mono_simulation.setGeometry(QtCore.QRect(round(f1*292), round(f2*50), round(f1*121), round(f2*20)))
        self.repeats_mono_simulation.setObjectName("repeats_simulation")
        self.repeats_mono_simulation.setFont(QtGui.QFont("Times", Font_size))

        self.num_simulations_mono_label = QtWidgets.QLabel(self.groupBox_20)
        self.num_simulations_mono_label.setGeometry(QtCore.QRect(round(f1*310), round(f2*26), round(f1*1.2*71), round(f2*16)))
        self.num_simulations_mono_label.setObjectName("num_simulations_label")
        self.num_simulations_mono_label.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime_1_sim_label = QtWidgets.QLabel(self.groupBox_21)
        self.lifetime_1_sim_label.setGeometry(QtCore.QRect(round(f1*10), round(f2*26), round(f1*131), round(f2*16)))
        self.lifetime_1_sim_label.setObjectName("lifetime_1_sim_label")
        self.lifetime_1_sim_label.setFont(QtGui.QFont("Times", Font_size))


        self.lifetime_1_sim = QtWidgets.QLineEdit(self.groupBox_21)
        self.lifetime_1_sim.setGeometry(QtCore.QRect(round(f1*10), round(f2*50), round(f1*121), round(f2*20)))
        self.lifetime_1_sim.setObjectName("lifetime_1_sim")
        self.lifetime_1_sim.setFont(QtGui.QFont("Times", Font_size))        
       
        self.lifetime_2_sim_label = QtWidgets.QLabel(self.groupBox_21)
        self.lifetime_2_sim_label.setGeometry(QtCore.QRect(round(f1*10), round(f2*76), round(f1*131), round(f2*16)))
        self.lifetime_2_sim_label.setObjectName("lifetime_2_sim_label")
        self.lifetime_2_sim_label.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime_2_sim = QtWidgets.QLineEdit(self.groupBox_21)
        self.lifetime_2_sim.setGeometry(QtCore.QRect(round(f1*10), round(f2*100), round(f1*121), round(f2*20)))
        self.lifetime_2_sim.setObjectName("lifetime_2_sim")
        self.lifetime_2_sim.setFont(QtGui.QFont("Times", Font_size))
        
        self.a_1_sim_label = QtWidgets.QLabel(self.groupBox_21)
        self.a_1_sim_label.setGeometry(QtCore.QRect(round(f1*180), round(f2*76), round(f1*45), round(f2*16)))
        self.a_1_sim_label.setObjectName("a_1_sim_label")
        self.a_1_sim_label.setFont(QtGui.QFont("Times", Font_size))

        self.a_1_sim = QtWidgets.QLineEdit(self.groupBox_21)
        self.a_1_sim.setGeometry(QtCore.QRect(round(f1*170), round(f2*100), round(f1*45), round(f2*20)))
        self.a_1_sim.setObjectName("a_1_sim")
        self.a_1_sim.setFont(QtGui.QFont("Times", Font_size))   
        
        self.a_2_sim_label = QtWidgets.QLabel(self.groupBox_21)
        self.a_2_sim_label.setGeometry(QtCore.QRect(round(f1*230), round(f2*76), round(f1*45), round(f2*16)))
        self.a_2_sim_label.setObjectName("a_2_sim_label")
        self.a_2_sim_label.setFont(QtGui.QFont("Times", Font_size))

        self.a_2_sim = QtWidgets.QLineEdit(self.groupBox_21)
        self.a_2_sim.setGeometry(QtCore.QRect(round(f1*220), round(f2*100), round(f1*45), round(f2*20)))
        self.a_2_sim.setObjectName("a_2_sim")
        self.a_2_sim.setFont(QtGui.QFont("Times", Font_size)) 

        self.num_photons_bi_simulation = QtWidgets.QLineEdit(self.groupBox_21)
        self.num_photons_bi_simulation.setGeometry(QtCore.QRect(round(f1*160), round(f2*50), round(f1*111), round(f2*20)))
        self.num_photons_bi_simulation.setObjectName("num_photons_simulation")
        self.num_photons_bi_simulation.setFont(QtGui.QFont("Times", Font_size))

        self.photons_sim_bi_label = QtWidgets.QLabel(self.groupBox_21)
        self.photons_sim_bi_label.setGeometry(QtCore.QRect(round(f1*160), round(f2*26), round(f1*111), round(f2*16)))
        self.photons_sim_bi_label.setObjectName("photons_sim_label")
        self.photons_sim_bi_label.setFont(QtGui.QFont("Times", Font_size))

        self.simulate_bi_decays = QtWidgets.QPushButton(self.groupBox_21)
        self.simulate_bi_decays.setGeometry(QtCore.QRect(round(f1*300), round(f2*80), round(f1*111), round(f2*31)))
        self.simulate_bi_decays.setObjectName("simulate_decays")
        self.simulate_bi_decays.setFont(QtGui.QFont("Times", Font_size))

        self.repeats_bi_simulation = QtWidgets.QLineEdit(self.groupBox_21)
        self.repeats_bi_simulation.setGeometry(QtCore.QRect(round(f1*292), round(f2*50), round(f1*121), round(f2*20)))
        self.repeats_bi_simulation.setObjectName("repeats_simulation")
        self.repeats_bi_simulation.setFont(QtGui.QFont("Times", Font_size))

        self.num_simulations_bi_label = QtWidgets.QLabel(self.groupBox_21)
        self.num_simulations_bi_label.setGeometry(QtCore.QRect(round(f1*310), round(f2*26), round(f1*1.2*71), round(f2*16)))
        self.num_simulations_bi_label.setObjectName("num_simulations_label")
        self.num_simulations_bi_label.setFont(QtGui.QFont("Times", Font_size))

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(round(f1*20), round(f2*240), round(f1*981), round(f2*691)))
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_3.setFont(QtGui.QFont("Times", Font_size))

        self.list_of_filenames = QtWidgets.QListWidget(self.groupBox_3)
        self.list_of_filenames.setGeometry(QtCore.QRect(round(f1*10), round(f2*40), round(f1*161), round(f2*260)))
        self.list_of_filenames.setObjectName("list_of_filenames")
        self.list_of_filenames.setFont(QtGui.QFont("Times", Font_size))

        self.files_label = QtWidgets.QLabel(self.groupBox_3)
        self.files_label.setGeometry(QtCore.QRect(round(f1*10), round(f2*20), round(f1*81), round(f2*16)))
        self.files_label.setObjectName("files_label")
        self.files_label.setFont(QtGui.QFont("Times", Font_size))


        self.groupBox_40 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_40.setGeometry(QtCore.QRect(round(f1*180), round(f2*10), round(f1*171), round(f2*341)))
        self.groupBox_40.setObjectName("groupBox_40")
        self.groupBox_40.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_41 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_41.setGeometry(QtCore.QRect(round(f1*180), round(f2*10), round(f1*171), round(f2*341)))
        self.groupBox_41.setObjectName("groupBox_41")
        self.groupBox_41.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_4 = QtWidgets.QTabWidget(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(round(f1*180), round(f2*10), round(f1*171), round(f2*341)))
        self.groupBox_4.setObjectName("groupBox_4")
        self.groupBox_4.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_4.addTab(self.groupBox_40,'Mono. param.')
        self.groupBox_4.addTab(self.groupBox_41,'Bi. param.')  


        self.pulse_position_label = QtWidgets.QLabel(self.groupBox_40)
        self.pulse_position_label.setGeometry(QtCore.QRect(round(f1*35), round(f2*140), round(f1*1.5*91), round(f2*16)))
        self.pulse_position_label.setObjectName("pulse_position_label")
        self.pulse_position_label.setFont(QtGui.QFont("Times", Font_size))

        self.pulse_position_initial_guess = QtWidgets.QLineEdit(self.groupBox_40)
        self.pulse_position_initial_guess.setGeometry(QtCore.QRect(round(f1*20), round(f2*160), round(f1*115), round(f2*20)))
        self.pulse_position_initial_guess.setObjectName("pulse_position_initial_guess")
        self.pulse_position_initial_guess.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime_initial_guess = QtWidgets.QLineEdit(self.groupBox_40)
        self.lifetime_initial_guess.setGeometry(QtCore.QRect(round(f1*20), round(f2*118), round(f1*115), round(f2*20)))
        self.lifetime_initial_guess.setObjectName("lifetime_initial_guess")
        self.lifetime_initial_guess.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime_fit_label = QtWidgets.QLabel(self.groupBox_40)
        self.lifetime_fit_label.setGeometry(QtCore.QRect(round(f1*50), round(f2*98), round(f1*1.5*71), round(f2*16)))
        self.lifetime_fit_label.setObjectName("lifetime_fit_label")
        self.lifetime_fit_label.setFont(QtGui.QFont("Times", Font_size))

        self.modify_initial_guess_label = QtWidgets.QLabel(self.groupBox_40)
        self.modify_initial_guess_label.setGeometry(QtCore.QRect(round(f1*30), round(f2*10), round(f1*1.5*101), round(f2*21)))
        self.modify_initial_guess_label.setObjectName("modify_initial_guess_label")
        self.modify_initial_guess_label.setFont(QtGui.QFont("Times", Font_size))

        self.update_fit = QtWidgets.QPushButton(self.groupBox_40)
        self.update_fit.setGeometry(QtCore.QRect(round(f1*40), round(f2*260), round(f1*75), round(f2*23)))
        self.update_fit.setObjectName("update_fit")
        self.update_fit.setFont(QtGui.QFont("Times", Font_size))

        self.reload_auto_fit = QtWidgets.QPushButton(self.groupBox_40)
        self.reload_auto_fit.setGeometry(QtCore.QRect(round(f1*7), round(f2*285), round(f1*1.2*121), round(f2*21)))
        self.reload_auto_fit.setObjectName("reload_auto_fit")
        self.reload_auto_fit.setFont(QtGui.QFont("Times", Font_size))        
        
        self.a1_initial_guess = QtWidgets.QLineEdit(self.groupBox_41)
        self.a1_initial_guess.setGeometry(QtCore.QRect(round(f1*20), round(f2*78), round(f1*55), round(f2*20)))
        self.a1_initial_guess.setObjectName("lifetime1_initial_guess")
        self.a1_initial_guess.setFont(QtGui.QFont("Times", Font_size))

        self.modify_initial_guess_biexp_label = QtWidgets.QLabel(self.groupBox_41)
        self.modify_initial_guess_biexp_label.setGeometry(QtCore.QRect(round(f1*30), round(f2*10), round(f1*1.5*101), round(f2*21)))
        self.modify_initial_guess_biexp_label.setObjectName("modify_initial_guess_biexp_label")
        self.modify_initial_guess_biexp_label.setFont(QtGui.QFont("Times", Font_size))

        self.a1_fit_label = QtWidgets.QLabel(self.groupBox_41)
        self.a1_fit_label.setGeometry(QtCore.QRect(round(f1*20), round(f2*58), round(f1*55), round(f2*20)))
        self.a1_fit_label.setObjectName("lifetime1_fit_label")
        self.a1_fit_label.setFont(QtGui.QFont("Times", Font_size))
                
        self.a2_initial_guess = QtWidgets.QLineEdit(self.groupBox_41)
        self.a2_initial_guess.setGeometry(QtCore.QRect(round(f1*80), round(f2*78), round(f1*55), round(f2*20)))
        self.a2_initial_guess.setObjectName("lifetime2_initial_guess")
        self.a2_initial_guess.setFont(QtGui.QFont("Times", Font_size))
        
        self.a2_fit_label = QtWidgets.QLabel(self.groupBox_41)
        self.a2_fit_label.setGeometry(QtCore.QRect(round(f1*80), round(f2*58), round(f1*55), round(f2*20)))
        self.a2_fit_label.setObjectName("lifetime2_fit_label")
        self.a2_fit_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.pulse_position_biexp_label = QtWidgets.QLabel(self.groupBox_41)
        self.pulse_position_biexp_label.setGeometry(QtCore.QRect(round(f1*35), round(f2*140), round(f1*1.5*91), round(f2*16)))
        self.pulse_position_biexp_label.setObjectName("pulse_position_biexp_label")
        self.pulse_position_biexp_label.setFont(QtGui.QFont("Times", Font_size))

        self.pulse_position_biexp_initial_guess = QtWidgets.QLineEdit(self.groupBox_41)
        self.pulse_position_biexp_initial_guess.setGeometry(QtCore.QRect(round(f1*20), round(f2*160), round(f1*115), round(f2*20)))
        self.pulse_position_biexp_initial_guess.setObjectName("pulse_position_biexp_initial_guess")
        self.pulse_position_biexp_initial_guess.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime1_initial_guess = QtWidgets.QLineEdit(self.groupBox_41)
        self.lifetime1_initial_guess.setGeometry(QtCore.QRect(round(f1*20), round(f2*118), round(f1*55), round(f2*20)))
        self.lifetime1_initial_guess.setObjectName("lifetime1_initial_guess")
        self.lifetime1_initial_guess.setFont(QtGui.QFont("Times", Font_size))

        self.lifetime1_fit_label = QtWidgets.QLabel(self.groupBox_41)
        self.lifetime1_fit_label.setGeometry(QtCore.QRect(round(f1*20), round(f2*98), round(f1*55), round(f2*20)))
        self.lifetime1_fit_label.setObjectName("lifetime1_fit_label")
        self.lifetime1_fit_label.setFont(QtGui.QFont("Times", Font_size))
                
        self.lifetime2_initial_guess = QtWidgets.QLineEdit(self.groupBox_41)
        self.lifetime2_initial_guess.setGeometry(QtCore.QRect(round(f1*80), round(f2*118), round(f1*55), round(f2*20)))
        self.lifetime2_initial_guess.setObjectName("lifetime2_initial_guess")
        self.lifetime2_initial_guess.setFont(QtGui.QFont("Times", Font_size))
        
        self.lifetime2_fit_label = QtWidgets.QLabel(self.groupBox_41)
        self.lifetime2_fit_label.setGeometry(QtCore.QRect(round(f1*80), round(f2*98), round(f1*55), round(f2*20)))
        self.lifetime2_fit_label.setObjectName("lifetime2_fit_label")
        self.lifetime2_fit_label.setFont(QtGui.QFont("Times", Font_size))


        self.update_biexp_fit = QtWidgets.QPushButton(self.groupBox_41)
        self.update_biexp_fit.setGeometry(QtCore.QRect(round(f1*40), round(f2*260), round(f1*75), round(f2*23)))
        self.update_biexp_fit.setObjectName("update_biexp_fit")
        self.update_biexp_fit.setFont(QtGui.QFont("Times", Font_size))

        self.reload_auto_biexp_fit = QtWidgets.QPushButton(self.groupBox_41)
        self.reload_auto_biexp_fit.setGeometry(QtCore.QRect(round(f1*7), round(f2*285), round(f1*1.2*121), round(f2*21)))
        self.reload_auto_biexp_fit.setObjectName("reload_auto_biexp_fit")
        self.reload_auto_biexp_fit.setFont(QtGui.QFont("Times", Font_size))
        

        self.widget = MplWidget(self.groupBox_3)
        self.widget.setGeometry(QtCore.QRect(round(f1*380), round(f2*0), round(f1*591), round(f2*461)))
        self.widget.setObjectName("widget")
        
        self.widget_2 = MplWidget(self.groupBox_3)
        self.widget_2.setGeometry(QtCore.QRect(round(f1*0), round(f2*420), round(f1*405), round(f2*280)))
        self.widget_2.setObjectName("widget_2")
        
        self.widget_3 = MplWidget(self.groupBox_3)
        self.widget_3.setGeometry(QtCore.QRect(round(f1*380), round(f2*420), round(f1*591), round(f2*280)))
        self.widget_3.setObjectName("widget_3")
        
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(round(f1*1036), round(f2*245), round(f1*350), round(f2*690)))
        self.groupBox_5.setObjectName("groupBox_4")
        self.groupBox_5.setFont(QtGui.QFont("Times", Font_size))
        
        self.widget_4 = MplWidget(self.groupBox_5)
        self.widget_4.setGeometry(QtCore.QRect(round(f1*10), round(f2*0), round(f1*320), round(f2*320)))
        self.widget_4.setObjectName("widget_3")
        
        self.widget_5 = MplWidget(self.groupBox_5)
        self.widget_5.setGeometry(QtCore.QRect(round(f1*10), round(f2*360), round(f1*320), round(f2*320)))
        self.widget_5.setObjectName("widget_4")
        
                
        self.analyze_folder = QtWidgets.QPushButton(self.centralwidget)
        self.analyze_folder.setGeometry(QtCore.QRect(round(f1*1210), round(f2*960), round(f1*141), round(f2*41)))
        self.analyze_folder.setObjectName("analyze_folder")
        self.analyze_folder.setFont(QtGui.QFont("Times", Font_size))

        self.IRF_filename = QtWidgets.QLabel(self.centralwidget)
        self.IRF_filename.setGeometry(QtCore.QRect(round(f1*270), round(f2*110), round(f1*241), round(f2*20)))
        self.IRF_filename.setObjectName("IRF_filename")
        self.IRF_filename.setFont(QtGui.QFont("Times", Font_size))

        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(round(f1*0), round(f2*0), round(f1*2), round(f2*2)))
        self.layoutWidget.setObjectName("layoutWidget")
        
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(round(f1*0), round(f2*0), round(f1*2), round(f2*2)))
        self.layoutWidget1.setObjectName("layoutWidget1")        

        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(round(f1*70), round(f2*90), round(f1*1.2*95), round(f2*50)))
        self.layoutWidget2.setObjectName("layoutWidget2")
        
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.red_channel = QtWidgets.QRadioButton(self.layoutWidget2)
        self.red_channel.setObjectName("red_channel")
        self.red_channel.setFont(QtGui.QFont("Times", Font_size))
        self.verticalLayout_2.addWidget(self.red_channel)
        
        self.green_channel = QtWidgets.QRadioButton(self.layoutWidget2)
        self.green_channel.setObjectName("green_channel")
        self.green_channel.setFont(QtGui.QFont("Times", Font_size))
        self.verticalLayout_2.addWidget(self.green_channel)
        
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(round(f1*70), round(f2*160), round(f1*1.2*95), round(f2*50)))
        self.layoutWidget3.setObjectName("layoutWidget2")
        
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_2")        
        
        self.monoexp_analysis = QtWidgets.QRadioButton(self.layoutWidget3)
        self.monoexp_analysis.setObjectName("monoexp_analysis")
        self.monoexp_analysis.setFont(QtGui.QFont("Times", Font_size))
        self.verticalLayout_3.addWidget(self.monoexp_analysis)
        
        self.biexp_analysis = QtWidgets.QRadioButton(self.layoutWidget3)
        self.biexp_analysis.setObjectName("green_channel")
        self.biexp_analysis.setFont(QtGui.QFont("Times", Font_size))
        self.verticalLayout_3.addWidget(self.biexp_analysis)        
       
        self.current_file_label = QtWidgets.QLabel(self.groupBox_3)
        self.current_file_label.setGeometry(QtCore.QRect(round(f1*11), round(f2*305), round(f1*77), round(f2*16)))
        self.current_file_label.setObjectName("current_file_label")
        self.current_file_label.setFont(QtGui.QFont("Times", Font_size))

        self.box_file = QtWidgets.QSpinBox(self.groupBox_3)
        self.box_file.setGeometry(QtCore.QRect(round(f1*11), round(f2*323), round(f1*161), round(f2*25)))
        self.box_file.setObjectName("box_file")       
        self.box_file.setFont(QtGui.QFont("Times", Font_size))

        self.dir_path_label = QtWidgets.QLabel(self.groupBox_3)
        self.dir_path_label.setGeometry(QtCore.QRect(round(f1*11), round(f2*360), round(f1*77), round(f2*16)))
        self.dir_path_label.setObjectName("dir_path_label")
        self.dir_path_label.setFont(QtGui.QFont("Times", Font_size))

        self.dir_path = QtWidgets.QLabel(self.groupBox_3)
        self.dir_path.setGeometry(QtCore.QRect(round(f1*11), round(f2*375), round(f1*370), round(f2*48)))
        self.dir_path.setObjectName("dir_path")
        self.dir_path.setWordWrap(True) 
        self.dir_path.setFont(QtGui.QFont("Times", Font_size))
  
        self.export_results = QtWidgets.QPushButton(self.centralwidget)
        self.export_results.setGeometry(QtCore.QRect(round(f1*1060), round(f2*960), round(f1*141), round(f2*41)))
        self.export_results.setObjectName("export_results")  
        self.export_results.setFont(QtGui.QFont("Times", Font_size))  
        
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(round(f1*20), round(f2*940), round(f1*390), round(f2*85)))
        self.groupBox_6.setObjectName("groupBox_6")
        self.groupBox_6.setFont(QtGui.QFont("Times", Font_size))        
       
        self.update_macrotime = QtWidgets.QPushButton(self.groupBox_6)
        self.update_macrotime.setGeometry(QtCore.QRect(round(f1*155), round(f2*53), round(f1*90), round(f2*25)))
        self.update_macrotime.setObjectName("update_macrotime")  
        self.update_macrotime.setFont(QtGui.QFont("Times", Font_size))
        
        self.bin_macrotime = QtWidgets.QLineEdit(self.groupBox_6)
        self.bin_macrotime.setGeometry(QtCore.QRect(round(f1*300), round(f2*25), round(f1*41), round(f2*20)))
        self.bin_macrotime.setObjectName("bin_macrotime")  
        self.bin_macrotime.setFont(QtGui.QFont("Times", Font_size))

        self.bin_macrotime_label = QtWidgets.QLabel(self.groupBox_6)
        self.bin_macrotime_label.setGeometry(QtCore.QRect(round(f1*285), round(f2*5), round(f1*141), round(f2*16)))
        self.bin_macrotime_label.setObjectName("bin_macrotime_label")    
        self.bin_macrotime_label.setFont(QtGui.QFont("Times", Font_size))
                
        self.intensity_threshold = QtWidgets.QLineEdit(self.groupBox_6)
        self.intensity_threshold.setGeometry(QtCore.QRect(round(f1*170), round(f2*25), round(f1*60), round(f2*20)))
        self.intensity_threshold.setObjectName("intensity_threshold")
        self.intensity_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.intensity_threshold_label = QtWidgets.QLabel(self.groupBox_6)
        self.intensity_threshold_label.setGeometry(QtCore.QRect(round(f1*160), round(f2*5), round(f1*1.5*121), round(f2*16)))
        self.intensity_threshold_label.setObjectName("intensity_threshold_label")  
        self.intensity_threshold_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.time1_threshold = QtWidgets.QLineEdit(self.groupBox_6)
        self.time1_threshold.setGeometry(QtCore.QRect(round(f1*20), round(f2*25), round(f1*40), round(f2*20)))
        self.time1_threshold.setObjectName("time1_threshold")
        self.time1_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.time2_threshold = QtWidgets.QLineEdit(self.groupBox_6)
        self.time2_threshold.setGeometry(QtCore.QRect(round(f1*75), round(f2*25), round(f1*40), round(f2*20)))
        self.time2_threshold.setObjectName("time2_threshold")  
        self.time2_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.macrotime_label = QtWidgets.QLabel(self.groupBox_6)
        self.macrotime_label.setGeometry(QtCore.QRect(round(f1*17), round(f2*5), round(f1*1.5*125), round(f2*16)))
        self.macrotime_label.setObjectName("macrotime_label")    
        self.macrotime_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.fixed_intensity_threshold = QtWidgets.QCheckBox(self.groupBox_6)
        self.fixed_intensity_threshold.setGeometry(QtCore.QRect(round(f1*17), round(f2*57), round(f1*1.5*125), round(f2*16)))
        self.fixed_intensity_threshold.setObjectName("fixed_intensity_threshold")    
        self.fixed_intensity_threshold.setFont(QtGui.QFont("Times", Font_size))
       
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(round(f1*720), round(f2*940), round(f1*290), round(f2*85)))
        self.groupBox_7.setObjectName("groupBox_6")
        self.groupBox_7.setFont(QtGui.QFont("Times", Font_size))      
        
        self.update_nanotime = QtWidgets.QPushButton(self.groupBox_7)
        self.update_nanotime.setGeometry(QtCore.QRect(round(f1*170), round(f2*53), round(f1*90), round(f2*25)))
        self.update_nanotime.setObjectName("update_nanotime")  
        self.update_nanotime.setFont(QtGui.QFont("Times", Font_size))
       
        self.bin_nanotime = QtWidgets.QLineEdit(self.groupBox_7)
        self.bin_nanotime.setGeometry(QtCore.QRect(round(f1*190), round(f2*25), round(f1*50), round(f2*20)))
        self.bin_nanotime.setObjectName("bin_nanotime")  
        self.bin_nanotime.setFont(QtGui.QFont("Times", Font_size))

        self.bin_nanotime_label = QtWidgets.QLabel(self.groupBox_7)
        self.bin_nanotime_label.setGeometry(QtCore.QRect(round(f1*175), round(f2*5), round(f1*141), round(f2*16)))
        self.bin_nanotime_label.setObjectName("bin_nanotime_label")    
        self.bin_nanotime_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.nanotime1_threshold = QtWidgets.QLineEdit(self.groupBox_7)
        self.nanotime1_threshold.setGeometry(QtCore.QRect(round(f1*25), round(f2*25), round(f1*55), round(f2*20)))
        self.nanotime1_threshold.setObjectName("time1_threshold")
        self.nanotime1_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.nanotime2_threshold = QtWidgets.QLineEdit(self.groupBox_7)
        self.nanotime2_threshold.setGeometry(QtCore.QRect(round(f1*85), round(f2*25), round(f1*55), round(f2*20)))
        self.nanotime2_threshold.setObjectName("time2_threshold")  
        self.nanotime2_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.nanotime_label = QtWidgets.QLabel(self.groupBox_7)
        self.nanotime_label.setGeometry(QtCore.QRect(round(f1*30), round(f2*5), round(f1*220), round(f2*16)))
        self.nanotime_label.setObjectName("nanotime_label")    
        self.nanotime_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.fixed_nanotime_threshold = QtWidgets.QCheckBox(self.groupBox_7)
        self.fixed_nanotime_threshold.setGeometry(QtCore.QRect(round(f1*25), round(f2*57), round(f1*141), round(f2*16)))
        self.fixed_nanotime_threshold.setObjectName("fixed_nanotime_threshold")    
        self.fixed_nanotime_threshold.setFont(QtGui.QFont("Times", Font_size))
        
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_8.setGeometry(QtCore.QRect(round(f1*415), round(f2*940), round(f1*295), round(f2*85)))
        self.groupBox_8.setObjectName("groupBox_6")
        self.groupBox_8.setFont(QtGui.QFont("Times", Font_size))         
        
        self.nanotime1_plot_threshold = QtWidgets.QLineEdit(self.groupBox_8)
        self.nanotime1_plot_threshold.setGeometry(QtCore.QRect(round(f1*60), round(f2*25), round(f1*55), round(f2*20)))
        self.nanotime1_plot_threshold.setObjectName("nanotime1_plot_threshold")
        self.nanotime1_plot_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.nanotime2_plot_threshold = QtWidgets.QLineEdit(self.groupBox_8)
        self.nanotime2_plot_threshold.setGeometry(QtCore.QRect(round(f1*120), round(f2*25), round(f1*55), round(f2*20)))
        self.nanotime2_plot_threshold.setObjectName("nanotime2_plot_threshold")  
        self.nanotime2_plot_threshold.setFont(QtGui.QFont("Times", Font_size))
        
        self.nanotime1_counts_plot_threshold = QtWidgets.QLineEdit(self.groupBox_8)
        self.nanotime1_counts_plot_threshold.setGeometry(QtCore.QRect(round(f1*60), round(f2*57), round(f1*55), round(f2*20)))
        self.nanotime1_counts_plot_threshold.setObjectName("nanotime1_counts_plot_threshold")
        self.nanotime1_counts_plot_threshold.setFont(QtGui.QFont("Times", Font_size))

        self.nanotime2_counts_plot_threshold = QtWidgets.QLineEdit(self.groupBox_8)
        self.nanotime2_counts_plot_threshold.setGeometry(QtCore.QRect(round(f1*120), round(f2*57), round(f1*55), round(f2*20)))
        self.nanotime2_counts_plot_threshold.setObjectName("nanotime2_counts_plot_threshold")  
        self.nanotime2_counts_plot_threshold.setFont(QtGui.QFont("Times", Font_size))               

        self.nanotime_plot_label = QtWidgets.QLabel(self.groupBox_8)
        self.nanotime_plot_label.setGeometry(QtCore.QRect(round(f1*80), round(f2*5), round(f1*180), round(f2*16)))
        self.nanotime_plot_label.setObjectName("nanotime_plot_label")    
        self.nanotime_plot_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.nanotime_xplot_label = QtWidgets.QLabel(self.groupBox_8)
        self.nanotime_xplot_label.setGeometry(QtCore.QRect(round(f1*20), round(f2*25), round(f1*30), round(f2*16)))
        self.nanotime_xplot_label.setObjectName("nanotime_xplot_label")    
        self.nanotime_xplot_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.nanotime_yplot_label = QtWidgets.QLabel(self.groupBox_8)
        self.nanotime_yplot_label.setGeometry(QtCore.QRect(round(f1*3), round(f2*57), round(f1*50), round(f2*16)))
        self.nanotime_yplot_label.setObjectName("nanotime_xplot_label")    
        self.nanotime_yplot_label.setFont(QtGui.QFont("Times", Font_size))
        
        self.update_plot_nanotime = QtWidgets.QPushButton(self.groupBox_8)
        self.update_plot_nanotime.setGeometry(QtCore.QRect(round(f1*240), round(f2*26), round(f1*55), round(f2*50)))
        self.update_plot_nanotime.setObjectName("update_plot_nanotime")  
        self.update_plot_nanotime.setFont(QtGui.QFont("Times", Font_size))

        
        self.fixed_plot_nanotime = QtWidgets.QCheckBox(self.groupBox_8)
        self.fixed_plot_nanotime.setGeometry(QtCore.QRect(round(f1*185), round(f2*28), round(f1*141), round(f2*16)))
        self.fixed_plot_nanotime.setObjectName("fixed_plot_nanotime")    
        self.fixed_plot_nanotime.setFont(QtGui.QFont("Times", Font_size))
        
        self.fixed_counts_plot_nanotime = QtWidgets.QCheckBox(self.groupBox_8)
        self.fixed_counts_plot_nanotime.setGeometry(QtCore.QRect(round(f1*185), round(f2*60), round(f1*141), round(f2*16)))
        self.fixed_counts_plot_nanotime.setObjectName("fixed_counts_plot_nanotime")    
        self.fixed_counts_plot_nanotime.setFont(QtGui.QFont("Times", Font_size))
        
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(round(f1*0), round(f2*0), round(f1*1036), round(f2*22)))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)      
        
        self.exp_IRF_yn.setChecked(True)
        self.green_channel.setChecked(True) 
        self.monoexp_analysis.setChecked(True)
        self.intensity_threshold.setText('0')
        self.bin_nanotime.setText('0') 
        self.bin_macrotime.setText('5E-3')  
        self.IRF_browsed = False
        self.box_file.setMinimum(1)
        self.time2_changed = False
        self.analyze_entire_folder = False
        self.clicked = False
        self.IRF_dir_yn = False
        self.files_dir_yn = False
        self.update_fit_yn = False
        self.biexp_analysis_yn = False
        self.monoexp_sim = True
        self.IRF_ptu_yn = False
        self.IRF_csv_yn = False

        
        self.update_fit.clicked.connect(self.change_input_values_fit_manually) # Here some buttons are connected to specific functions (see below)
        self.update_biexp_fit.clicked.connect(self.change_input_values_fit_manually)
        self.reload_auto_fit.clicked.connect(self.reload_auto_fit_func)   
        self.reload_auto_biexp_fit.clicked.connect(self.reload_auto_fit_func)   
        self.browse_dir.clicked.connect(self.browse_directory)
        self.browse_IRF.clicked.connect(self.browse_file_IRF)
        self.list_of_filenames.currentItemChanged.connect(self.Clicked_file)
        self.box_file.valueChanged.connect(self.box_file_func)    
        self.time2_threshold.textChanged.connect(self.time2_changed_func)
        self.simulate_bi_decays.clicked.connect(self.simulate_data_function)
        self.simulate_mono_decays.clicked.connect(self.simulate_data_function)
        self.export_results.clicked.connect(self.export)
        self.analyze_folder.clicked.connect(self.analyze_folder_func)
        self.a_1_sim.textChanged.connect(self.a1_changed_func)
        self.a1_initial_guess.textChanged.connect(self.a1_initial_guess_changed_func)
        self.update_macrotime.clicked.connect(self.change_input_values_fit_manually)
        self.update_nanotime.clicked.connect(self.change_input_values_fit_manually)

        
        self.widget.canvas.ax.clear() # plots are cleared when the program starts
        self.widget_2.canvas.ax.clear()
        self.widget_3.canvas.ax.clear()
        self.widget_4.canvas.ax.clear()


        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Lifetime fitting tool"))
        self.browse_dir.setText(_translate("MainWindow", "Browse directory"))
        self.browse_IRF.setText(_translate("MainWindow", "Browse IRF file"))        
        self.file_format_label.setText(_translate("MainWindow", "Data file format"))
        
        self.dir_path_label.setText(_translate("MainWindow", "Dir path:"))
        self.current_file_label.setText(_translate("MainWindow", "Current file:"))
        
        self.red_channel.setText(_translate("MainWindow", "Red ch."))
        self.green_channel.setText(_translate("MainWindow", "Green ch."))
        self.monoexp_analysis.setText(_translate("MainWindow", "Monoexp. fit"))
        self.biexp_analysis.setText(_translate("MainWindow", "Biexp. fit"))
        
        self.groupBox.setTitle(_translate("MainWindow", "IRF"))
        self.exp_IRF_yn.setText(_translate("MainWindow", "Experimental IRF"))
        self.IRF_FWHM_label.setText(_translate("MainWindow", "IRF FWHM (ps)"))
        self.IRF_fwhm_sim.setText(_translate("MainWindow", "450"))
        self.simulated_IRF_yn.setText(_translate("MainWindow", "Simulated IRF"))
        self.IRF_filename_label.setText(_translate("MainWindow", "IRF filename:"))
        
        self.lifetime_ns_sim_label.setText(_translate("MainWindow", "Lifetime (ns)"))
        self.lifetime_sim.setText(_translate("MainWindow", "3"))
        self.num_photons_mono_simulation.setText(_translate("MainWindow", "200000"))
        self.num_photons_bi_simulation.setText(_translate("MainWindow", "200000"))        
        self.photons_sim_mono_label.setText(_translate("MainWindow", "# Photons"))
        self.photons_sim_bi_label.setText(_translate("MainWindow", "# Photons"))
        self.simulate_mono_decays.setText(_translate("MainWindow", "Simulate"))
        self.simulate_bi_decays.setText(_translate("MainWindow", "Simulate"))
        self.repeats_mono_simulation.setText(_translate("MainWindow", "20"))
        self.repeats_bi_simulation.setText(_translate("MainWindow", "20"))
        self.num_simulations_mono_label.setText(_translate("MainWindow", "# simulations"))
        self.num_simulations_bi_label.setText(_translate("MainWindow", "# simulations"))
        self.lifetime_1_sim_label.setText(_translate("MainWindow", "Lifetime 1 (ns)"))
        self.lifetime_1_sim.setText(_translate("MainWindow", "0.6"))        
        self.lifetime_2_sim_label.setText(_translate("MainWindow", "Lifetime 2 (ns)"))
        self.lifetime_2_sim.setText(_translate("MainWindow", "3"))
        self.a_1_sim_label.setText(_translate("MainWindow", "A1"))
        self.a_1_sim.setText(_translate("MainWindow", "0.2"))
        self.a_2_sim_label.setText(_translate("MainWindow", "A2"))
        self.a_2_sim.setText(_translate("MainWindow", str(1-float(self.a_1_sim.text()))))   
        
        self.groupBox_3.setTitle(_translate("MainWindow", ""))
        self.files_label.setText(_translate("MainWindow", "Files"))
        
        self.pulse_position_label.setText(_translate("MainWindow", "Pulse position (ns)"))
        self.lifetime_fit_label.setText(_translate("MainWindow", "<p>&tau;<sub>FL</sub> (ns)</p>")) 
        self.intensity_threshold_label.setText(_translate("MainWindow", "Int. threshold (Hz)"))
        self.macrotime_label.setText(_translate("MainWindow", "Macrotime limits (s)"))
        self.modify_initial_guess_label.setText(_translate("MainWindow", "Modify initial guess"))
        self.update_fit.setText(_translate("MainWindow", "Update fit"))
        self.reload_auto_fit.setText(_translate("MainWindow", "Reload automatic fit"))
        self.time1_threshold.setText(_translate("MainWindow", "0"))
        self.time2_threshold.setText(_translate("MainWindow", "1000"))
        
        self.modify_initial_guess_biexp_label.setText(_translate("MainWindow", "Modify initial guess"))
        self.pulse_position_biexp_label.setText(_translate("MainWindow", "Pulse position (ns)"))
        self.a1_fit_label.setText(_translate("MainWindow", "<p>a<sub>1</sub></p>"))  
        self.a2_fit_label.setText(_translate("MainWindow", "<p>a<sub>2</sub></p>"))
        self.lifetime1_fit_label.setText(_translate("MainWindow", "<p>&tau;<sub>FL, 1</sub> (ns)</p>"))  
        self.lifetime2_fit_label.setText(_translate("MainWindow", "<p>&tau;<sub>FL, 2</sub> (ns)</p>"))
        self.update_biexp_fit.setText(_translate("MainWindow", "Update fit"))
        self.reload_auto_biexp_fit.setText(_translate("MainWindow", "Reload automatic fit"))
        
        self.nanotime_label.setText(_translate("MainWindow", "Nanotime fit limits (ns)"))

        
        self.analyze_folder.setText(_translate("MainWindow", "Analyze entire folder"))
        self.export_results.setText(_translate("MainWindow", "Export results"))
        
        self.bin_macrotime_label.setText(_translate("MainWindow", "Bin macrotime (s)"))
        self.update_macrotime.setText(_translate("MainWindow", "Update"))
        self.fixed_intensity_threshold.setText(_translate("MainWindow", "Fix threshold"))
        
        
        self.bin_nanotime_label.setText(_translate("MainWindow", "Bin nanotime (ps)"))
        self.update_nanotime.setText(_translate("MainWindow", "Update"))
        
        self.fixed_nanotime_threshold.setText(_translate("MainWindow", "Fix range"))
        
        self.nanotime_plot_label.setText(_translate("MainWindow", "Nanotime plot limits"))
        self.nanotime_xplot_label.setText(_translate("MainWindow", "x (ns)"))
        self.nanotime_yplot_label.setText(_translate("MainWindow", "y (counts)"))
        self.update_plot_nanotime.setText(_translate("MainWindow", "Update \n plot"))
        self.fixed_plot_nanotime.setText(_translate("MainWindow", "Fix"))
        self.fixed_counts_plot_nanotime.setText(_translate("MainWindow", "Fix"))


  


#%% Functions

    def browse_file_IRF(self): # function that allows choosing the IRF file
        tk.Tk().withdraw()
        if self.IRF_dir_yn == False: 
            self.filename_IRF_full = askopenfilename()
        else:
            self.filename_IRF_full = askopenfilename(initialdir = self.IRF_dir)
        idx_folder = self.filename_IRF_full.rfind('/')
        self.filename_IRF_short = self.filename_IRF_full[(idx_folder+1):]
        self.IRF_dir = self.filename_IRF_full[:idx_folder]
        self.IRF_filename.setText(self.filename_IRF_short)
        self.IRF_browsed = True
        self.IRF_dir_yn = True
        self.IRF_ptu_yn = False
        
        if not '.dat' in self.filename_IRF_full:
            if '.ptu' in self.filename_IRF_full:
                if self.green_channel.isChecked() == True:
                    channel = 0
                else:
                    channel = 1
                micro_channel_IRF, macrotime_IRF = read_ptu(self.filename_IRF_full, channel, 'IRF')
                time_IRF = np.arange(0, np.max(micro_channel_IRF), 5E-11)
                time_IRF, counts_IRF = obtain_histogram_values(micro_channel_IRF, time_IRF)
                time_IRF = 1E9*time_IRF
                self.filename_IRF_ptu = list([micro_channel_IRF, time_IRF, counts_IRF])
                self.IRF_ptu_yn = True
                np.savetxt(self.filename_IRF_full[:-4]+'.csv', micro_channel_IRF)
            else:
                micro_channel_IRF = np.genfromtxt(self.filename_IRF_full[:-4]+'.csv')
                time_IRF = np.arange(0, np.max(micro_channel_IRF), 1E-11)
                time_IRF, counts_IRF = obtain_histogram_values(micro_channel_IRF, time_IRF)
                time_IRF = 1E9*time_IRF
                self.filename_IRF_csv = list([micro_channel_IRF, time_IRF, counts_IRF])
                self.IRF_csv_yn = True
                

    def browse_directory(self): # function that allows choosing the folder of the .fifo files
   
       self.list_of_filenames.clear()   
       tk.Tk().withdraw() 
       if self.files_dir_yn == False:   
           directory = askdirectory() 
       else:
           directory = askdirectory(initialdir = self.files_dir)
       os.chdir(directory)
       self.dir_path.setText(directory)
       idx = 0
       self.fileformat = self.file_format.currentText()
       for file in glob.glob("*"+self.fileformat):         
           idx += 1
           self.list_of_filenames.addItem(str(idx)+'. '+file) # the list is completed with all the .fifo files
       self.Simulate_data = 0 # If we open a directory, we are not simulating the data
       self.files_dir_yn = True
       self.files_dir = directory
       
    def Clicked_file(self): # Run analysis to clicked file

       filename_with_idx = self.list_of_filenames.currentItem().text()           
       idx_limit = filename_with_idx.index('.')
       filename = filename_with_idx[idx_limit+2:]
       self.box_file.setValue(int(filename_with_idx[:idx_limit]))
       FWHM_IRF =   float(self.IRF_fwhm_sim.text())
       
       if self.IRF_browsed == False and self.simulated_IRF_yn.isChecked() == False:   
           msg = QMessageBox()
           msg.setIcon(QMessageBox.Critical)
           msg.setText("Please select a file for the IRF")
           msg.exec_()
       else:
           if self.simulated_IRF_yn.isChecked() == False:
               if self.IRF_ptu_yn == True:
                   filename_IRF = self.filename_IRF_ptu
               else:
                   if self.IRF_csv_yn == True:
                       filename_IRF = self.filename_IRF_csv
                   else:
                       filename_IRF = self.filename_IRF_full
           else:
               filename_IRF = '.'
           if self.Simulate_data == 1:
               if self.monoexp_sim == True:
                   number_of_photons_sim = float(self.num_photons_mono_simulation.text())
                   lifetime_sim =  float(self.lifetime_sim.text())
               else:
                   number_of_photons_sim = float(self.num_photons_bi_simulation.text())
                   lifetime1_sim = float(self.lifetime_1_sim.text())
                   lifetime2_sim = float(self.lifetime_2_sim.text())  
                   lifetime_sim = np.array([lifetime1_sim, lifetime2_sim])
           else:
               lifetime_sim = 0
               number_of_photons_sim = 0
                   

           
           if len(self.nanotime1_threshold.text()) >0 and self.fixed_nanotime_threshold.isChecked():
               self.nanotime1_threshold_var = float(self.nanotime1_threshold.text())
               self.nanotime2_threshold_var = float(self.nanotime2_threshold.text())
           else:
               self.nanotime1_threshold_var = ''
               self.nanotime2_threshold_var = ''

           simulate_IRF_yn = self.simulated_IRF_yn.isChecked()
           if self.green_channel.isChecked() == True:
               channel = 0  
           else:
               channel = 1
               
           if self.monoexp_analysis.isChecked() == True:
               self.biexp_analysis_yn = False
               if len(self.lifetime_initial_guess.text()) > 0:
                   lifetime_guess =  1E-9*float(self.lifetime_initial_guess.text())
               else:
                   lifetime_guess = 0
               if len(self.pulse_position_initial_guess.text()) > 0:
                    IRF_pos_guess =  1E-9*float(self.pulse_position_initial_guess.text())
               else:
                    IRF_pos_guess = 0
               
               if not self.update_fit_yn:
                   self.time1_threshold.setText('0')          
               self.time1_threshold_var = float(self.time1_threshold.text())
               if self.time2_changed == True:
                   self.time2_threshold_var = float(self.time2_threshold.text())
               else:
                   self.time2_threshold_var = False
               a1_guess = 0
           else:
               self.biexp_analysis_yn = True
               if not self.update_fit_yn:
                   self.time1_threshold.setText('0')
               self.time1_threshold_var = float(self.time1_threshold.text()) 
               if self.time2_changed == True:
                   self.time2_threshold_var = float(self.time2_threshold.text())
               else:
                   self.time2_threshold_var = False
               if len(self.lifetime1_initial_guess.text()) > 0:
                   lifetime1_guess =  1E-9*float(self.lifetime1_initial_guess.text())
               else:
                   lifetime1_guess = 0
               if len(self.lifetime2_initial_guess.text()) > 0:
                    lifetime2_guess =  1E-9*float(self.lifetime2_initial_guess.text())
               else:
                    lifetime2_guess = 0    
               lifetime_guess = np.array([lifetime1_guess, lifetime2_guess])
               if len(self.a1_initial_guess.text()) > 0:
                   a1_guess =  float(self.a1_initial_guess.text())                   
               else:
                   a1_guess = 0       
               if len(self.pulse_position_biexp_initial_guess.text()) > 0:
                   IRF_pos_guess =  1E-9*float(self.pulse_position_biexp_initial_guess.text())
               else:
                   IRF_pos_guess = 0

           self.clicked = True # This variable let us know that we are already looking at some file
           
           self.bin_macrotime_trace = float(self.bin_macrotime.text())
           
           self.int_threshold = float(self.intensity_threshold.text())/(1/self.bin_macrotime_trace) # convert threshold (Hz) to true binning
           
           if not self.update_fit_yn and not self.fixed_intensity_threshold.isChecked():
               self.int_threshold = 0
               self.intensity_threshold.setText('0')
               
           
           if self.Simulate_data == 0:
               self.array_of_arrival_times_list = list()           
               

           self.bin_nanotime_var = 1E-11*float(self.bin_nanotime.text())
           print("AAA", self.bin_nanotime_var)
           channel = float(self.nanotime2_threshold.text()) # TODO: NEW LINE TO CHOOSE CHANNEL easyly
           print("CHANNEL", channel)

           results_analysis = analysis_data(self.biexp_analysis_yn,simulate_IRF_yn, 
                                                                 self.Simulate_data, channel, filename, 
                                                                 filename_IRF, FWHM_IRF, lifetime_sim, 
                                                                 number_of_photons_sim, self.int_threshold, 
                                                                 self.time1_threshold_var, self.time2_threshold_var,
                                                                 self.bin_macrotime_trace,
                                                                 self.nanotime1_threshold_var, self.nanotime2_threshold_var,
                                                                 a1_guess, lifetime_guess, IRF_pos_guess, 
                                                                 self.array_of_arrival_times_list, self.fileformat,
                                                                 self.bin_nanotime_var)
           self.min_lifetime = results_analysis[0]
           self.num_photons = results_analysis[1]
           self.a1 = results_analysis[2]
           self.min_IRF_pos = results_analysis[3]
           self.bins_histogram = results_analysis[4]
           self.values_wout_bg = results_analysis[5]
           self.fit = results_analysis[6]
           self.residuals = results_analysis[7]
           self.bins_macrotime = results_analysis[8]
           self.values_macrotime = results_analysis[9]
           self.max_time = results_analysis[10]
           self.max_macrotime = results_analysis[11]
           self.bins_histogram_data_IRF = results_analysis[12]
           self.values_histogram_data_IRF = results_analysis[13]
           self.bg_fit = results_analysis[14]         
           self.idx_on = results_analysis[15]
           self.idx_off = results_analysis[16]      
           self.bin_nanotime_var = results_analysis[17]
           self.bin_nanotime.setText(str(np.round(self.bin_nanotime_var*1E11,decimals=2)))
                                        
           if self.time2_changed == False: 
               self.time2_threshold_var = self.max_macrotime       
               
               
           self.time2_threshold.setText(str(np.round(self.time2_threshold_var, decimals = 4)))
           
           self.time2_changed = False # restore False value 


           
           self.values_histogram_data_IRF_shifted = shift_IRF(self.values_histogram_data_IRF, 
                                                              self.min_IRF_pos+0.2E-9,  
                                                              self.bins_histogram_data_IRF)
           
           self.values_histogram_data_IRF_shifted_norm = np.max(self.fit)*self.values_histogram_data_IRF_shifted/np.max(self.values_histogram_data_IRF_shifted)
           
           self.time_limit_IRF_idx = np.where(self.bins_histogram_data_IRF>np.max(self.bins_histogram))
           if len(self.time_limit_IRF_idx[0])>0:
                self.time_limit_IRF_idx = int(self.time_limit_IRF_idx[0][0])
           else:
               self.time_limit_IRF_idx = len(self.bins_histogram_data_IRF)
           
           # Plot results in widgets
           
           if self.analyze_entire_folder == False:       
               self.widget.canvas.ax.clear()
               self.widget.canvas.ax.plot(1E9*self.bins_histogram, self.values_wout_bg)
               self.widget.canvas.ax.plot(1E9*self.bins_histogram, self.fit)
               # self.widget.canvas.ax.plot(1E9*self.bins_histogram_data_IRF[:self.time_limit_IRF_idx], 
               #                            self.values_histogram_data_IRF_shifted_norm[:self.time_limit_IRF_idx], linestyle = 'dashed')
               if self.biexp_analysis_yn == False:
                   title_monoexp = r'$\tau$ = '+ str(np.round(1E9*self.min_lifetime, decimals = 2))
                   title_monoexp +=' ns; #photons = '+str(int(self.num_photons))
                   title_monoexp +='; pulse position = '
                   title_monoexp += str(np.round(1E9*self.min_IRF_pos, decimals = 2))+' ns'
                   self.widget.canvas.ax.set_title(title_monoexp)
               else:
                   title_biexp = r'$\tau_1$ = '+ str(np.round(1E9*self.min_lifetime[0], decimals = 2))+' ns; '
                   title_biexp += r'$\tau_2$ = '+ str(np.round(1E9*self.min_lifetime[1], decimals = 2))+' ns'
                   title_biexp += '\n'+'#photons = '
                   title_biexp +=  str(int(self.num_photons))+'; pulse position = '
                   title_biexp += str(np.round(1E9*self.min_IRF_pos, decimals = 2))+' ns \n'
                   title_biexp += r'$a_1$ = '+ str(np.round(self.a1, decimals = 2))
                   title_biexp += '; $a_2$ = '+ str(np.round((1-self.a1), decimals = 2))
                   
                   self.widget.canvas.ax.set_title(title_biexp)
               self.widget.canvas.ax.set_xlabel('Time [ns]')
               self.widget.canvas.ax.set_yscale('log')
               self.widget.canvas.ax.set_ylabel('Counts')
               
               if len(self.nanotime1_plot_threshold.text()) == 0 or not self.fixed_plot_nanotime.isChecked():
                   self.xlim0 = np.max(np.array([0, 1E9*self.min_IRF_pos-3]))
               else:
                   self.xlim0 = float(self.nanotime1_plot_threshold.text())
                   
               if len(self.nanotime2_plot_threshold.text()) == 0 or not self.fixed_plot_nanotime.isChecked():
                    self.xlim1 = 1E9*np.max(self.bins_histogram)
               else:
                    self.xlim1 = float(self.nanotime2_plot_threshold.text())
                    
               if len(self.nanotime1_counts_plot_threshold.text()) == 0 or not self.fixed_counts_plot_nanotime.isChecked():
                    self.ylim0 = 0.01*np.max(self.values_wout_bg)
               else:
                    self.ylim0 = float(self.nanotime1_counts_plot_threshold.text())
                    
               if len(self.nanotime2_counts_plot_threshold.text()) == 0 or not self.fixed_counts_plot_nanotime.isChecked():
                     self.ylim1 = 1.25*np.max(self.values_wout_bg)
               else:
                     self.ylim1 = float(self.nanotime2_counts_plot_threshold.text())
                   
               
               self.widget.canvas.ax.set_ylim(self.ylim0,self.ylim1)
               self.widget.canvas.ax.set_xlim((self.xlim0,self.xlim1))
               self.widget.canvas.draw()               
               
               
               self.nanotime1_plot_threshold.setText(str(np.round(self.xlim0, decimals=2)))
               self.nanotime2_plot_threshold.setText(str(np.round(self.xlim1, decimals=2)))
               self.nanotime1_counts_plot_threshold.setText(str(np.round(self.ylim0, decimals=2)))
               self.nanotime2_counts_plot_threshold.setText(str(np.round(self.ylim1, decimals=2)))
               
               
               self.widget_2.canvas.ax.clear()
               
               if self.Simulate_data == 0:
                   self.widget_2.canvas.ax.plot(self.bins_macrotime, self.values_macrotime*(1/self.bin_macrotime_trace))
                   if not type(self.idx_on) == int:
                        for i in range(len(self.idx_on)):
                            
                            self.widget_2.canvas.ax.axvspan(self.bins_macrotime[int(self.idx_on[i])],
                                                            self.bins_macrotime[int(self.idx_off[i])],
                                                            alpha=0.3,
                                                            color='red')

                        self.widget_2.canvas.ax.plot(np.array([np.min(self.bins_macrotime), 
                                                               np.max(self.bins_macrotime)]),
                                                     [(1/self.bin_macrotime_trace)*self.int_threshold,
                                                      (1/self.bin_macrotime_trace)*self.int_threshold], 
                                                     linestyle = 'dashed', color='purple')
                            
                   else:
                           
                       self.widget_2.canvas.ax.plot([self.time1_threshold_var, self.time1_threshold_var], 
                                                    np.array([np.min(self.values_macrotime), 
                                                              (1/self.bin_macrotime_trace)*np.max(self.values_macrotime)]),
                                                    color = 'violet')
                       self.widget_2.canvas.ax.plot([self.time2_threshold_var, 
                                                     self.time2_threshold_var], 
                                                    np.array([np.min(self.values_macrotime), 
                                                              (1/self.bin_macrotime_trace)*np.max(self.values_macrotime)]), 
                                                    color = 'violet')

                   self.widget_2.canvas.ax.set_xlabel('Time [s]')
                   self.widget_2.canvas.ax.set_ylabel('Counts [Hz]')
                   self.widget_2.canvas.draw()  
                               
               self.widget_3.canvas.ax.clear()
               self.widget_3.canvas.ax.plot(1E9*self.bins_histogram, self.residuals)
               self.widget_3.canvas.ax.set_ylabel('Residuals [Counts] ')
               self.widget_3.canvas.ax.set_xlabel('Time [ns]')
               self.widget_3.canvas.ax.set_xlim((self.xlim0,self.xlim1))
               self.widget_3.canvas.draw()  
               
               self.widget_4.canvas.ax.clear()
               self.widget_4.canvas.ax.plot(1E9*self.bins_histogram, self.values_wout_bg)
               self.widget_4.canvas.ax.set_ylabel('Counts ')
               self.widget_4.canvas.ax.set_xlabel('Time [ns]')
               self.widget_4.canvas.ax.set_title('Data (linear scale)')
               if isinstance(self.nanotime1_threshold_var,float):
                   xlim1 = self.nanotime1_threshold_var
               else:
                   xlim1 = 0
               if isinstance(self.nanotime2_threshold_var,float):
                   xlim2 = self.nanotime2_threshold_var  
               else:
                   xlim2 = np.max(1E9*self.bins_histogram)
               self.widget_4.canvas.ax.set_xlim((xlim1, xlim2))               
               self.widget_4.canvas.draw()  
               
               self.widget_5.canvas.ax.clear()
               self.widget_5.canvas.ax.plot(1E9*self.bins_histogram_data_IRF[:self.time_limit_IRF_idx], 
                                          self.values_histogram_data_IRF_shifted_norm[:self.time_limit_IRF_idx])
               self.widget_5.canvas.ax.set_ylabel('Counts ')
               self.widget_5.canvas.ax.set_xlabel('Time [ns]')
               self.widget_5.canvas.ax.set_title('IRF (linear scale)') 
               self.widget_5.canvas.ax.set_xlim((xlim1, xlim2))             
          
               self.widget_5.canvas.draw() 
           
           else:            
               self.export()
       
    def change_input_values_fit_manually(self): # update analysis considering the input values (initial guesses) introduced by the user
       if self.clicked == True:
           self.update_fit_yn = True
           self.Clicked_file()
           self.update_fit_yn = False

    def reload_auto_fit_func(self):
        self.lifetime_initial_guess.clear()
        self.pulse_position_initial_guess.clear()
        self.lifetime1_initial_guess.clear()
        self.lifetime2_initial_guess.clear()
        self.a1_initial_guess.clear()
        self.a2_initial_guess.clear()
        self.pulse_position_biexp_initial_guess.clear()
        
        self.Clicked_file()

    def box_file_func(self):
        if self.IRF_browsed == True:
            self.list_of_filenames.setCurrentRow(self.box_file.value()-1)       
       

    def simulate_data_function(self): # data simulation function
       self.list_of_filenames.clear()
       self.fileformat = 'sim'
       FWHM_IRF =   float(self.IRF_fwhm_sim.text())
       simulate_IRF_yn = self.simulated_IRF_yn.isChecked()

       if simulate_IRF_yn == False:
           if self.IRF_ptu_yn == False and self.IRF_csv_yn == False:
               filename_IRF = self.filename_IRF_full
           else:
               if self.IRF_ptu_yn == True:
                   filename_IRF = self.filename_IRF_ptu
               else:
                   filename_IRF = self.filename_IRF_csv
       else:
           filename_IRF = 'a'
       if int(self.groupBox_2.currentIndex()) == 0:
           number_of_photons_sim = float(self.num_photons_mono_simulation.text())
           Number_of_repeats = int(self.repeats_mono_simulation.text())
           lifetime_sim =  float(self.lifetime_sim.text())
           a1_sim = 0
           self.monoexp_sim = True
               
       else:
           number_of_photons_sim = float(self.num_photons_bi_simulation.text())
           Number_of_repeats = int(self.repeats_bi_simulation.text())
           lifetime1_sim = float(self.lifetime_1_sim.text())
           lifetime2_sim = float(self.lifetime_2_sim.text())  
           lifetime_sim = np.array([lifetime1_sim, lifetime2_sim])
           a1_sim = float(self.a_1_sim.text()) 
           self.monoexp_sim = False
       self.array_of_arrival_times_list = simulate_data_func(self.monoexp_sim, simulate_IRF_yn, filename_IRF, FWHM_IRF, lifetime_sim, a1_sim, number_of_photons_sim, Number_of_repeats)
   
       for i in range(Number_of_repeats):
            self.list_of_filenames.addItem(str(i+1)+'. Simulated_file_'+str(i+1))   
           

               
       
       self.Simulate_data = 1 # If we open a directory, we are not simulating the data
       
    def analyze_folder_func(self): # This analyzes all the files introduced in the list (all the fifo files from folder)
        self.analyze_entire_folder = True
        for i in range(int(self.list_of_filenames.count())):
            self.list_of_filenames.setCurrentRow(i) # Changing the item in the list will analyze each file        
        self.analyze_entire_folder = False
    
    def time2_changed_func(self):
        self.time2_changed = True
        
    def a1_changed_func(self):
        a2_float = 1-float(self.a_1_sim.text())
        a2_float = np.round(a2_float, decimals = 2)
        a2_str = str(a2_float)
        self.a_2_sim.setText(a2_str)  

    def a1_initial_guess_changed_func(self):
        a2_float = 1-float(self.a1_initial_guess.text())
        a2_float = np.round(a2_float, decimals = 2)
        a2_str = str(a2_float)
        self.a2_initial_guess.setText(a2_str)  
        
       
                        
        
    def export(self):
        if self.clicked == True:
            
            filename_with_idx = self.list_of_filenames.currentItem().text()
            idx_limit = filename_with_idx.index('.')
            filename = filename_with_idx[idx_limit+2:]
            
            print(filename_with_idx)
            
            plt.ion()
            fig = plt.figure()
    
    
            ax1 = plt.subplot(311)
            ax1.plot(1E9*self.bins_histogram, self.values_wout_bg)
            ax1.plot(1E9*self.bins_histogram, self.fit)
            ax1.set_xlabel('Time [ns]')
            ax1.set_ylim((0.01*np.max(self.values_wout_bg),1.25*np.max(self.values_wout_bg)))
            xlim0 = np.max(np.array([0, 1E9*self.min_IRF_pos-3]))
            ax1.set_xlim((xlim0,1E9*np.max(self.bins_histogram)))

            ax1.set_ylabel('Counts')
            ax1.set_yscale('log')
            ax1.set_title((r'$\tau$ = '+ str(np.round(1E9*self.min_lifetime, decimals = 2))+' ns; #photons = '+str(int(self.num_photons))+'; pulse position = '+ str(np.round(1E9*self.min_IRF_pos, decimals = 2))+' ns'))        
            
            ax2 = plt.subplot(312)
            ax2.plot(1E9*self.bins_histogram, self.residuals)
            ax2.set_xlabel('Time [ns]')
            ax2.set_ylabel('Counts')  
                
            ax3 = plt.subplot(313)
            ax3.set_title('Time trace')
            ax3.plot(self.bins_macrotime, (1/self.bin_macrotime_trace)*self.values_macrotime)
            if not type(self.idx_on) == int:
                for i in range(len(self.idx_on)):
                    ax3.axvspan(self.bins_macrotime[int(self.idx_on[i])],
                                                    self.bins_macrotime[int(self.idx_off[i])],
                                                    alpha=0.3,
                                                    color='red')
                ax3.plot(np.array([np.min(self.bins_macrotime), 
                                                       np.max(self.bins_macrotime)]),
                                             [(1/self.bin_macrotime_trace)*self.int_threshold,
                                              (1/self.bin_macrotime_trace)*self.int_threshold], 
                                             linestyle = 'dashed', color='purple')
                
            else:                
                ax3.plot([self.time1_threshold_var, self.time1_threshold_var], 
                                             np.array([np.min(self.values_macrotime), 
                                                       (1/self.bin_macrotime_trace)*np.max(self.values_macrotime)]),
                                             color = 'violet')
                ax3.plot([self.time2_threshold_var, 
                                              self.time2_threshold_var], 
                                             np.array([np.min(self.values_macrotime), 
                                                       (1/self.bin_macrotime_trace)*np.max(self.values_macrotime)]), 
                                             color = 'violet')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Counts [Hz]')            
        
            
            fig.set_size_inches(16, 16)
    
            plt.show()            
                            
            extension = self.fileformat
            
            if self.Simulate_data == 0:
                extension_position = filename.find(extension)
                filename = filename[:extension_position]            
                            
                if self.green_channel.isChecked() == True:
                    channel_name = 'green'
                else:
                    channel_name = 'red'
                    
                filename = filename + '_' + channel_name

            
            directory = os.getcwd()
            if not os.path.exists('Results'):
                os.makedirs('Results')
            plt.savefig(directory+'/Results/'+filename+'.pdf', format = 'pdf')
            
            plt.close()
            
            file1 = np.column_stack((self.bins_histogram, self.values_wout_bg, self.fit, self.residuals))
            file2 = np.column_stack((self.bins_macrotime, 100*self.values_macrotime))
            
            # Save individual results
            
                           
            np.savetxt(directory+'/Results/'+filename+'_rawdata_fit_residuals.csv',
                       file1, delimiter=',', header = 'Time_s, raw, fit, residuals', comments = '')
            np.savetxt(directory+'/Results/'+filename+'_macrotime_trace.csv', 
                       file2, delimiter=',', header = 'Time_s, Counts_Hz', comments = '')
            if self.biexp_analysis_yn == False:
                np.savetxt(directory+'/Results/'+filename+'_lifetime_ns.csv', 
                           np.array([1E9*self.min_lifetime]))                
            else:
                 np.savetxt(directory+'/Results/'+filename+'_lifetime_ns.csv', 
                            np.array([1E9*self.min_lifetime[0], 1E9*self.min_lifetime[1]]), 
                            delimiter=',')   
            
            # Add lifetime fitting results to list of lifetimes inside folder
            
            if not os.path.exists(directory+'/Results/'+'list_of_lifetimes_with_file_info.csv'):
                if self.biexp_analysis_yn == True:
                    d = {'filename': [filename],
                         'Lifetime1 (ns)': [1E9*self.min_lifetime[0]],
                         'Lifetime2 (ns)': [1E9*self.min_lifetime[1]],
                         'a1': [self.a1],
                         'a2': [1-self.a1],
                         'N': [int(self.num_photons)]}
                else:
                    d = {'filename': [filename],
                         'Lifetime1 (ns)': [1E9*self.min_lifetime],
                         'Lifetime2 (ns)': [0],
                         'a1': [0],
                         'a2': [0],
                         'N': [int(self.num_photons)]}
                        
                df = pd.DataFrame(data=d)
                df.to_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info.csv', index = False)
                np.savetxt(directory+'/Results/list_of_lifetimes_ns.csv', np.array([1E9*self.min_lifetime]))            

            else:
                df_global = pd.read_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info.csv')
                check_filename_saved =  np.where(df_global['filename'] == filename)
                if self.biexp_analysis_yn == True:
                    d = {'filename': [filename],
                         'Lifetime1 (ns)': [1E9*self.min_lifetime[0]],
                         'Lifetime2 (ns)': [1E9*self.min_lifetime[1]],
                         'a1': [self.a1],
                         'a2': [1-self.a1],
                         'N': [int(self.num_photons)]}
                else:
                    d = {'filename': [filename],
                         'Lifetime1 (ns)': [1E9*self.min_lifetime],
                         'Lifetime2 (ns)': [0],
                         'a1': [0],
                         'a2': [0],
                         'N': [int(self.num_photons)]}
                df = pd.DataFrame(data=d)
                if len(check_filename_saved[0]) == 0:
                    df_global = df_global.append(df)                    
                else:
                    df_global.reset_index(drop=True, inplace=True)
                    df_global = df_global.drop(check_filename_saved[0][0])
                    df_global = df_global.append(df)                 

                df_global_sorted = df_global.sort_values(by = 'filename')
                df_global_sorted.to_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info.csv', index = False)      
                lifetime_list = df_global_sorted['Lifetime1 (ns)'].to_numpy()
                np.savetxt(directory+'/Results/list_of_lifetimes_ns.csv', lifetime_list)
                
                
                # Add lifetime fitting results to list of lifetimes inside folder (separated channels)
                
                if self.fileformat == '.fifo':
                    if self.green_channel.isChecked() == True:
                        if not os.path.exists(directory+'/Results/'+'list_of_lifetimes_with_file_info_green.csv'):
                            df.to_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info_green.csv', index = False)
                        else:
                            dfg_global = pd.read_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info_green.csv')
                            check_filename_saved =  np.where(dfg_global['filename'] == filename)
                            if len(check_filename_saved[0]) == 0:
                                dfg_global = dfg_global.append(df)                    
                            else:
                                dfg_global.reset_index(drop=True, inplace=True)
                                dfg_global = dfg_global.drop(check_filename_saved[0][0])
                                dfg_global = dfg_global.append(df)                 
            
                            dfg_global_sorted = dfg_global.sort_values(by = 'filename')
                            dfg_global_sorted.to_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info_green.csv', index = False) 
                    else:
                        if not os.path.exists(directory+'/Results/'+'list_of_lifetimes_with_file_info_red.csv'):
                            df.to_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info_red.csv', index = False)
                        else:
                            dfr_global = pd.read_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info_red.csv')
                            check_filename_saved =  np.where(dfr_global['filename'] == filename)
                            if len(check_filename_saved[0]) == 0:
                                dfr_global = dfr_global.append(df)                    
                            else:
                                dfr_global.reset_index(drop=True, inplace=True)
                                dfr_global = dfr_global.drop(check_filename_saved[0][0])
                                dfr_global = dfr_global.append(df)                 
            
                            dfr_global_sorted = dfr_global.sort_values(by = 'filename')
                            dfr_global_sorted.to_csv(directory+'/Results/'+'list_of_lifetimes_with_file_info_red.csv', index = False)  
 
                

from mplwidget import MplWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())