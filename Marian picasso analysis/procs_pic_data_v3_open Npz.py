# -*- coding: utf-8 -*-
"""
Created on Tuesday Novemeber 16 2021

@author: Mariano Barella

Version 2. Changes:
    - new flag that select if structures are origamis or hybridized structures
    - automatically selects the best threshold of the peak finding algorithm

This script analyzes already-processed Picasso data. It opens .dat files that 
were generated with "extract_and_save_data_from_hdf5_picasso_files.py".

When the program starts select ANY .dat file. This action will determine the 
working folder.

As input it uses:
    - main folder
    - number of frames
    - exposure time
    - if NP is present (hybridized structure)
    - pixel size of the original video
    - size of the pick you used in picasso analysis pipeline
    - desired radius of analysis to average localization position
    - number of dokcing sites you are looking foor (defined by origami design)
    
Outputs are:
    - plots per pick (scatter plot of locs, fine and coarse 2D histograms,
                      binary image showing center of docking sites,
                      matrix of relative distances, matrix of localization precision)
    - traces per pick
    - a single file with ALL traces of the super-resolved image
    - global figures (photons vs time, localizations vs time and background vs time)

Warning: the program is coded to follow filename convention of the script
"extract_and_save_data_from_hdf5_picasso_files.py".

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Circle as plot_circle
import os
import re
import tkinter as tk
import tkinter.filedialog as fd
from auxiliary_functions import detect_peaks, distance, fit_linear, \
    perpendicular_distance, manage_save_directory, position_peaks, \
        matrix_calculation, plot_matrix_distance
#%%
import numpy as np
import matplotlib.pyplot as plt
import time as time
import os

from tkinter import Tk, filedialog


root = Tk()
nametoload = filedialog.askopenfilename(filetypes=(("", "*.npz"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(nametoload)
only_name = os.path.basename(nametoload)

# all_detected_peaks[str(i)] =  (cm_binding_sites_x[final_order],cm_binding_sites_y[final_order])

loaded_data = np.load(nametoload)

#print("time one load = ", time.time()-tic)
indexs = []
for k,values in loaded_data.items():
    indexs.append(k)
    print(k,values)
# plt.plot(loaded_data["40"])
# plt.show()
data = dict()
for i in range(len(indexs)):
    data[int(indexs[i])] = loaded_data[str(indexs[i])]
    
# plt.plot(data[40])
# plt.show()

print("\n")
#%%

lengths = dict()
distances_from_1 = dict()
try:
    i_min = int(indexs[0])
    for i in range(i_min,len(indexs)+i_min):
        index_peaks = data[i]
        total_peaks_found = index_peaks.shape[1]
        distances_nm = np.zeros((total_peaks_found,total_peaks_found))
        for j in range(total_peaks_found):
            for k in range(total_peaks_found):
                distances_nm[k,j] = distance(index_peaks[0][j], index_peaks[1][j], index_peaks[0][k], index_peaks[1][k])*1e3
        # print(i, distances_nm)
        distances_from_1[i] = distances_nm[0][np.nonzero(distances_nm[0])]
        lengths[i] =  np.sort(distances_nm.ravel())[np.nonzero(np.sort(distances_nm.ravel()))]
except:
    print("except, " , i) 
    pass

    # plt.hist(lengths[i])
# plt.show()
#%%
total_length = []

# for l in range(5*5):
#     total_length[l] = [0]
for i in lengths.keys():
    for j in range(len(lengths[i])):
        # print(i, j)
        total_length.append(lengths[i][j])  # WRONG!!!! The ravel vector have repeated distances

total_distance1 = []
for i in distances_from_1.keys():
    for j in range(len(distances_from_1[i])):

        total_distance1.append(distances_from_1[i][j])  # WRONG!!!! The ravel vector have repeated distances


plt.hist(total_length, label="amoun of data={}".format(len(total_length)),color="cornflowerblue", edgecolor="darkblue", bins=50)

plt.hist(total_distance1, label="amoun of data={}".format(len(total_distance1)),color="orange", edgecolor="red", bins=50)

# plt.hist(total_length_True, label="amoun of data={}".format(len(total_length_True)),color="orange", edgecolor="red", bins=50)
# for l in range(1,9):
#     plt.hist(total_length[l],label="length {}".format(l), bins=5, alpha=0.9)
plt.legend()
# plt.xlim(0,max(total_length) + min(total_length))
plt.show()


