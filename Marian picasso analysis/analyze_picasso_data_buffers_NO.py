# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:24:34 2021

@author: Mariano Barella

This script analyzes the output of Picasso software. 
It opens .dat files that are generated with 
"extract_and_save_data_from_hdf5_picasso_files.py".

As input it uses the number of frames and the exposure time.

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import re
# from tkinter import Tk, filedialog

plt.ioff()
plt.close("all")    

# load and open file
folder = "C:\\datos_mariano\\posdoc\\DNA-PAINT\\data_fribourg\\buffer_analysis"
save_folder = folder
list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('.dat',f)]
list_of_files.sort()
L = len(list_of_files)

data = {}
data_name = {}
for i in range(L):
    filename = list_of_files[i]
    filepath = os.path.join(folder, filename)
    data_all = np.loadtxt(filepath)
    # remove locs that ocurred after 15 min
    to_keep = np.where(data_all <= 9000)
    data[i] = data_all[to_keep]
    data_name[i] = filename
    # print(max(data[i]))

#%%

number_of_picks = np.array(
    [637, #1xTAE12
     356, #Trolox/Glox
     668, #Trolox/10Glox
     685, #Trolox/50Glox
     479, #PPT 
     526, #COT/Glox
     393, #Trolox only
     332, #Trolox+561
     316, #PPT
     544]) #PPT+561

labels = ['1xTAE12', 
          'Trolox/Glox',
          'Trolox/10Glox',
          'Trolox/50Glox',
          'PPT',      
          'COT/Glox',
          'Trolox only',
          'Trolox+561',
          'PTT',
          'PTT+561']
    
#################### INPUTS ####################
#################### INPUTS ####################
#################### INPUTS ####################

# time parameters
# number of frame to analyze
number_frames = 9000
exp_time = 0.1 # in s
total_time_sec = number_frames*exp_time # in sec
total_time_min = total_time_sec/60 # in min
print('Total time %.1f min' % total_time_min)

# locs/pick vs buffer
total_locs = np.array([sum(data[i]) for i in range(L)])
total_locs_per_pick = total_locs/number_of_picks
plt.figure(1)
plt.bar(range(L), total_locs_per_pick/1e6, color='C0')
ax = plt.gca()
ax.set_xticks(range(L))
ax.set_xticklabels(labels, rotation=60, fontsize=12)
ax.set_ylabel('10$^{6}$ counts')
ax.set_title('Number of locs per pick in 15 min')
figure_name = 'total_locs_normalized'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

# make histogram of locs  
number_of_bins = 10
hist_range = [0, number_frames]
bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
bin_size_seconds = bin_size*exp_time
bin_size_minutes = bin_size_seconds/60
print('Bin size %.1f s' % bin_size_seconds)
locs_per_bin = {}
locs_per_bin_per_pick = {}
for i in range(L):
    locs_per_bin[i], bin_edges = np.histogram(data[i], bins = number_of_bins, range=hist_range)
    locs_per_bin_per_pick = locs_per_bin[i]/number_of_picks[i]
    
    # plot
    bin_centers = bin_edges[:-1] + bin_size/2
    bin_centers_minutes = bin_centers*exp_time/60
    plt.figure(2)
    plt.plot(bin_centers_minutes, locs_per_bin_per_pick, label = labels[i])

plt.figure(2)
plt.legend(loc='upper right')
plt.xlabel('Time (min)')
plt.ylabel('Counts')
ax = plt.gca()
ax.set_title('Number of locs per pick vs time. Bin size %d min' % bin_size_minutes)
figure_name = 'locs_per_pick_vs_time'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')

plt.close()
plt.show()


