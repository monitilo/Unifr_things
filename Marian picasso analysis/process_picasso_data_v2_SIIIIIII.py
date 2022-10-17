
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
    
plt.ioff()
plt.close("all")    
cmap = plt.cm.get_cmap('viridis')
bkg_color = cmap(0)

##############################################################################
# INPUTS

# base folder to select data
base_folder = 'C:/Projects/Super resolution/Photons quenching two color super res/Marian code test/SametAAA'
# docking site per origami
docking_sites = 4
# is there any NP (hybridized structure)
NP_flag = False
# camera pixel size
pixel_size = 0.065 # 0.130 # in um
# size of the pick used in picasso
pick_size = 3 # in camera pixels (put the same number used in Picasso)
# size of the pick to include locs around the detected peaks
radius_of_pick_to_average = 0.25 # 0.25 # in camera pixel size
# set an intensity threshold to avoid dumb peak detection in the background
# this threshold is arbitrary, don't worry about this parameter, the code 
# change it automatically to detect the number of docking sites set above
th = 30
# time parameters
number_of_frames = 10000
exp_time = 0.1 # in s

##############################################################################
# PROGRAM STARTS

total_time_sec = number_of_frames*exp_time # in sec
total_time_min = total_time_sec/60 # in min
print('Video Total time %.1f min' % total_time_min)

# select any file (will use the selected folder actually)
root = tk.Tk()
dat_files = fd.askopenfilenames(initialdir = base_folder, 
                                      filetypes=(("", "*.dat"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(dat_files[0])

# create folder to save data
# global figures folder
figures_folder = manage_save_directory(folder, 'figures_global')
# figures per pick folder
per_pick_folder = os.path.join(folder, 'per_pick')
figures_per_pick_folder = manage_save_directory(per_pick_folder, 'figures')
traces_per_pick_folder = manage_save_directory(per_pick_folder, 'traces')

# list files
list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('.dat',f)]
list_of_files.sort()
if NP_flag:
    list_of_files_origami = [f for f in list_of_files if re.search('combined',f)]
    list_of_files_NP = [f for f in list_of_files if re.search('twoCh',f)]
else:
    list_of_files_origami = [f for f in list_of_files if re.search('Single',f)]

##############################################################################
# load data

# frame number, used for time estimation
print(list_of_files_origami)
frame_file = [f for f in list_of_files_origami if re.search('_frame',f)][0]
frame_filepath = os.path.join(folder, frame_file)
frame = np.loadtxt(frame_filepath)

# photons
photons_file = [f for f in list_of_files_origami if re.search('_photons',f)][0]
photons_filepath = os.path.join(folder, photons_file)
photons = np.loadtxt(photons_filepath)

# bkg
bkg_file = [f for f in list_of_files_origami if re.search('_bkg',f)][0]
bkg_filepath = os.path.join(folder, bkg_file)
bkg = np.loadtxt(bkg_filepath)

# xy positions
# origami
position_file = [f for f in list_of_files_origami if re.search('_xy',f)][0]
position_filepath = os.path.join(folder, position_file)
position = np.loadtxt(position_filepath)
x = position[:,0]*pixel_size
y = position[:,1]*pixel_size
# NP
if NP_flag:
    position_file_NP = [f for f in list_of_files_NP if re.search('_xy',f)][0]
    position_filepath_NP = os.path.join(folder, position_file_NP)
    position_NP = np.loadtxt(position_filepath_NP)
    xy_NP = np.loadtxt(position_filepath_NP)
    x_NP = xy_NP[:,0]*pixel_size
    y_NP = xy_NP[:,1]*pixel_size

# number of pick
# origami
pick_file = [f for f in list_of_files_origami if re.search('_pick_number',f)][0]
pick_filepath = os.path.join(folder, pick_file)
pick_list = np.loadtxt(pick_filepath)
# NP
if NP_flag:
    pick_file_NP = [f for f in list_of_files_NP if re.search('_pick_number',f)][0]
    pick_filepath_NP = os.path.join(folder, pick_file_NP)
    pick_list_NP = np.loadtxt(pick_filepath_NP)

data_length = len(x)
##############################################################################

# how many picks?
pick_number = np.unique(pick_list)
total_number_of_picks = len(pick_number)
print('Total picks', total_number_of_picks)

# allocate arrays for statistics
locs_of_picked = np.zeros(total_number_of_picks)
# number of bins for temporal binning
number_of_bins = 60
locs_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
photons_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
bkg_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
photons_concat = np.array([])
bkg_concat = np.array([])
frame_concat = np.array([])
positions_concat_NP = np.array([])
positions_concat_origami = np.array([])
all_traces = np.zeros(number_of_frames)

all_detected_peaks = dict()
# data assignment per pick

photons_delete = True

i_min = 0
for i in range(i_min, 100):  #total_number_of_picks):
    # print(i)
    pick_id = pick_number[i]
    print('\n---------- Pick number %d \n' % i)
    index_picked = np.where(pick_list == pick_id)
    # for origami
    frame_of_picked = frame[index_picked]
    photons_of_picked = photons[index_picked]
    bkg_of_picked = bkg[index_picked]
    x_position_of_picked = x[index_picked]
    y_position_of_picked = y[index_picked]
    L = len(x_position_of_picked)
    # make FINE 2D histogram of locs
    # set number of bins
    N = int(2*pick_size*pixel_size*1000/10)
    hist_2D_bin_size = pixel_size*1000*pick_size/N # this should be around 5 nm
    x_min = min(x_position_of_picked)
    y_min = min(y_position_of_picked)
    x_max = x_min + pick_size*pixel_size
    y_max = y_min + pick_size*pixel_size
    z_hist, x_hist, y_hist = np.histogram2d(x_position_of_picked, 
                                            y_position_of_picked, 
                                            bins = N, 
                                            range = [[x_min, x_max], \
                                                     [y_min, y_max]])
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose z_hist for visualization purposes.
    z_hist = z_hist.T
    x_hist_step = np.diff(x_hist)
    y_hist_step = np.diff(y_hist)
    x_hist_centers = x_hist[:-1] + x_hist_step/2
    y_hist_centers = y_hist[:-1] + y_hist_step/2
    total_peaks_found = 10
    threshold_COARSE = 100 # th
    while total_peaks_found > 5 or total_peaks_found < 2:  # docking_sites
        # make COARSE 2D histogram of locs
        # number of bins is arbitrary, determined after trial and error
        bins_COARSE = 25
        x_min = min(x_position_of_picked)
        y_min = min(y_position_of_picked)
        x_max = x_min + pick_size*pixel_size
        y_max = y_min + pick_size*pixel_size
        z_hist_COARSE, x_hist_COARSE, y_hist_COARSE = np.histogram2d(x_position_of_picked, 
                                                                     y_position_of_picked, 
                                                                     bins = bins_COARSE,
                                                                     range = [[x_min, x_max], \
                                                                              [y_min, y_max]],
                                                                     density = True)
        z_hist_COARSE = z_hist_COARSE.T
        x_hist_step_COARSE = np.diff(x_hist_COARSE)
        y_hist_step_COARSE = np.diff(y_hist_COARSE)
        x_hist_COARSE_centers = x_hist_COARSE[:-1] + x_hist_step_COARSE/2
        y_hist_COARSE_centers = y_hist_COARSE[:-1] + y_hist_step_COARSE/2
        z_hist_COARSE = np.where(z_hist_COARSE < threshold_COARSE, 0, z_hist_COARSE)
        
        # peak detection for Center of Mass localization
        detected_peaks = detect_peaks(z_hist_COARSE)
        # find Center of Mass of locs near the peaks that were found
        index_peaks = np.where(detected_peaks == True) # this is a tuple
        total_peaks_found = len(index_peaks[0])
        
        if total_peaks_found <= 5:
            distances_index = np.zeros((total_peaks_found,total_peaks_found))
            old_indexing = np.arange(0,total_peaks_found)
            delete_indexing = []
            for j in range(total_peaks_found):
                for k in range(total_peaks_found):
                    distances_index[k,j] = distance(index_peaks[0][j], index_peaks[1][j], index_peaks[0][k], index_peaks[1][k])

            for j in range(total_peaks_found):
                if (distances_index[j][np.nonzero(distances_index[j])] < 3).any():
                    # print(j, distances_index[j])
                    delete_indexing.append(j)

            final_indexing = []
            for j in range(len(old_indexing)):
                if j not in delete_indexing[1::2]:
                    final_indexing.append(old_indexing[j])
            
            total_peaks_found = len(final_indexing)
            index_peaks = [index_peaks[0][final_indexing], index_peaks[1][final_indexing]]

#
        threshold_COARSE += 5
        if threshold_COARSE > 5000:
            # this MAX value is arbitrary
            break
    print('threshold_COARSE reached', threshold_COARSE)

    print(total_peaks_found, 'total peaks found\n')
    if total_peaks_found < 2:
        peaks_flag = False
        continue
    else:
        peaks_flag = True

    analysis_radius = radius_of_pick_to_average*pixel_size
    cm_binding_sites_x = np.array([])
    cm_binding_sites_y = np.array([])
    cm_std_dev_binding_sites_x = np.array([])
    cm_std_dev_binding_sites_y = np.array([])
    # array where traces are going to be saved
    # all_traces_per_pick = np.zeros(number_of_frames)
    no_order = np.arange(0, total_peaks_found)
    (cm_binding_sites_x, cm_binding_sites_y,
      cm_std_dev_binding_sites_x, cm_std_dev_binding_sites_y,
      frame_of_picked_filtered, photons_of_picked_filtered, all_traces_per_pick) = position_peaks(no_order, analysis_radius, number_of_frames, index_peaks,
                        x_hist_COARSE_centers,y_hist_COARSE_centers,
                        x_position_of_picked,y_position_of_picked, frame_of_picked,photons_of_picked)

    #%%    
    # plt.plot(all_traces_per_pick)
    # plt.plot(all_traces_per_pick[:,2])
    # plt.show()
    #%%
    # save traces per pick
    if peaks_flag:
        new_filename = 'TRACES_pick_%02d_%s.txt' % (i, frame_file[:-10])
        new_filepath = os.path.join(traces_per_pick_folder, new_filename)
        np.savetxt(new_filepath, all_traces_per_pick, fmt='%05d')

    
    # fit linear (origami direction) of the binding sites 
    # to find the perpendicular distance to the NP
    x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(cm_binding_sites_x, 
                                                                cm_binding_sites_y)
    
# =============================================================================
#     # calculate relative distances between all points
#     # ------------------ in nanometers -----------------------
#     # # allocate: total size = number of detected peaks + 1 for NP
#     # matrix_distance = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
#     # matrix_std_dev = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
#     # Me not use NP (Germán)
#     matrix_distance = np.zeros([total_peaks_found, total_peaks_found])
#     matrix_std_dev = np.zeros([total_peaks_found, total_peaks_found])
#     # calcualte first row of the matrix distance
#     # calcualte the rest of the rows of the matrix distance
#     for j in range(total_peaks_found):
#         x_binding_row = cm_binding_sites_x[j]
#         y_binding_row = cm_binding_sites_y[j]
#         matrix_std_dev[j, j] = max(cm_std_dev_binding_sites_x[j], \
#                                            cm_std_dev_binding_sites_y[j])*1e3
#         for k in range(j, total_peaks_found):
#             x_binding_col = cm_binding_sites_x[k]
#             y_binding_col = cm_binding_sites_y[k]
#             distance_between_locs_CM = distance(x_binding_col, y_binding_col, \
#                                               x_binding_row, y_binding_row)*1e3
#             matrix_distance[j, k] = distance_between_locs_CM
#             matrix_distance[k, j] = distance_between_locs_CM
#             positions_concat_origami = np.append(positions_concat_origami, distance_between_locs_CM)
# =============================================================================

    (matrix_distance, matrix_std_dev,positions_concat_origami) = matrix_calculation(no_order, cm_binding_sites_x,
                        cm_binding_sites_y, cm_std_dev_binding_sites_x, cm_std_dev_binding_sites_y)
#%%

    # plot matrix distance   
    plot_matrix_distance (matrix_distance, i, 10)
    # plt.figure(10)
    # plt.imshow(matrix_distance, interpolation='none', cmap='spring')
    # ax = plt.gca()
    # for l in range(matrix_distance.shape[0]):
    #     for m in range(matrix_distance.shape[1]):
    #         if l == m:
    #             ax.text(m, l, '%.0f' % matrix_distance[l, m], #, '-' ,
    #                 ha="center", va="center", color=[0,0,0], 
    #                 fontsize = 18)
    #         else:
    #             ax.text(m, l, '%.0f' % matrix_distance[l, m],
    #                 ha="center", va="center", color=[0,0,0], 
    #                 fontsize = 18)
    # ax.xaxis.tick_top()
    # ax.set_xticks(np.array(range(matrix_distance.shape[1])))
    # ax.set_yticks(np.array(range(matrix_distance.shape[0])))
    # axis_string = []  # ['NP']
    # for j in range(matrix_distance.shape[1]):
    #     axis_string.append('Site %d' % (j+1))
    # ax.set_xticklabels(axis_string)
    # ax.set_yticklabels(axis_string)
    aux_folder = manage_save_directory(figures_per_pick_folder,'matrix_distance')
    figure_name = 'order_matrix_distance_%02d' % i
    figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
    plt.show()
    plt.close()

     #%%   
    # plot matrix of max std dev   
        # locs_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])

    plt.figure(11)
    plt.imshow(matrix_std_dev, interpolation='none', cmap='spring')
    ax = plt.gca()
    for l in range(matrix_distance.shape[0]):
        for m in range(matrix_distance.shape[1]):
            if not l == m:
                ax.text(m, l, '-' ,
                    ha="center", va="center", color=[0,0,0], 
                    fontsize = 18)
            else:
                ax.text(m, l, '%.0f' % matrix_std_dev[l, m],
                    ha="center", va="center", color=[0,0,0], 
                    fontsize = 18)
    ax.xaxis.tick_top()
    ax.set_xticks(np.array(range(matrix_distance.shape[1])))
    ax.set_yticks(np.array(range(matrix_distance.shape[0])))
    axis_string = []
    for j in range(total_peaks_found):
        axis_string.append('Site %d' % (j+1))
    ax.set_xticklabels(axis_string)
    ax.set_yticklabels(axis_string)
    aux_folder = manage_save_directory(figures_per_pick_folder,'matrix_std_dev')
    figure_name = 'matrix_std_dev_%02d' % i
    figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 100, bbox_inches='tight')

    # plt.show()
    plt.close()

    number_of_bins =60
    # plots of the binding sites
    photons_concat = np.concatenate([photons_concat, photons_of_picked])
    bkg_concat = np.concatenate([bkg_concat, bkg_of_picked])
    frame_concat = np.concatenate([frame_concat, frame_of_picked])
    locs_of_picked[i] = len(frame_of_picked)
    hist_range = [0, number_of_frames]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    locs_of_picked_vs_time[i,:], bin_edges = np.histogram(frame_of_picked, bins = number_of_bins, range = hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    bin_centers_minutes = bin_centers*exp_time/60 

    #%% 2nd try , Using Photons.
    
    new_peaks = []  # in_order[0
    photons_order = np.argsort(np.sum(all_traces_per_pick, axis=0))[::-1]
    if photons_delete:
        for l in photons_order:
                if np.sum(all_traces_per_pick, axis=0)[l] < 0.24*np.mean(np.sum(all_traces_per_pick, axis=0)):
                    print ("DELETEANDO because of photons",l, np.sum(all_traces_per_pick, axis=0)[l]/np.mean(np.sum(all_traces_per_pick, axis=0)))
                    photons_order = np.delete(photons_order,np.where(photons_order==l))
        if len(photons_order) < 2:
            continue
    in_order = np.copy(photons_order)

    x_coord_check = np.zeros((len(in_order)))
    y_coord_check = np.zeros((len(in_order)))
    x_line = np.linspace(np.min(cm_binding_sites_x), np.max(cm_binding_sites_x), 100)
    # A (x-x0) + y0
    A = (cm_binding_sites_y[in_order[1]]-cm_binding_sites_y[in_order[0]])/(cm_binding_sites_x[in_order[1]]-cm_binding_sites_x[in_order[0]])
    y0 = np.min(cm_binding_sites_y[in_order[0]])  # cm_binding_sites_y[0]
    x0 = cm_binding_sites_x[in_order[0]]  # cm_binding_sites_x[0]
    y_line = A * (x_line-x0) + y0
    tolerance = 0.05*(cm_binding_sites_y[in_order[0]]/cm_binding_sites_y[in_order[1]])
    
    x_spot, y_spot = ([x0,x0,x_line[-1],x_line[-1]],
                      [y0-tolerance, y0+tolerance, y_line[-1]+tolerance,y_line[-1]-tolerance])
    real_peaks = []

    newindex = 0
    for p in range(len(in_order)):
        x_coord_check[p] = min(x_line, key=lambda x:abs(x - cm_binding_sites_x[in_order[p]]))  ## THIS IS THE WAY!!
        y_coord_check[p] = y_line[np.argwhere(x_line==x_coord_check[p])[0][0]]
        # y_coord_check = min(y_line, key=lambda x:abs(x - cm_binding_sites_y[p]))        
        if (cm_binding_sites_y[in_order[p]] < y_coord_check[p]-tolerance) or (cm_binding_sites_y[in_order[p]] > y_coord_check[p]+tolerance):
            print("BAAAD", p)
        else:
            print("good", p)
            real_peaks.append(in_order[p])
    if len(real_peaks) < 2:
        continue

    plt.plot(x_coord_check,y_coord_check,"--o", markersize=10, alpha=0.9, markeredgecolor="Blue", markerfacecolor="green", label="Proyection over Line")
    plt.plot(cm_binding_sites_x,cm_binding_sites_y,marker="o", markersize=10, alpha=0.9, markeredgecolor="red", markerfacecolor="green", label="CM position")
total_length    plt.legend()
    plt.show()
    print( "real peaks ", real_peaks)
        # y_coord_check
    x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(cm_binding_sites_x[real_peaks], 
                                                                cm_binding_sites_y[real_peaks])
#%%
    (matrix_distance, matrix_std_dev, positions_concat_origami) = matrix_calculation(real_peaks, cm_binding_sites_x,
                        cm_binding_sites_y, cm_std_dev_binding_sites_x, cm_std_dev_binding_sites_y)
    plot_matrix_distance (matrix_distance, i, 15)
    plt.show()
# %%
    min_distance = np.zeros(((matrix_distance.shape[1])))
    max_distance = np.zeros((matrix_distance.shape[1]))
    double_point = False
    for d in range(matrix_distance.shape[1]):
        # print(d)
        max_distance[d] = np.max(matrix_distance[d][np.nonzero(matrix_distance[d])])
        if np.min(matrix_distance[d][np.nonzero(matrix_distance[d])]) < 15:          
            if double_point == False:
                min_distance[d] = np.min(matrix_distance[d][np.nonzero(matrix_distance[d])])
                double_point = True
            else:
                min_distance[d] = 0
                print("there is a double point",d)  
            
        else:
            min_distance[d] = np.min(matrix_distance[d][np.nonzero(matrix_distance[d])])
            
    
    if double_point:
        min_order = np.argsort(min_distance)[1:]

    else:
        min_order = np.argsort(min_distance)
    max_order = np.argsort(max_distance)[::-1]

    final_order = np.copy(min_order)  
    if len(final_order) < 2:
        continue
    if min_order[1] == max_order[0] or min_order[1] == max_order[1] :
        print("inverted")
        final_order[0], final_order[1] = final_order[1], final_order[0]
    for o in range(len(final_order)):
        final_order[o] = real_peaks[final_order[o]]
    print("final order= ", final_order, "real_peaks",real_peaks)

#%%
    (matrix_distance, matrix_std_dev, positions_concat_origami) = matrix_calculation(final_order, cm_binding_sites_x,
                        cm_binding_sites_y, cm_std_dev_binding_sites_x, cm_std_dev_binding_sites_y)
    plot_matrix_distance (matrix_distance, i, "pepe")
    plt.show()
#%%
    # plot when the pick was bright vs time
    if False:
        plt.figure()
        # plt.plot(bin_centers, locs_of_picked_vs_time, label = 'Pick %04d' % i)
        plt.step(bin_centers_minutes, locs_of_picked_vs_time[i,:], where = 'mid', label = 'Pick %04d' % i)
        # plt.legend(loc='upper right')
        plt.xlabel('Time (min)')
        plt.ylabel('Locs')
        plt.ylim([0, 80])
        ax = plt.gca()
        ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '2', linestyle = '--')
        ax.set_title('Number of locs per pick vs time. Bin size %.1f min' % (bin_size*0.1/60))
        aux_folder = manage_save_directory(figures_per_pick_folder,'locs_vs_time_per_pick')
        figure_name = 'locs_per_pick_vs_time_pick_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
        plt.close()
#%%
    # plot xy coord of the pick in several ways, including the peaks detected
    if True:
        # plot all RAW
        plt.figure(1)
        plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'PAINT')
        if NP_flag:
            plt.plot(x_position_of_picked_NP, y_position_of_picked_NP, '.', color = 'C1', alpha = 0.5, label = 'NP')
            plt.plot(x_avg_NP, y_avg_NP, 'x', color = 'k',  label = 'Avg position NP')
            plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('square')
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        aux_folder = manage_save_directory(figures_per_pick_folder,'scatter_plots')        
        figure_name = 'xy_pick_scatter_NP_and_PAINT_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        # plt.show()
        plt.close()

#%%
    if False:
        # plot SCATTER + NP
        plt.figure(2)
        plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'PAINT')
        if NP_flag:
            plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'C1',
                     markeredgecolor = 'k',alpha = 0.8, label = 'NP')
            plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('square')
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        aux_folder = manage_save_directory(figures_per_pick_folder,'scatter_plots')        
        figure_name = 'xy_pick_scatter_PAINT_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        # plt.show()
        plt.close()
#%%
    if True:
        # plot FINE 2d image
        plt.figure(3)
        plt.imshow(z_hist, interpolation='none', origin='lower',
                   extent=[x_hist_centers[0], x_hist_centers[-1], 
                           y_hist_centers[0], y_hist_centers[-1]])
        ax = plt.gca()
        ax.set_facecolor(bkg_color)
        plt.plot(cm_binding_sites_x[final_order], cm_binding_sites_y[final_order], 'x', markersize = 9, 
                 color = 'white', label = 'Photons per peak')
        plt.plot(x_fitted, y_fitted, '--', linewidth = 1, color = 'white')
        num_pick = 1
        # plt.plot(x_coord_check,y_coord_check,marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
        for circle_x, circle_y in zip(cm_binding_sites_x[final_order], cm_binding_sites_y[final_order]):
            circ = plot_circle((circle_x, circle_y), radius = analysis_radius, 
                        color = 'white', fill = False, label="{} | {:.4}".format(num_pick, np.sum(all_traces_per_pick[:,final_order[num_pick-1]])))
            print(final_order[num_pick-1])
            ax.add_patch(circ)
            plt.text(circle_x-analysis_radius, circle_y+(analysis_radius), "{}".format(num_pick),color="white")
            # plt.text(circle_x-analysis_radius, circle_y-1.5*(analysis_radius), "{:.4}".format(np.sum(all_traces_per_pick[:,final_order[num_pick-1]])),color="white")
            
            num_pick += 1
        xlimit = plt.gca().get_xlim()
        ylimit = plt.gca().get_ylim()
        plt.legend()
        plt.plot(x_line, y_line, ls="--", c=".3")
        plt.fill_between(x_line, y_line-tolerance, y_line+tolerance, alpha = 0.2)
        # x_spot, y_spot = ([x0,x0,x0*1.1,x0*1.1], [y0*0.99, y0*1.01, y_line[-1]*1.01,y_line[-1]*0.99])
        # plt.plot(x_spot, y_spot, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="green")
        plt.xlim(xlimit)
        plt.ylim(ylimit)
        # if NP_flag:
        #     plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 8, markerfacecolor = 'C1', 
        #              markeredgecolor = 'white', label = 'NP')
        #     plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        
        
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs', fontsize = 16)
        cbar.ax.tick_params(labelsize = 16)
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        aux_folder = manage_save_directory(figures_per_pick_folder,'image_FINE')        
        figure_name = 'xy_pick_image_PAINT_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
#%%
    if True:
        # plot FINE 2d image FOR PAPER
        plt.figure(3)
        plt.imshow(z_hist, interpolation='none', origin='lower',
                   extent=[x_hist_centers[0], x_hist_centers[-1], 
                           y_hist_centers[0], y_hist_centers[-1]])
        plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize = 9, 
                 color = 'black', mew = 2, label = 'binding sites')
        plt.plot(x_fitted, y_fitted, '--', linewidth = 1, color = 'wheat')
        ax = plt.gca()
        ax.set_facecolor(bkg_color)
        num_pick = 1
        for circle_x, circle_y in zip(cm_binding_sites_x, cm_binding_sites_y):
            circ = plot_circle((circle_x, circle_y), radius = analysis_radius, 
                        color = 'white', fill = False)
            ax.add_patch(circ)
            plt.text(circle_x-analysis_radius, circle_y+(analysis_radius), "{}".format(num_pick))
            num_pick += 1
        if NP_flag:
            plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 8, markerfacecolor = 'white', 
                     markeredgecolor = 'black', label = 'NP')
            plt.legend(loc='upper right')
        scalebar = ScaleBar(1e3, 'nm', location = 'lower left') 
        ax.add_artist(scalebar)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        aux_folder = manage_save_directory(figures_per_pick_folder,'image_FINE')        
        figure_name = 'PAPER_xy_pick_image_PAINT_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
 #%%       
    if True:
        # plot COARSE 2d image
        plt.figure(4)
        plt.imshow(z_hist_COARSE, interpolation='none', origin='lower',
                   extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                           y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
        if NP_flag:
            plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'white', 
                     markeredgecolor = 'k', label = 'NP')
            plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        ax = plt.gca()
        ax.set_facecolor(bkg_color)
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs')
        cbar.ax.tick_params()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        aux_folder = manage_save_directory(figures_per_pick_folder,'image_COARSE')
        figure_name = 'xy_pick_image_COARSE_PAINT_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        # plt.show()
        plt.close()
 #%%       
    if False:
        # plot BINARY 2d image
        plt.figure(5)
        plt.imshow(detected_peaks, interpolation='none', origin='lower', cmap = 'binary',
                   extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                           y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
        if NP_flag:
            plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'C1', 
                     markeredgecolor = 'k', alpha = 0.5, label = 'NP')
            plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        aux_folder = manage_save_directory(figures_per_pick_folder,'binary_image')
        figure_name = 'xy_pick_image_peaks_PAINT_%02d' % i
        figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        # plt.show()
        plt.close()
       #%%

    all_detected_peaks[str(i)] =  (cm_binding_sites_x[final_order],cm_binding_sites_y[final_order])

# proper_data = dict()
# proper_data["a"] = all_detected_peaks

last_part = dat_files[0].rfind("_")
nametosave = dat_files[0][:last_part] + "_from_{}_to_{}_photons{}.npz".format(i_min,i,photons_delete)

# nametosave = dat_files[0][:-5] + "from_{}_to_{}.npz".format(0,i)
with open(nametosave,"w") as f:
    np.savez(nametosave, **all_detected_peaks)
print("{} NPZ saved with the spots from {} to {} in \n".format(photons_delete,i_min,i), nametosave)

## plot relative positions of the binding sites with respect NP
number_of_bins2 = 16
hist_range = [25, 160]
bin_size = (hist_range[-1] - hist_range[0])/number_of_bins2
print('\nRelative position NP-sites bin size', bin_size)
position_bins, bin_edges = np.histogram(positions_concat_NP, bins = number_of_bins2, \
                                        range = hist_range)
bin_centers = bin_edges[:-1] + bin_size/2
plt.figure()
plt.bar(bin_centers, position_bins, width = 0.8*bin_size, align = 'center')
plt.xlabel('Position (nm)')
plt.ylabel('Counts')
figure_name = 'relative_positions_NP_sites'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

## plot relative positions between binding sites
number_of_bins2 = 16
hist_range = [25, 160]
bin_size = (hist_range[-1] - hist_range[0])/number_of_bins2
print('\nRelative position between binding sites bin size', bin_size)
position_bins, bin_edges = np.histogram(positions_concat_origami, bins = number_of_bins2, \
                                        range = hist_range)
bin_centers = bin_edges[:-1] + bin_size/2
plt.figure()
plt.bar(bin_centers, position_bins, width = 0.8*bin_size, align = 'center')
plt.xlabel('Position (nm)')
plt.ylabel('Counts')
figure_name = 'relative_positions_binding_sites'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

# plot global variables, all the picks of the video
time_concat = frame_concat*exp_time/60

## LOCS
sum_of_locs_of_picked_vs_time = np.sum(locs_of_picked_vs_time, axis=0)
plt.figure()
plt.step(bin_centers_minutes, sum_of_locs_of_picked_vs_time, where = 'mid')
plt.xlabel('Time (min)')
plt.ylabel('Locs')
x_limit = [0, total_time_min]
plt.xlim(x_limit)
ax = plt.gca()
ax.set_title('Sum of localizations vs time. Binning time %d s. %d picks. ' \
             % ((bin_size*0.1), total_number_of_picks))
figure_name = 'locs_vs_time_all'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

## PHOTONS
plt.figure()
plt.step(time_concat, photons_concat, '.')
plt.xlabel('Time (min)')
plt.ylabel('Photons')
x_limit = [0, total_time_min]
plt.xlim(x_limit)
ax = plt.gca()
ax.set_title('Photons vs time')
figure_name = 'photons_vs_time'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

## BACKGROUND
plt.figure()
plt.step(time_concat, bkg_concat, '.')
plt.xlabel('Time (min)')
plt.ylabel('Background')
x_limit = [0, total_time_min]
plt.xlim(x_limit)
ax = plt.gca()
ax.set_title('Background vs time')
figure_name = 'bkg_vs_time'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

################################### save data

# delete first fake and empty trace (needed to make the proper array)
all_traces = np.delete(all_traces, 0, axis = 0)
all_traces = all_traces.T
# save ALL traces in one file
if peaks_flag:
    new_filename = 'TRACES_ALL_%s.txt' % (frame_file[:-10])
    new_filepath = os.path.join(folder, new_filename)
    np.savetxt(new_filepath, all_traces_per_pick, fmt='%05d')
# compile all traces of the image in one array

# number of locs
data_to_save = np.asarray([pick_number, locs_of_picked]).T
new_filename = 'number_of_locs_per_pick.dat'
new_filepath = os.path.join(figures_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%i')

# relative positions
data_to_save = positions_concat_NP
new_filename = 'relative_positions_NP_sites_in_nm.dat'
new_filepath = os.path.join(figures_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%.1f')
data_to_save = positions_concat_origami
new_filename = 'relative_positions_binding_sites_in_nm.dat'
new_filepath = os.path.join(figures_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%.1f')

#%% Functions


def peak_finder (docking_sites, th):
    total_peaks_found = 10
    threshold_COARSE = th
    while total_peaks_found > 3 or total_peaks_found == 0:  # docking_sites
        # make COARSE 2D histogram of locs
        # number of bins is arbitrary, determined after trial and error
        bins_COARSE = 25
        x_min = min(x_position_of_picked)
        y_min = min(y_position_of_picked)
        x_max = x_min + pick_size*pixel_size
        y_max = y_min + pick_size*pixel_size
        z_hist_COARSE, x_hist_COARSE, y_hist_COARSE = np.histogram2d(x_position_of_picked, 
                                                                     y_position_of_picked, 
                                                                     bins = bins_COARSE,
                                                                     range = [[x_min, x_max], \
                                                                              [y_min, y_max]],
                                                                     density = True)
        z_hist_COARSE = z_hist_COARSE.T
        x_hist_step_COARSE = np.diff(x_hist_COARSE)
        y_hist_step_COARSE = np.diff(y_hist_COARSE)
        x_hist_COARSE_centers = x_hist_COARSE[:-1] + x_hist_step_COARSE/2
        y_hist_COARSE_centers = y_hist_COARSE[:-1] + y_hist_step_COARSE/2
        z_hist_COARSE = np.where(z_hist_COARSE < threshold_COARSE, 0, z_hist_COARSE)
        
        # peak detection for Center of Mass localization
        detected_peaks = detect_peaks(z_hist_COARSE)
        # find Center of Mass of locs near the peaks that were found
        index_peaks = np.where(detected_peaks == True) # this is a tuple
        total_peaks_found = len(index_peaks[0])
    #
        plt.figure("COARSE 2D")
        print('threshold_COARSE reached', threshold_COARSE, "peaks= ", total_peaks_found)
        ax = plt.gca()
        ax.set_facecolor(bkg_color)
        plt.imshow(z_hist_COARSE, interpolation='none', origin='lower',
                    extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                            y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
        # num_pick = 1
        for l in range(total_peaks_found):
            circ = plot_circle((x_hist_COARSE[index_peaks[1]][l], y_hist_COARSE[index_peaks[0]][l]), radius = 0.01, 
                        color = 'white', fill = False)
            ax.add_patch(circ)
            # plt.text(circle_x-analysis_radius, circle_y+(analysis_radius), "{}".format(num_pick),color="white")
            # num_pick += 1
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs')
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        plt.show()
    #
        threshold_COARSE += 5
        if threshold_COARSE > 5000:
            # this MAX value is arbitrary
            break
        if total_peaks_found == 0:
            threshold_COARSE -= 7
