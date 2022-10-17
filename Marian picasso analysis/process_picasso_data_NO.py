# -*- coding: utf-8 -*-
"""
Created on Tuesday Novemeber 16 2021

@author: Mariano Barella

This script processes the output of Picasso software. 
It opens .dat files that are generated with 
"extract_and_save_data_from_hdf5_picasso_files.py".

As input it uses the number of frames and the exposure time.

"""
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Circle as plot_circle
import os
import re
import tkinter as tk
import tkinter.filedialog as fd
from auxiliary_functions import detect_peaks, distance, \
                                                fit_linear, perpendicular_distance
    
plt.ioff()
plt.close("all")    

##############################################################################
# INPUTS

# camera pixel size
pixel_size = 0.0650 # in um
# size of the pick used in picasso
pick_size = 7 # in camera pixels (put the same number used in picasso)
# size of the pick to include locs around the detected peaks
radius_of_pick_to_average = 0.5 # in camera pixel size
# time parameters
number_of_frames = 10000
exp_time = 0.1 # in s
total_time_sec = number_of_frames*exp_time # in sec
total_time_min = total_time_sec/60 # in min
print('Total time %.1f min' % total_time_min)

# folders to look data
# base_folder = 'C:\\datos_mariano\\posdoc\\MoS2\\DNA-PAINT_measurements'
base_folder = 'C:\\datos_mariano\\posdoc\\DNA-PAINT\\data_fribourg\\distance_between_NP_and_origami_picks'


##############################################################################
# PROGRAM STARTS

# select any file (will use the selected folder actually)
root = tk.Tk()
dat_files = fd.askopenfilenames(initialdir = base_folder, 
                                      filetypes=(("", "*.dat"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(dat_files[0])

# create folder to save data
# figures folder
figures_folder = os.path.join(folder, 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)
# figures per pick folder
figures_per_pick_folder = os.path.join(figures_folder, 'per_pick')
if not os.path.exists(figures_per_pick_folder):
    os.makedirs(figures_per_pick_folder)
# traces per pick folder
traces_per_pick_folder = os.path.join(folder, 'traces_per_pick')
if not os.path.exists(traces_per_pick_folder):
    os.makedirs(traces_per_pick_folder)

# list files
list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('.dat',f)]
list_of_files.sort()
list_of_files_origami = [f for f in list_of_files if re.search('merged',f)]
list_of_files_NP = [f for f in list_of_files if re.search('MMStack_Pos0.ome',f)]

##############################################################################
# load data

# frame number, used for time estimation
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

# data assignment per pick
for i in range(total_number_of_picks):
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
    z_hist, x_hist, y_hist = np.histogram2d(x_position_of_picked, 
                                            y_position_of_picked, 
                                            bins = N)
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    z_hist = z_hist.T
    x_hist_step = np.diff(x_hist)
    y_hist_step = np.diff(y_hist)
    x_hist_centers = x_hist[:-1] + x_hist_step/2
    y_hist_centers = y_hist[:-1] + y_hist_step/2
    # make COARSE 2D histogram of locs
    # number of bins is arbitrary, determined after trial and error
    z_hist_COARSE, x_hist_COARSE, y_hist_COARSE = np.histogram2d(x_position_of_picked, 
                                                                 y_position_of_picked, 
                                                                 bins = 20,
                                                                 density = True)
    z_hist_COARSE = z_hist_COARSE.T
    x_hist_step_COARSE = np.diff(x_hist_COARSE)
    y_hist_step_COARSE = np.diff(y_hist_COARSE)
    x_hist_COARSE_centers = x_hist_COARSE[:-1] + x_hist_step_COARSE/2
    y_hist_COARSE_centers = y_hist_COARSE[:-1] + y_hist_step_COARSE/2
    # set an intensity threshold to avoid dumb peak detection in the background
    # this threshold is arbitrary, determined after trial and error
    threshold_COARSE = 30
    z_hist_COARSE = np.where(z_hist_COARSE < threshold_COARSE, 0, z_hist_COARSE)
    
    # peak detection for Center of Mass localization
    detected_peaks = detect_peaks(z_hist_COARSE)
    # find Center of Mass of locs near the peaks that were found
    index_peaks = np.where(detected_peaks == True) # this is a tuple
    total_peaks_found = len(index_peaks[0])
    analysis_radius = radius_of_pick_to_average*pixel_size
    cm_binding_sites_x = np.array([])
    cm_binding_sites_y = np.array([])
    cm_std_dev_binding_sites_x = np.array([])
    cm_std_dev_binding_sites_y = np.array([])
    # array where traces are going to be saved
    all_traces = np.zeros(number_of_frames)
    for j in range(total_peaks_found):
        print('Binding site %d of %d' % (j+1, total_peaks_found))
        index_x_peak = index_peaks[1][j] # first element of the tuple are rows
        index_y_peak = index_peaks[0][j] # second element of the tuple are columns
        x_peak = x_hist_COARSE_centers[index_x_peak]
        y_peak = y_hist_COARSE_centers[index_y_peak]
        # grab all locs inside the selected circle 
        # circle = selected radius around the (x,y) of the detected peak
        # 1) calculate distance of all locs with respecto to the (x,y) of the peak
        d = distance(x_position_of_picked, y_position_of_picked, x_peak, y_peak)
        # 2) filter by the radius
        index_inside_radius = np.where(d < analysis_radius)
        x_position_of_picked_filtered = x_position_of_picked[index_inside_radius]
        y_position_of_picked_filtered = y_position_of_picked[index_inside_radius]
        # 3) calculate average position of the binding site
        cm_binding_site_x = np.mean(x_position_of_picked_filtered)
        cm_binding_site_y = np.mean(y_position_of_picked_filtered)
        cm_std_dev_binding_site_x = np.std(x_position_of_picked_filtered, ddof = 1)
        cm_std_dev_binding_site_y = np.std(y_position_of_picked_filtered, ddof = 1)
        print('CM binding in nm (x y):', cm_binding_site_x*1e3, \
              cm_binding_site_y*1e3)
        print('std dev CM binding in nm (x y):', cm_std_dev_binding_site_x*1e3, \
              cm_std_dev_binding_site_y*1e3)
        # 4) save the averaged position in a new array
        cm_binding_sites_x = np.append(cm_binding_sites_x, cm_binding_site_x)
        cm_binding_sites_y = np.append(cm_binding_sites_y, cm_binding_site_y)
        cm_std_dev_binding_sites_x = np.append(cm_std_dev_binding_sites_x, cm_std_dev_binding_site_x)
        cm_std_dev_binding_sites_y = np.append(cm_std_dev_binding_sites_y, cm_std_dev_binding_site_y)
        # 5) export the trace of the binding site
        frame_of_picked_filtered = np.array(frame_of_picked[index_inside_radius], dtype=int)
        photons_of_picked_filtered = photons_of_picked[index_inside_radius]
        empty_trace = np.zeros(number_of_frames)
        empty_trace[frame_of_picked_filtered] = photons_of_picked_filtered
        trace = empty_trace
        # compile all traces in one array
        all_traces = np.vstack([all_traces, trace])
    # delete first fake and empty trace (needed to make the proper array)
    all_traces = np.delete(all_traces, 0, axis = 0)
    all_traces = all_traces.T
    # save traces per pick
    new_filename = 'TRACES_pick_%02d_%s.txt' % (i, frame_file[:-10])
    new_filepath = os.path.join(traces_per_pick_folder, new_filename)
    np.savetxt(new_filepath, all_traces, fmt='%05d')
    
    # get NP coords in um
    index_picked_NP = np.where(pick_list_NP == pick_id)
    x_position_of_picked_NP = x_NP[index_picked_NP]
    y_position_of_picked_NP = y_NP[index_picked_NP]
    x_avg_NP = np.mean(x_position_of_picked_NP)
    y_avg_NP = np.mean(y_position_of_picked_NP)
    x_std_dev_NP = np.std(x_position_of_picked_NP, ddof = 1)
    y_std_dev_NP = np.std(y_position_of_picked_NP, ddof = 1)
    # print them in nm
    print('\nCM NP in nm (x y):', x_avg_NP*1e3, y_avg_NP*1e3)
    print('std dev CM NP in nm (x y):', x_std_dev_NP*1e3, y_std_dev_NP*1e3)
    
    # fit linear (origami direction) of the binding sites 
    # to find the perpendicular distance to the NP
    x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(cm_binding_sites_x, 
                                                                cm_binding_sites_y)
    # distance between NP and the line fitted by the three binding sites
    distance_to_NP = perpendicular_distance(slope, intercept, 
                                            x_avg_NP, y_avg_NP)
    distance_to_NP_nm = distance_to_NP*1e3
    print('Perpendicular distance to NP: %.1f nm' % distance_to_NP_nm) 
    
    # calculate relative distances between all points
    # ------------------ in nanometers -----------------------
    # allocate: total size = number of detected peaks + 1 for NP
    matrix_distance = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
    matrix_std_dev = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
    # calcualte first row of the matrix distance
    for j in range(total_peaks_found):
        x_binding = cm_binding_sites_x[j]
        y_binding = cm_binding_sites_y[j]
        distance_between_locs_CM = distance(x_binding, y_binding, x_avg_NP, y_avg_NP)*1e3
        matrix_distance[0, j + 1] = distance_between_locs_CM
        matrix_distance[j + 1, 0] = distance_between_locs_CM
    matrix_std_dev[0, 0] = max(x_std_dev_NP, y_std_dev_NP)*1e3
    # calcualte the rest of the rows of the matrix distance
    for j in range(total_peaks_found):
        x_binding_row = cm_binding_sites_x[j]
        y_binding_row = cm_binding_sites_y[j]
        matrix_std_dev[j + 1, j + 1] = max(cm_std_dev_binding_sites_x[j], \
                                           cm_std_dev_binding_sites_y[j])*1e3
        for k in range(j + 1, total_peaks_found):
            x_binding_col = cm_binding_sites_x[k]
            y_binding_col = cm_binding_sites_y[k]
            distance_between_locs_CM = distance(x_binding_col, y_binding_col, \
                                              x_binding_row, y_binding_row)*1e3
            matrix_distance[j + 1, k + 1] = distance_between_locs_CM
            matrix_distance[k + 1, j + 1] = distance_between_locs_CM
    # designed distances
    # asymmetric origami (fourth origami): 
    #   - 136 nm between ends
    #   - 83 nm between center and far end
    #   - 54 nm between center and closer end
    # symmetric origami (third and fifth origami): 
    #   - 125 nm between ends
    #   - 54 nm between centers and far end
    #   - 18 nm between the two center peaks
    # print(matrix_distance)
    # print(matrix_std_dev)
    
    # plot matrix distance    
    plt.figure(10)
    plt.imshow(matrix_distance, interpolation='none', cmap='spring')
    ax = plt.gca()
    for l in range(matrix_distance.shape[0]):
        for m in range(matrix_distance.shape[1]):
            if l == m:
                ax.text(m, l, '-' ,
                    ha="center", va="center", color=[0,0,0], 
                    fontsize = 18)
            else:
                ax.text(m, l, '%.0f' % matrix_distance[l, m],
                    ha="center", va="center", color=[0,0,0], 
                    fontsize = 18)
    ax.xaxis.tick_top()
    ax.set_xticks(np.array(range(matrix_distance.shape[1])))
    ax.set_yticks(np.array(range(matrix_distance.shape[0])))
    axis_string = ['NP']
    for j in range(total_peaks_found):
        axis_string.append('Site %d' % (j+1))
    ax.set_xticklabels(axis_string)
    ax.set_yticklabels(axis_string)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # cbar = plt.colorbar()
    # cbar.ax.set_title(u'T$_{0}$ (K)', fontsize=13)
    figure_name = 'matrix_distance_%02d' % i
    figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
    plt.close()
    
    # plot matrix of max std dev    
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
    axis_string = ['NP']
    for j in range(total_peaks_found):
        axis_string.append('Site %d' % (j+1))
    ax.set_xticklabels(axis_string)
    ax.set_yticklabels(axis_string)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # cbar = plt.colorbar()
    # cbar.ax.set_title(u'T$_{0}$ (K)', fontsize=13)
    figure_name = 'matrix_std_dev_%02d' % i
    figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
    plt.close()
    
    # plots of the binding sites
    photons_concat = np.concatenate([photons_concat, photons_of_picked])
    bkg_concat = np.concatenate([bkg_concat, bkg_of_picked])
    frame_concat = np.concatenate([frame_concat, frame_of_picked])
    locs_of_picked[i] = len(frame_of_picked)
    # print(locs_of_picked)
    hist_range = [0, number_of_frames]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    locs_of_picked_vs_time[i,:], bin_edges = np.histogram(frame_of_picked, bins = number_of_bins, range = hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    bin_centers_minutes = bin_centers*exp_time/60  
    # plot when the pick was bright vs time
    if False:
        plt.figure()
        # plt.plot(bin_centers, locs_of_picked_vs_time, label = 'Pick %04d' % i)
        plt.step(bin_centers_minutes, locs_of_picked_vs_time[i,:], where = 'mid', label = 'Pick %04d' % i)
        plt.legend(loc='upper right')
        plt.xlabel('Time (min)')
        plt.ylabel('Locs')
        plt.ylim([0, 80])
        ax = plt.gca()
        ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '2', linestyle = '--')
        ax.set_title('Number of locs per pick vs time. Bin size %.1f min' % (bin_size*0.1/60))
        figure_name = 'locs_per_pick_vs_time_pick_%02d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
        plt.close()
        
    # plot xy coord of the pick in several ways, including the peaks detected
    if True:
        # plot all RAW
        plt.figure(1)
        plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'PAINT')
        plt.plot(x_position_of_picked_NP, y_position_of_picked_NP, '.', color = 'C1', label = 'NP')
        plt.plot(x_avg_NP, y_avg_NP, 'x', color = 'k', label = 'Avg position NP')
        plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('scaled')
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        figure_name = 'xy_pick_scatter_NP_and_PAINT_%02d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.close()
        
    if True:
        # plot SCATTER + NP
        plt.figure(2)
        plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'PAINT')
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'C1', 
                 markeredgecolor = 'k', label = 'NP')
        plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('scaled')
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        figure_name = 'xy_pick_scatter_PAINT_%02d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.close()
        
    if True:
        # plot FINE 2d image
        plt.figure(3)
        plt.imshow(z_hist, interpolation='none', origin='lower',
                   extent=[x_hist_centers[0], x_hist_centers[-1], 
                           y_hist_centers[0], y_hist_centers[-1]])
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 8, markerfacecolor = 'C1', 
                 markeredgecolor = 'white', label = 'NP')
        plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize = 9, 
                 color = 'white', label = 'binding sites')
        plt.plot(x_fitted, y_fitted, '--', linewidth = 1, color = 'white')
        ax = plt.gca()
        for circle_x, circle_y in zip(cm_binding_sites_x, cm_binding_sites_y):
            circ = plot_circle((circle_x, circle_y), radius = analysis_radius, 
                        color = 'white', fill = False)
            ax.add_patch(circ)
        plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('scaled')
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs', fontsize = 16)
        cbar.ax.tick_params(labelsize = 16)
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        figure_name = 'xy_pick_image_PAINT_%02d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.close()
        
    if True:
        # plot FINE 2d image FOR PAPER
        plt.figure(3)
        plt.imshow(z_hist, interpolation='none', origin='lower',
                   extent=[x_hist_centers[0], x_hist_centers[-1], 
                           y_hist_centers[0], y_hist_centers[-1]])
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 8, markerfacecolor = 'white', 
                 markeredgecolor = 'black', label = 'NP')
        plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize = 9, 
                 color = 'black', mew = 2, label = 'binding sites')
        plt.plot(x_fitted, y_fitted, '--', linewidth = 1, color = 'wheat')
        ax = plt.gca()
        for circle_x, circle_y in zip(cm_binding_sites_x, cm_binding_sites_y):
            circ = plot_circle((circle_x, circle_y), radius = analysis_radius, 
                        color = 'white', fill = False)
            ax.add_patch(circ)
        plt.legend(loc='upper right')
        # plt.ylabel('y ($\mu$m)')
        # plt.xlabel('x ($\mu$m)')
        plt.axis('scaled')
        # cbar = plt.colorbar()
        # cbar.ax.set_title(u'Locs', fontsize = 16)
        # cbar.ax.tick_params(labelsize = 16)
        scalebar = ScaleBar(1e3, 'nm', location = 'lower left') 
        ax.add_artist(scalebar)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax.set_title('Position of locs per pick. Pick %02d' % i)
        figure_name = 'xy_pick_image_PAINT_%02d_for_PAPER' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.close()
        
    if True:
        # plot COARSE 2d image
        plt.figure(4)
        plt.imshow(z_hist_COARSE, interpolation='none', origin='lower',
                   extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                           y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'white', 
                 markeredgecolor = 'k', label = 'NP')
        plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('scaled')
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs')
        cbar.ax.tick_params()
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        figure_name = 'xy_pick_image_COARSE_PAINT_%02d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.close()
        
    if True:
        # plot BINARY 2d image
        plt.figure(5)
        plt.imshow(detected_peaks, interpolation='none', origin='lower', cmap = 'binary',
                   extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                           y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'C1', 
                 markeredgecolor = 'k', label = 'NP')
        plt.legend(loc='upper right')
        plt.ylabel('y ($\mu$m)')
        plt.xlabel('x ($\mu$m)')
        plt.axis('scaled')
        ax = plt.gca()
        ax.set_title('Position of locs per pick. Pick %02d' % i)
        figure_name = 'xy_pick_image_peaks_PAINT_%02d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.close()
        
# plot global variables, all the picks of the video
time_concat = frame_concat*exp_time/60

## LOCS
sum_of_locs_of_picked_vs_time = np.sum(locs_of_picked_vs_time, axis=0)
plt.figure()
plt.step(bin_centers_minutes, sum_of_locs_of_picked_vs_time, where = 'mid')
# plt.legend(loc='upper right')
plt.xlabel('Time (min)')
plt.ylabel('Locs')
x_limit = [0, total_time_min]
y_limit = [0, 100]
# plt.ylim(y_limit)
plt.xlim(x_limit)
ax = plt.gca()
# ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.axvline(x=11, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.fill_between([10, 11], y_limit[0], y_limit[1], 
#                 facecolor='gray', edgecolor = 'None', alpha = 0.4, zorder=1)
ax.set_title('Sum of localizations vs time. Binning time %d s. %d picks. ' \
             % ((bin_size*0.1), total_number_of_picks))
figure_name = 'locs_vs_time_all'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

## PHOTONS
plt.figure()
plt.step(time_concat, photons_concat, '.')
# plt.legend(loc='upper right')
plt.xlabel('Time (min)')
plt.ylabel('Photons')
x_limit = [0, total_time_min]
y_limit = [0, 7000]
# plt.ylim(y_limit)
plt.xlim(x_limit)
ax = plt.gca()
# ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.axvline(x=11, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.fill_between([10, 11], y_limit[0], y_limit[1], 
#                 facecolor='gray', edgecolor = 'None', alpha = 0.4, zorder=1)
ax.set_title('Photons vs time')
figure_name = 'photons_vs_time'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

## BACKGROUND
plt.figure()
plt.step(time_concat, bkg_concat, '.')
# plt.legend(loc='upper right')
plt.xlabel('Time (min)')
plt.ylabel('Background')
x_limit = [0, total_time_min]
y_limit = [0, 200]
# plt.ylim(y_limit)
plt.xlim(x_limit)
ax = plt.gca()
# ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.axvline(x=11, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.fill_between([10, 11], y_limit[0], y_limit[1], 
#                 facecolor='gray', edgecolor = 'None', alpha = 0.4, zorder=1)
ax.set_title('Background vs time')
figure_name = 'bkg_vs_time'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

# save data
# number of locs
data_to_save = np.asarray([pick_number, locs_of_picked]).T
new_filename = 'number_of_locs_per_pick.dat'
new_filepath = os.path.join(figures_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%i')

# plt.close()
plt.show()