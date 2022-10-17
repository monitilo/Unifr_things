# -*- coding: utf-8 -*-
"""
Created on Tuesday December 7 2021

@author: Mariano Barella

This script analyzes already-processed Picasso data. It opens .dat files that 
were generated with "extract_and_save_data_from_hdf5_picasso_files.py".

When the program starts select ANY .dat file. This action will determine the 
working folder.

As input it uses:
    - main folder
    - number of frames
    - exposure time
    - pixel size of the original video
    
Outputs are:
    - plots per pick (scatter plot of locs, fine 2D histograms)
    - global figures (photons vs time, localizations vs time and background vs time)

Warning: the program is coded to follow filename convention of the script
"extract_and_save_data_from_hdf5_picasso_files.py".

"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
import re
import tkinter as tk
import tkinter.filedialog as fd
from auxiliary_functions import manage_save_directory
    
plt.ioff()
plt.close("all")    
cmap = plt.cm.get_cmap('viridis')
bkg_color = cmap(0)

##############################################################################
# INPUTS

# base folder to select data
base_folder = 'C:\\datos_mariano\\posdoc\\DNA-PAINT\\data_fribourg'
# camera pixel size
pixel_size = 0.130 # in um
# size of the pick used in picasso
pick_size = 5 # in camera pixels (put the same number used in Picasso)
resolution = 0.018 # in um as measured with DNA-PAINT in our setup on 20210705 lab meeting
# time parameters
number_of_frames = 12000
exp_time = 0.1 # in s
# parameters for smoothing x, y drift
window = 11
deg = 1
##############################################################################
# PROGRAM STARTS

total_time_sec = number_of_frames*exp_time # in sec
total_time_min = total_time_sec/60 # in min
print('Total time %.1f min' % total_time_min)

# select any file (will use the selected folder actually)
root = tk.Tk()
dat_files = fd.askopenfilenames(initialdir = base_folder, 
                                      filetypes=(("", "*.pkl"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(dat_files[0])

# create folder to save data
# global figures folder
figures_folder = manage_save_directory(folder, 'figures_global')
# figures per pick folder
figures_per_pick_folder = os.path.join(folder, 'figures_per_pick')

# list files
list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('.pkl',f)]
list_of_files.sort()

##############################################################################

# initialization of arrays
# photons_concat = np.array([])
# bkg_concat = np.array([])
# frame_concat = np.array([])
x_positions_concat = []
y_positions_concat = []
x_positions_smooth_concat = []
y_positions_smooth_concat = []
power_label_concat = []
pick_id_concat = []
mean_sqr_displacement_x_concat = []
mean_sqr_displacement_y_concat = []
mean_sqr_displacement_r_concat = []

# load data
for filename in list_of_files:
# for filename in dat_files:    
    filepath = os.path.join(folder, filename)
    print('\n Opening', filename)
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    
    # frame number, used for time estimation
    frame_file = loaded_dict['frame']
    frame_filepath = os.path.join(folder, frame_file)
    frame = np.loadtxt(frame_filepath)
    
    # photons
    photons_file = loaded_dict['photons']
    photons_filepath = os.path.join(folder, photons_file)
    photons = np.loadtxt(photons_filepath)
    
    # bkg
    bkg_file = loaded_dict['bkg']
    bkg_filepath = os.path.join(folder, bkg_file)
    bkg = np.loadtxt(bkg_filepath)
    
    # xy positions
    position_file = loaded_dict['positions']
    position_filepath = os.path.join(folder, position_file)
    position = np.loadtxt(position_filepath)
    x = position[:,0]*pixel_size
    y = position[:,1]*pixel_size
    data_length = len(x)
    
    # number of pick
    pick_file = loaded_dict['pick_number']
    pick_filepath = os.path.join(folder, pick_file)
    pick_list = np.loadtxt(pick_filepath)
        
    ##############################################################################
    
    # how many picks?
    pick_number = np.unique(pick_list)
    total_number_of_picks = len(pick_number)
    print('Total picks', total_number_of_picks)
    
    power_label = re.search('\\d\\dmW', filename)[0]
    
    # data assignment per pick
    for i in range(total_number_of_picks):
        pick_id = pick_number[i]
        # print('\n---------- Pick number %d' % i)
        index_picked = np.where(pick_list == pick_id)
        # for origami
        frame_of_picked = frame[index_picked]
        time_picked_sec = frame_of_picked*exp_time
        time_picked_min = time_picked_sec/60
        photons_of_picked = photons[index_picked]
        bkg_of_picked = bkg[index_picked]
        x_position_of_picked = x[index_picked]
        y_position_of_picked = y[index_picked]
        r_position_of_picked = (x_position_of_picked**2 + y_position_of_picked**2)**0.5
        # make relative
        x_position_of_picked_relative = x_position_of_picked - x_position_of_picked[0]
        y_position_of_picked_relative = y_position_of_picked - y_position_of_picked[0]
        r_position_of_picked_relative = r_position_of_picked - r_position_of_picked[0]      
        x_position_of_picked_relative_smooth = sig.savgol_filter(x_position_of_picked_relative, 
                                          window, deg, axis = 0, mode='interp')
        y_position_of_picked_relative_smooth = sig.savgol_filter(y_position_of_picked_relative, 
                                          window, deg, axis = 0, mode='interp')
        mean_sqr_displacement_x = (1/number_of_frames)*np.sum( np.abs(x_position_of_picked_relative)**2 )
        mean_sqr_displacement_y = (1/number_of_frames)*np.sum( np.abs(y_position_of_picked_relative)**2 ) 
        mean_sqr_displacement_r = (1/number_of_frames)*np.sum( np.abs(r_position_of_picked_relative)**2 )
        
        x_positions_concat.append(x_position_of_picked_relative)
        y_positions_concat.append(y_position_of_picked_relative)
        x_positions_smooth_concat.append(x_position_of_picked_relative_smooth)
        y_positions_smooth_concat.append(y_position_of_picked_relative_smooth)
        power_label_concat.append(power_label)
        pick_id_concat.append(pick_id)
        mean_sqr_displacement_x_concat.append(mean_sqr_displacement_x)
        mean_sqr_displacement_y_concat.append(mean_sqr_displacement_y)
        mean_sqr_displacement_r_concat.append(mean_sqr_displacement_r)
        
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
    
        x_avg = np.mean(x_position_of_picked)
        y_avg = np.mean(y_position_of_picked)
        x_std_dev = np.std(x_position_of_picked, ddof = 1)
        y_std_dev = np.std(y_position_of_picked, ddof = 1)
        # # print them in nm
        # print('\nCM NP in nm (x y):', x_avg*1e3, y_avg*1e3)
        # print('std dev CM NP in nm (x y):', x_std_dev*1e3, y_std_dev*1e3)
            
        # plot individual xy coord of the pick in several ways
        y_lim = [-pick_size*pixel_size/2, pick_size*pixel_size/2]
#%%
        if True:
            # plot xy vs time
            plt.figure(0)
            plt.plot(time_picked_min, x_position_of_picked_relative_smooth, label = power_label)
            ax = plt.gca()
            plt.legend(loc='upper right')
#            plt.ylim(y_lim)
            plt.ylabel('x ($\mu$m)')
            plt.xlabel('Time (min)')
            ax.set_title('x position of locs. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'x_vs_time')        
            figure_name = '%s_x_vs_time_pick_%02d' % (filename[:-9], i)
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.close()
            plt.show()

            plt.figure(1)
            plt.plot(time_picked_min, y_position_of_picked_relative_smooth, label = power_label)
            ax = plt.gca()
            plt.legend(loc='upper right')
#            plt.ylim(y_lim)
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('Time (min)')
            ax.set_title('y position of locs. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'y_vs_time')        
            figure_name = '%sy_vs_time_pick_%02d' % (filename[:-9], i)
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.close()
            plt.show()

            # plot all RAW
            plt.figure(2)
            plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'Locs')
            plt.plot(x_avg, y_avg, 'x', color = 'k', label = 'Avg position')
            plt.legend(loc='upper right')
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            plt.axis('square')
            ax = plt.gca()
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'scatter_plots')        
            figure_name = '%s_xy_scatter_pick_%02d' % (filename[:-9], i)
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.close()
            plt.show()
                   
            # plot FINE 2d image
            plt.figure(3)
            plt.imshow(z_hist, interpolation='none', origin='lower')#,
#                       extent=[x_hist_centers[0], x_hist_centers[-1], 
#                               y_hist_centers[0], y_hist_centers[-1]])
            ax = plt.gca()
            ax.set_facecolor(bkg_color)
            plt.plot(x_avg, y_avg, 'o', markersize = 8, markerfacecolor = 'C1', 
                         markeredgecolor = 'white', label = 'Avg position')
            plt.legend()
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            cbar = plt.colorbar()
            cbar.ax.set_title(u'Locs', fontsize = 16)
            cbar.ax.tick_params(labelsize = 16)
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'image_FINE')
            figure_name = '%s_xy_image_pick_%02d' % (filename[:-9], i)
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.close()
            plt.show()

#%%
            
# plot for different powers
unique_power = np.unique(power_label_concat)
# plot xy vs time
for power in unique_power:
    index_power = np.where(np.array(power_label_concat) == power)[0]
    plt.figure(10)
    for i in index_power:
        time_min = np.array(range(len(x_positions_smooth_concat[i])))*exp_time/60
        plt.plot(time_min, x_positions_smooth_concat[i], label = 'Pick %i' % pick_id_concat[i])
    ax = plt.gca()
    ax.annotate(power, xy=(340, 250), xycoords='axes points',
            size=13, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))    
    plt.ylabel('x ($\mu$m)')
    plt.xlabel('Time (min)')
    plt.ylim(y_lim)
    figure_name = 'x_vs_time_%s' % power
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    plt.figure(11)
    for i in index_power:
        time_min = np.array(range(len(x_positions_smooth_concat[i])))*exp_time/60
        plt.plot(time_min, y_positions_smooth_concat[i], label = 'Pick %i' % pick_id_concat[i])
    ax = plt.gca()
    ax.annotate(power, xy=(340, 250), xycoords='axes points',
            size=13, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))   
    plt.ylabel('y ($\mu$m)')
    plt.xlabel('Time (min)')
    plt.ylim(y_lim)
    figure_name = 'y_vs_time_%s' % power
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
# plot MSD vs power
power_int_concat = [int(p[:-2]) for p in power_label_concat]
# WARNING: replace 78 mW by 75 mW with the purpose of comparison in same plot
power_int_concat = [75 if p == 78 else p for p in power_int_concat]

y_lim_msd = [5e-5, 1.2e-1]

plt.figure(12)
plt.plot(power_int_concat, mean_sqr_displacement_x_concat, 'o')
ax = plt.gca()
plt.ylabel('MSD in x ($\mu$m$^{2}$)')
plt.xlabel('Power BFP (mW)')
plt.ylim(y_lim_msd)
plt.xlim([-10, 160])
ax.set_yscale('log')
ax.set_title('At 12 °C')
figure_name = 'MSD_x_vs_power'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

plt.figure(13)
plt.plot(power_int_concat, mean_sqr_displacement_y_concat, 'o')
ax = plt.gca()
plt.ylabel('MSD in y ($\mu$m$^{2}$)')
plt.xlabel('Power BFP (mW)')
plt.ylim(y_lim_msd)
plt.xlim([-10, 160])
ax.set_yscale('log')
ax.set_title('At 12 °C')
figure_name = 'MSD_y_vs_power'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

plt.figure(14)
plt.plot(power_int_concat, mean_sqr_displacement_r_concat, 'o')
ax = plt.gca()
plt.ylabel('MSD in r ($\mu$m$^{2}$)')
plt.xlabel('Power BFP (mW)')
plt.ylim(y_lim_msd)
plt.xlim([-10, 160])
ax.set_yscale('log')
ax.set_title('At 12 °C')
figure_name = 'MSD_r_vs_power'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

unique_power_int = np.unique(power_int_concat)
unique_power_str = [str(p) for p in unique_power_int]
msd_r_dataset = []
for p in unique_power_int:
    index_power = np.where(power_int_concat == p)[0]
    msd_r_single_power = np.array(mean_sqr_displacement_r_concat)[index_power]
    msd_r_dataset.append(msd_r_single_power)

number_of_powers_tested = len(unique_power_int)
iterator = range(number_of_powers_tested)

fig, axs = plt.subplots(number_of_powers_tested, 1, gridspec_kw = {'wspace':0, 'hspace':0})
num_of_bins = 50
range_of_hist = [1e-4,1e-1]
range_of_hist = y_lim_msd
bin_size = (range_of_hist[1] - range_of_hist[0])/num_of_bins
print('Bin size = %.2f' % bin_size)
for i in iterator:
    ax = axs
    out_hist = ax.hist(msd_r_dataset[i], bins = num_of_bins, range = range_of_hist, \
                       rwidth = 1, label = '%s mW' % unique_power_str[i], align='mid', 
                       color='C0', alpha = 1, edgecolor='k', density = True, zorder=2)
    leg = ax.legend(loc = 'upper right', handlelength=0, handletextpad=0, \
                    fancybox=True, prop = {'size':13})
    plt.show()
    for item in leg.legendHandles:
        item.set_visible(False)
    ax.axvline(pixel_size**2, ymin = 0, ymax = 1, color='k', linestyle='--', linewidth=2, zorder=3)
    ax.axvline(resolution**2, ymin = 0, ymax = 1, color='C3', linestyle=':', linewidth=2, zorder=3)
    ax.grid('on', linestyle=':')
    ax.set_axisbelow(True)
    ax.get_yaxis().set_ticks([])
    if not i == number_of_powers_tested - 1:
        ax.get_xaxis().set_ticklabels([])
    ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
    ax.set_ylim([0, 1.2*np.max(out_hist[0])])
    ax.set_xlim(range_of_hist)

ax.set_xlabel('MSD in r ($\mu$m$^{2}$)', fontsize = 20)
# ax.set_title('At 12 °C')
figure_name = 'MSD_r_vs_power_histograms_linear.png'
figure_name = os.path.join(figures_folder, figure_name)
plt.savefig(figure_name, dpi=300, bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(number_of_powers_tested, 1, gridspec_kw = {'wspace':0, 'hspace':0})
num_of_bins = 12
for i in iterator:
    ax = axs[i]
    height, bins = np.histogram(msd_r_dataset[i], bins = num_of_bins, \
                            range = range_of_hist, density = True)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    out_hist = ax.hist(msd_r_dataset[i], bins = logbins, \
                       label = '%s mW' % unique_power_str[i], 
                       color='C0', alpha = 1, edgecolor='k', zorder=2)
    leg = ax.legend(loc = 'upper right', handlelength=0, handletextpad=0, \
                    fancybox=True, prop = {'size':13})
    for item in leg.legendHandles:
        item.set_visible(False)
    ax.axvline(pixel_size**2, ymin = 0, ymax = 1, color='k', linestyle='--', linewidth=2, zorder=3)
    ax.axvline(resolution**2, ymin = 0, ymax = 1, color='C3', linestyle=':', linewidth=2, zorder=3)
    ax.grid('on', linestyle=':')
    ax.set_axisbelow(True)
    ax.get_yaxis().set_ticks([])
    ax.set_xscale('log')
    if not i == number_of_powers_tested - 1:
        ax.get_xaxis().set_ticklabels([])
    ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
    # ax.set_ylim([0, 1.2*np.max(height)])
    ax.set_xlim(range_of_hist)

ax.set_xlabel('MSD in r ($\mu$m$^{2}$)', fontsize = 20)
# ax.set_title('At 12 °C')
figure_name = 'MSD_r_vs_power_histograms_log.png'
figure_name = os.path.join(figures_folder, figure_name)
plt.savefig(figure_name, dpi=300, bbox_inches='tight')
plt.close()


# plt.figure(16)
# plt.violinplot(msd_r_dataset, showmeans=False, showmedians=True)
# ax = plt.gca()
# ax.set_xticks(list(range(1,len(unique_power_int)+1)))
# ax.set_xticklabels(unique_power_str)
# plt.ylabel('MSD in r ($\mu$m$^{2}$)')
# plt.xlabel('Power BFP (mW)')
# plt.ylim([1e-4, 0.12])
# ax.set_yscale('log')
# ax.set_title('At 12 °C')
# figure_name = 'MSD_r_vs_power_violinplot'
# figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
# plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
# plt.close()

    
print('\nDone.')