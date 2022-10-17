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
from tkinter import Tk, filedialog

plt.ioff()
plt.close("all")    

pixel_size = 0.0650 # in um 0.130

##############################################################################
# pick one file of the folder
base_folder = 'C:\\datos_mariano\\posdoc\\MoS2\\DNA-PAINT_measurements'
base_folder = 'C:\\datos_mariano\\posdoc\\DNA-PAINT\\data_fribourg'
base_folder = 'C:/Origami testing Widefield/2022-04-20 Dimers 3 and 1 spot/well1_488nm_1mW_TIRF2470_zone1_PAINT_1'

root = Tk()
dat_file = filedialog.askopenfilename(initialdir = base_folder, 
                                      filetypes=(("", "*.dat"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(dat_file)

# figures folder
figures_folder = os.path.join(folder, 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)
# figures folder
figures_per_pick_folder = os.path.join(folder, 'figures\\per_pick')
if not os.path.exists(figures_per_pick_folder):
    os.makedirs(figures_per_pick_folder)


list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('.dat',f)]
list_of_files.sort()
L = len(list_of_files)

##############################################################################
# load data
frame_file = [f for f in list_of_files if re.search('_frame',f)][0]
frame_filepath = os.path.join(folder, frame_file)
frame = np.loadtxt(frame_filepath)

photons_file = [f for f in list_of_files if re.search('_photons',f)][0]
photons_filepath = os.path.join(folder, photons_file)
photons = np.loadtxt(photons_filepath)

bkg_file = [f for f in list_of_files if re.search('_bkg',f)][0]
bkg_filepath = os.path.join(folder, bkg_file)
bkg = np.loadtxt(bkg_filepath)

xy_file = [f for f in list_of_files if re.search('_xy',f)][0]
xy_filepath = os.path.join(folder, xy_file)
xy = np.loadtxt(xy_filepath)
x = xy[:,0]*pixel_size
y = xy[:,1]*pixel_size


pick_file = [f for f in list_of_files if re.search('_pick_number',f)][0]
pick_filepath = os.path.join(folder, pick_file)
pick_list = np.loadtxt(pick_filepath)

data_length = len(x)
##############################################################################

# time parameters
number_frames = 10000
exp_time = 0.1 # in s
total_time_sec = number_frames*exp_time # in sec
total_time_min = total_time_sec/60 # in min
print('Total time %.1f min' % total_time_min)


##############################################################################

# separate picks
pick_number = np.unique(pick_list)
total_number_of_picks = len(pick_number)
print('Total picks', total_number_of_picks)
locs_of_picked = np.zeros(total_number_of_picks)
number_of_bins = 60
locs_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
photons_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
bkg_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
photons_concat = np.array([])
bkg_concat = np.array([])
frame_concat = np.array([])

for i in range(total_number_of_picks):
    pick_id = pick_number[i]
    index_picked = np.where(pick_list == pick_id)
    frame_of_picked = frame[index_picked]
    photons_of_picked = photons[index_picked]
    bkg_of_picked = bkg[index_picked]
    
    photons_concat = np.concatenate([photons_concat, photons_of_picked])
    bkg_concat = np.concatenate([bkg_concat, bkg_of_picked])
    frame_concat = np.concatenate([frame_concat, frame_of_picked])
    
    locs_of_picked[i] = len(frame_of_picked)
    # print(locs_of_picked)
    
    hist_range = [0, number_frames]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    locs_of_picked_vs_time[i,:], bin_edges = np.histogram(frame_of_picked, bins = number_of_bins, range = hist_range)

    bin_centers = bin_edges[:-1] + bin_size/2
    bin_centers_minutes = bin_centers*exp_time/60
    if True:
        plt.figure()
#        plt.plot(bin_centers, locs_of_picked_vs_time[i], label = 'Pick %04d' % i)
        plt.step(bin_centers_minutes, locs_of_picked_vs_time[i,:], where = 'mid', label = 'Pick %04d' % i)
        plt.legend(loc='upper right')
        plt.xlabel('Time (min)')
        plt.ylabel('Locs')
#        plt.ylim([0, 80])
        ax = plt.gca()
        ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '2', linestyle = '--')
        ax.set_title('Number of locs per pick vs time. Bin size %.1f min' % (bin_size*0.1/60))
        figure_name = 'locs_per_pick_vs_time_pick_%04d' % i
        figure_path = os.path.join(figures_per_pick_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
        plt.close()


time_concat = frame_concat*exp_time/60

## LOCS
sum_of_locs_of_picked_vs_time = np.sum(locs_of_picked_vs_time, axis=0)
plt.figure()
plt.step(bin_centers_minutes, sum_of_locs_of_picked_vs_time, where = 'mid')
# plt.legend(loc='upper right')
plt.xlabel('Time (min)')
plt.ylabel('Locs')
x_limit = [0, 20]
y_limit = [0, 100]
plt.ylim(y_limit)
plt.xlim(x_limit)
ax = plt.gca()
# ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.axvline(x=11, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
ax.fill_between([10, 11], y_limit[0], y_limit[1], 
                facecolor='gray', edgecolor = 'None', alpha = 0.4, zorder=1)
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
x_limit = [0, 20]
y_limit = [0, 7000]
plt.ylim(y_limit)
plt.xlim(x_limit)
ax = plt.gca()
# ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.axvline(x=11, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
ax.fill_between([10, 11], y_limit[0], y_limit[1], 
                facecolor='gray', edgecolor = 'None', alpha = 0.4, zorder=1)
ax.set_title('Photons vs time')
figure_name = 'photons_vs_time'
figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.close()

## PHOTONS
plt.figure()
plt.step(time_concat, bkg_concat, '.')
# plt.legend(loc='upper right')
plt.xlabel('Time (min)')
plt.ylabel('Background')
x_limit = [0, 20]
y_limit = [0, 200]
plt.ylim(y_limit)
plt.xlim(x_limit)
ax = plt.gca()
# ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
# ax.axvline(x=11, ymin=0, ymax=1, color = 'k', linewidth = '1', linestyle = '--')
ax.fill_between([10, 11], y_limit[0], y_limit[1], 
                facecolor='gray', edgecolor = 'None', alpha = 0.4, zorder=1)
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

# # make histogram of locs per pick
# number_of_bins = 30
# hist_range = [0, number_frames//2]
# bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
# locs_of_picked_per_bin, bin_edges = np.histogram(locs_of_picked, bins = number_of_bins, range = hist_range)

# # plot
# bin_centers = bin_edges[:-1] + bin_size/2
# plt.figure()
# plt.bar(bin_centers, locs_of_picked_per_bin, width=1*bin_size)
# # plt.legend()
# plt.xlabel('Localizations per pick 20 min')
# plt.ylabel('Counts')
# ax = plt.gca()
# figure_name = 'histogram_of_locs_per_pick'
# figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
# plt.savefig(figure_path, dpi = 300, bbox_inches='tight')


# plt.close()
plt.show()

#%% Germán aditions.

plt.figure(1)
plt.hist2d(x, y, 
           bins=100, range=([10, 25], [12.5, 20]),
           cmin=0, cmax=1000)
plt.colorbar()
plt.show()



