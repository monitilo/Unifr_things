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

pixel_size = 0.065  #130 # in um

##############################################################################
# pick one file of the folder
base_folder = 'C:\\datos_mariano\\posdoc\\MoS2\\DNA-PAINT_measurements'
base_folder = 'C:\\datos_mariano\\posdoc\\DNA-PAINT\\data_fribourg'

root = Tk()
dat_file = filedialog.askopenfilename(initialdir = base_folder, 
                                      filetypes=(("", "*.dat"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(dat_file)

save_folder = folder
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
#%%
# time parameters
number_frames = 12000
exp_time = 0.1 # in s
total_time_sec = number_frames*exp_time # in sec
total_time_min = total_time_sec/60 # in min
print('Total time %.1f min' % total_time_min)


##############################################################################

# filter data spatially
def ellipse(x, y, xc, yc):
    radius_x = 5.3/2 # in um
    radius_y = 4.3/2 # in um
    f = ((x - xc)/radius_x)**2 + ((y - yc)/radius_y)**2
    return f

bines = 50
x_image_pixel = 400 # 152 # 76
y_image_pixel = np.copy(x_image_pixel)  # 152 # 76
x_lim = x_image_pixel*pixel_size
y_lim = y_image_pixel*pixel_size
xc = x_lim/2
yc = y_lim/2

# plot 2D localization map
plt.figure(0)
plt.hist2d(x, y, bins=bines, range=([0, x_lim], [0, y_lim]), 
                 cmin=0, cmax=4000)
plt.colorbar()
plt.show()
plt.close()

# allocate
# roi_frame = np.array([])
# roi_x = np.array([])
# roi_y = np.array([])
# roi_photons = np.array([])
# roi_bkg = np.array([])
# roi_pick = np.array([])
roi_ellipse = np.zeros(data_length) - 1
radii = [0, 1, 2, 3,4]
for i in range(data_length):
    x_loc = x[i]
    y_loc = y[i]
    for c in radii:
        if c**2 < ellipse(x_loc, y_loc, xc, yc) < (c + 1)**2:
            roi_ellipse[i] = c
            
index_zero = np.where(roi_ellipse == radii[0])
index_one = np.where(roi_ellipse == radii[1])
index_two = np.where(roi_ellipse == radii[2])
index_three = np.where(roi_ellipse == radii[3])

first_roi_x = x[index_zero]
first_roi_y = y[index_zero]

second_roi_x = x[index_one]
second_roi_y = y[index_one]

third_roi_x = x[index_two]
third_roi_y = y[index_two]

fourth_roi_x = x[index_three]
fourth_roi_y = y[index_three]

# plot 2D localization map
plt.figure(1)
plt.hist2d(first_roi_x, first_roi_y, 
           bins=bines, range=([0, x_lim], [0, y_lim]), 
           cmin=0, cmax=4000)
plt.colorbar()
plt.show()

# plot 2D localization map
plt.figure(2)
plt.hist2d(second_roi_x, second_roi_y, 
           bins=bines, range=([0, x_lim], [0, y_lim]), 
           cmin=0, cmax=4000)
plt.colorbar()
plt.show()

# plot 2D localization map
plt.figure(3)
plt.hist2d(third_roi_x, third_roi_y, 
           bins=bines, range=([0, x_lim], [0, y_lim]), 
           cmin=0, cmax=4000)
plt.colorbar()
plt.show()

# plot 2D localization map
plt.figure(4)
plt.hist2d(fourth_roi_x, fourth_roi_y, 
           bins=bines, range=([0, x_lim], [0, y_lim]), 
           cmin=0, cmax=4000)
plt.colorbar()
plt.show()

# make histogram of photons
number_of_bins = 100
hist_range = [0, 10000]
bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
photons_all_per_bin, bin_edges = np.histogram(photons, bins = number_of_bins, range = hist_range)
photons_1_per_bin, bin_edges = np.histogram(photons[index_zero], bins = number_of_bins, range = hist_range)
photons_2_per_bin, bin_edges = np.histogram(photons[index_one], bins = number_of_bins, range = hist_range)
photons_3_per_bin, bin_edges = np.histogram(photons[index_two], bins = number_of_bins, range = hist_range)
photons_4_per_bin, bin_edges = np.histogram(photons[index_three], bins = number_of_bins, range = hist_range)

# plot
bin_centers = bin_edges[:-1] + bin_size/2
plt.figure(10)
plt.bar(bin_centers, photons_all_per_bin, width=1*bin_size, alpha = 0.3)
plt.bar(bin_centers, photons_1_per_bin, width=1*bin_size, alpha = 0.3)
plt.bar(bin_centers, photons_2_per_bin, width=1*bin_size, alpha = 0.3)
plt.bar(bin_centers, photons_3_per_bin, width=1*bin_size, alpha = 0.3)
plt.bar(bin_centers, photons_4_per_bin, width=1*bin_size, alpha = 0.3)
# plt.legend()
plt.xlabel('Photons')
plt.ylabel('Counts')
ax = plt.gca()
figure_name = 'histogram_of_photons'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.show()

#%%

# data = {}
# data_name = {}
# for i in range(L):
#     filename = list_of_files[i]
#     filepath = os.path.join(folder, filename)
#     data[i] = np.loadtxt(filepath)
#     data_name[i] = filename
#     # print(max(data[i]))

pick_number = np.unique(pick_list)
locs_of_picked = np.zeros(len(pick_number))
for i in range(len(pick_number)):
    pick_id = pick_number[i]
    index_picked = np.where(pick_list == pick_id)
    frame_of_picked = frame[index_picked]
    locs_of_picked[i] = len(frame_of_picked)
    
    plt.figure()
    plt.plot(frame_of_picked)
    plt.legend()
    plt.xlabel('i')
    plt.ylabel('Frame')
    ax = plt.gca()
    ax.set_title('Number of locs per pick vs time. Bin size %d min' % bin_size)
    figure_name = 'histogram_of_photons'
    figure_path = os.path.join(save_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')

# save data
# number of locs
data_to_save = np.asarray([pick_number, locs_of_picked]).T
new_filename = 'number_of_locs_per_pick.dat'
new_filepath = os.path.join(save_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%i')

# make histogram of locs per pick
number_of_bins = 30
hist_range = [0, number_frames//2]
bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
locs_of_picked_per_bin, bin_edges = np.histogram(locs_of_picked, bins = number_of_bins, range = hist_range)

# plot
bin_centers = bin_edges[:-1] + bin_size/2
plt.figure()
plt.bar(bin_centers, locs_of_picked_per_bin, width=1*bin_size)
# plt.legend()
plt.xlabel('Localizations per pick 20 min')
plt.ylabel('Counts')
ax = plt.gca()
figure_name = 'histogram_of_locs_per_pick'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')


# plt.close()
plt.show()


