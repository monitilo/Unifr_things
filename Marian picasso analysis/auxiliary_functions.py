"""
Created on Tuesday Novemeber 17 2021

@author: Mariano Barella

This script contains the auxiliary functions that process_picasso_data
main script uses.

"""

import os
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# 2D peak detection algorithm
# taken from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value 
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    # local_max is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image==0)

    # a little technicality: we must erode the background in order to 
    # successfully subtract it form local_max, otherwise a line will 
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

# distance calculation circle
def distance(x, y, xc, yc):
    d = ((x - xc)**2 + (y - yc)**2)**0.5
    return d

# Calculate coefficient of determination
def calc_r2(observed, fitted):
    avg_y = observed.mean()
    # sum of squares of residuals
    ssres = ((observed - fitted)**2).sum()
    # total sum of squares
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

# linear fit without weights
def fit_linear(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    p, residuals, _, _ = np.linalg.lstsq(X, y, rcond = None)
    x_fitted = np.array(x)
    y_fitted = np.polyval(p, x_fitted)
    Rsquared = calc_r2(y, y_fitted)
    # p[0] is the slope
    # p[1] is the intercept
    slope = p[0]
    intercept = p[1]
    return x_fitted, y_fitted, slope, intercept, Rsquared

def perpendicular_distance(slope, intercept, x_point, y_point):
    # source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    numerator = np.abs(slope*x_point + (-1)*y_point + intercept)
    denominator = distance(slope, (-1), 0, 0)
    d = numerator/denominator
    return d

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def classification(value, totalbins, rango):
    # Bin the data. Classify a value into a bin.
    # totalbins = number of bins to divide rango (range)
    bin_max = totalbins - 1
    numbin = 0
    inf = rango[0]
    sup = rango[1]
    if value > sup:
        print('Value higher than max')
        return bin_max
    if value < inf:
        print('Value lower than min')
        return 0
    step = (sup - inf)/totalbins
    # tiene longitud totalbins + 1
    # pero en total son totalbins "cajitas" (bines)
    binned_range = np.arange(inf, sup + step, step)
    while numbin < bin_max:
        if (value >= binned_range[numbin] and value < binned_range[numbin+1]):
            break
        numbin += 1
        if numbin > bin_max:
            break
    return numbin
#%% German Adittions
def position_peaks(in_order_peaks, analysis_radius, number_of_frames, index_peaks,
                   x_hist_COARSE_centers,y_hist_COARSE_centers,
                   x_position_of_picked,y_position_of_picked, frame_of_picked,photons_of_picked):
    """ To Find the peaks every time is needed """
    # analysis_radius = radius_of_pick_to_average*pixel_size
    cm_binding_sites_x = np.array([])
    cm_binding_sites_y = np.array([])
    cm_std_dev_binding_sites_x = np.array([])
    cm_std_dev_binding_sites_y = np.array([])
    # array where traces are going to be saved
    all_traces_per_pick = np.zeros(number_of_frames)
    all_traces = np.copy(all_traces_per_pick)
    for j in in_order_peaks:
        print('Binding site %d of %d' % (j+1, len(in_order_peaks)))
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
        print('CM binding in nm (x y) A:', cm_binding_site_x*1e3, \
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
        # compile traces of the pick in one array
        all_traces_per_pick = np.vstack([all_traces_per_pick, trace])
        
        # compile all traces of the image in one array
        all_traces = np.vstack([all_traces, trace])
    # delete first fake and empty trace (needed to make the proper array)
    all_traces_per_pick = np.delete(all_traces, 0, axis = 0)
    all_traces_per_pick = all_traces_per_pick.T

    return cm_binding_sites_x, cm_binding_sites_y,cm_std_dev_binding_sites_x, cm_std_dev_binding_sites_y, frame_of_picked_filtered, photons_of_picked_filtered, all_traces_per_pick
  #%%
def matrix_calculation(in_order, cm_binding_sites_x, cm_binding_sites_y,
                         cm_std_dev_binding_sites_x, cm_std_dev_binding_sites_y):
      
    total_peaks_found = len(in_order)
    # Me not use NP (Germán)
    matrix_distance = np.zeros([total_peaks_found, total_peaks_found])
    matrix_std_dev = np.zeros([total_peaks_found, total_peaks_found])
    # calcualte first row of the matrix distance
    positions_concat_origami = np.array([])
    # calcualte the rest of the rows of the matrix distance
    for j in range(total_peaks_found):
        x_binding_row = cm_binding_sites_x[in_order[j]]
        y_binding_row = cm_binding_sites_y[in_order[j]]
        matrix_std_dev[j, j] = max(cm_std_dev_binding_sites_x[in_order[j]], \
                                   cm_std_dev_binding_sites_y[in_order[j]])*1e3
        for k in range(j, total_peaks_found):
            x_binding_col = cm_binding_sites_x[in_order[k]]
            y_binding_col = cm_binding_sites_y[in_order[k]]
            distance_between_locs_CM = distance(x_binding_col, y_binding_col, \
                                                x_binding_row, y_binding_row)*1e3
            matrix_distance[j, k] = distance_between_locs_CM
            matrix_distance[k, j] = distance_between_locs_CM
            positions_concat_origami = np.append(positions_concat_origami, distance_between_locs_CM)
    return matrix_distance, matrix_std_dev, positions_concat_origami

def plot_matrix_distance (matrix_distance,i, N=1):
    import matplotlib.pyplot as plt
    # plot matrix distance    
    plt.figure(N)
    plt.imshow(matrix_distance, interpolation='none', cmap='spring')
    ax = plt.gca()
    for l in range(matrix_distance.shape[0]):
        for m in range(matrix_distance.shape[1]):
            if l == m:
                ax.text(m, l, '%.0f' % matrix_distance[l, m], #, '-' ,
                    ha="center", va="center", color=[0,0,0], 
                    fontsize = 18)
            else:
                ax.text(m, l, '%.0f' % matrix_distance[l, m],
                    ha="center", va="center", color=[0,0,0], 
                    fontsize = 18)
    ax.xaxis.tick_top()
    ax.set_xticks(np.array(range(matrix_distance.shape[1])))
    ax.set_yticks(np.array(range(matrix_distance.shape[0])))
    axis_string = []  # ['NP']
    for j in range(matrix_distance.shape[1]):
        axis_string.append('Site %d' % (j+1))
    ax.set_xticklabels(axis_string)
    ax.set_yticklabels(axis_string)
    # aux_folder = manage_save_directory(figures_per_pick_folder,'order_matrix_distance')
    figure_name = 'order_matrix_distance_%02d' % i
    # figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
    # plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
    # plt.show()
    # plt.close()