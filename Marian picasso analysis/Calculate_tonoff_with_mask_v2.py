# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:47:28 2019

@author: Cecilia Zaza

Para arreglar las trazas que fueron mal extraidas del trace inspector, posteriores al 
cambio realizado el 31.10.2019
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
from tkinter import Tk, filedialog
from itertools import groupby

plt.close("all")

root = Tk()
trace_file = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(trace_file)
trace_name = os.path.basename(trace_file)

print()
print(trace_name)

traces = np.loadtxt(folder + '/' + trace_name)
def mask(number_of_dips = 1):
    if number_of_dips == 1:
        mask_array = 0.5*np.array([0,1,-1,0,0])
    elif number_of_dips == 2:
        mask_array = 0.5*np.array([0,1,0,-1,0])
    elif number_of_dips == 3:
        mask_array = 0.5*np.array([0,1,0,0,-1,0])
    elif number_of_dips == -1:
        mask_array = 0.5*np.array([0,-1,1,0])
    elif number_of_dips == 99:
        mask_array = np.array([0,1,0])
    else:
        mask_array = np.array([0,0,0,0,0])
    return mask_array

def calculate_tau_on_times(trace, threshold, exposure_time): 
    # exposure_time in ms
    # threshold in number of photons (integer)
    number_of_frames = int(trace.shape[0])
    # while the trace is below the threshold leave 0, while is above replace by 1
    binary_trace = np.zeros(number_of_frames, dtype = int)
    binary_trace = np.where(trace < threshold, binary_trace, 1)

    # mask 1 step dips using convolution
    diff_binary = np.diff(binary_trace)
    conv_one_dip = sig.convolve(diff_binary, mask(1))
    localization_index_dips = np.where(conv_one_dip == 1)[0] - 1
    binary_trace[localization_index_dips] = 1
    
    # mask 2 step dips using convolution
    diff_binary = np.diff(binary_trace)
    conv_two_dip = sig.convolve(diff_binary, mask(2))
    localization_index_dips = np.where(conv_two_dip == 1)[0] - 1
    binary_trace[localization_index_dips] = 1
    localization_index_dips = np.where(conv_two_dip == 1)[0] - 2
    binary_trace[localization_index_dips] = 1

    # remove 1 step blips using convolution
    diff_binary = np.diff(binary_trace)
    conv_one_blip = sig.convolve(diff_binary, mask(-1))
    localization_index_blips = np.where(np.abs(conv_one_blip) == 1)[0] - 1
    binary_trace[localization_index_blips] = 0   
        
    # now, with the trace "restored" we can estimate tau_on...
    
    # estimate number of frames the dye was ON
    # keep indexes where localizations have been found (> 1)
    localization_index = np.where(binary_trace > 0)[0]
    localization_index_diff = np.diff(localization_index)
    keep_steps = np.where(localization_index_diff == 1)[0]
    localization_index_steps = localization_index[keep_steps]
    binary_trace[localization_index_steps] = 1
    
    # mask starting time using convolution
    diff_binary = np.diff(binary_trace)
    conv_start_time = sig.convolve(diff_binary, mask(99))
    localization_index_start = np.where(conv_start_time == 1)[0] - 1
    
    # ### uncomment plot to check filters and binary trace
    # plt.figure()
    # plt.plot(trace/max(trace))
    # plt.plot(binary_trace)
    # # plt.plot(conv_one_blip)
    # # plt.plot(conv_one_dip)
    # # plt.plot(conv_two_dip)
    # plt.xlim([0,500])
    # plt.show()
    
    t_on = [len(l) for l in [list(g) for k, g in groupby(list(binary_trace), key = lambda x:x!=0) if k]]
    t_off = [len(l) for l in [list(g) for k, g in groupby(list(binary_trace), key = lambda x:x==0) if k]]
    
    if binary_trace[0] == 1:
        t_on = t_on[1:]
    else:
        t_off = t_off[1:]
    if binary_trace[-1] == 1:
        t_on = t_on[:-1]
    else:
        t_off = t_off[:-1]     
        
    t_on = np.asarray(t_on)
    t_off = np.asarray(t_off)
    start_time = np.asarray(localization_index_start)
    
    return t_on*exposure_time, t_off*exposure_time, binary_trace, start_time*exposure_time

##############################################################################

exp_time = 100 # in ms
tons = np.array([],dtype = int)
toffs = np.array([],dtype = int)
tstarts = np.array([],dtype = int)
number_of_traces = int(traces.shape[1])
threshold = 10 # in number of photons

for i in range(number_of_traces):
# for i in range(1):
    trace = traces[:,i]
    [ton, toff, binary, tstart] = calculate_tau_on_times(trace, threshold, exp_time)     
    tons = np.append(tons, ton)
    toffs = np.append(toffs, toff)
    tstarts = np.append(tstarts, tstart)
        
tons = np.trim_zeros(tons)
toffs = np.trim_zeros(toffs) 
 
np.savetxt(folder+'\\'+'ton_'+trace_name[:-4]+'.txt', tons, fmt = '%d')
np.savetxt(folder+'\\'+'toff_'+trace_name[:-4]+'.txt', toffs, fmt = '%d')
np.savetxt(folder+'\\'+'tstarts_'+trace_name[:-4]+'.txt', tstarts, fmt = '%d')

print('\nTotal tau_ons', len(tons))

