# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 13:50:46 2021

@author: alanszalai
"""

# Auxiliary functions

import numpy as np
from pqreader import load_ptu
from matplotlib import cm

#%%
def obtain_histogram_values(x, bins): # obtain bins and values from histogram
    histogram = np.histogram(x, bins)
    values_histogram = histogram[0]
    bins_histogram = histogram[1][:-1]+0.5*(histogram[1][1]-histogram[1][0])
    return bins_histogram, values_histogram

#%%
def gauss_function(x, mu, sigma): 
    y = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
    return y

#%%
def shift_IRF(values_IRF, center_guess, bins_IRF):
    if np.size(center_guess)>1:
        center_guess = center_guess[0]
    IRF_center_pos = bins_IRF[np.where(values_IRF == np.max(values_IRF))]
    if np.size(IRF_center_pos)>1:
        IRF_center_pos = IRF_center_pos[0]
    IRF_shifted = np.zeros(len(bins_IRF))
    shift = center_guess - IRF_center_pos

    shift_bins = shift/(bins_IRF[1]-bins_IRF[0])
    if shift > 0:
        IRF_shifted[int(shift_bins):] = values_IRF[:(len(values_IRF)-int(shift_bins))]
    else:
        IRF_shifted[:(len(IRF_shifted)-int(-shift_bins))] = values_IRF[int(-shift_bins):]
    return IRF_shifted 



#%%
def read_ptu(filename, channel, data_or_IRF):
    data = load_ptu(filename)
    dtime_channel_data = data[0]*data[3]['timestamps_unit']
    micro_channel_data = data[2]*data[3]['nanotimes_unit']    

    dtime_channel_data = dtime_channel_data[data[1]==channel]
    micro_channel_data = micro_channel_data[data[1]==channel]

    return micro_channel_data,dtime_channel_data
       

#%%
def get_intensity_lifetime_color(lifetime_ref_fraction,intensity_ref_fraction):    
    rgb_map = cm.jet(range(256)) # change map name with the map you want    
    #maps: brg
    
    rgb_map = rgb_map[:,:3]        
    color_only_lifetime = rgb_map[int((lifetime_ref_fraction*256)-1)]            
    fixed_lifetime_variable_intensity = np.zeros((256,3))

    fixed_lifetime_variable_intensity[:,0] = np.reshape(np.linspace(0,color_only_lifetime[0],
                                                            num=256), (256,))

    fixed_lifetime_variable_intensity[:,1] = np.reshape(np.linspace(0,color_only_lifetime[1],
                                                            num=256), (256,))

    fixed_lifetime_variable_intensity[:,2] = np.reshape(np.linspace(0,color_only_lifetime[2],
                                                            num=256), (256,))     
    intensity_color = fixed_lifetime_variable_intensity[int(intensity_ref_fraction*255)]
    
    return intensity_color

#%%
def generate_IRF(pulse_width, init_pulse, time_range):
    if init_pulse < 5*pulse_width:
        init_pulse = 5*pulse_width # to ensure having the whole pulse within the time range
    gauss_pulse = gauss_function(time_range, init_pulse, pulse_width) # gaussian pulse    
    return gauss_pulse


#%%
def generate_data(lifetime, IRF, number_of_photons, time_range):
    decay_array = np.convolve(IRF, np.exp(-time_range/lifetime))
    decay_array = decay_array[:int((0.5*len(decay_array)+1))]
    decay_array = decay_array*(number_of_photons/np.sum(decay_array)) # adapt decay to #photons
    decay_array = decay_array+0.01*np.max(decay_array)
    
    # Convert data into array of "photon time arrival"
    array_of_arrival_times = np.array([]) # raw data is usually obtained like this
    
    # Add Poisson noise to artificial data
    decay_array = np.random.poisson(decay_array)
    array_of_arrival_times = np.repeat(time_range, decay_array) # creates array of time arrival
        
    return array_of_arrival_times

#%%
def generate_data_biexp(a1, lifetime1,lifetime2, IRF, number_of_photons, x):
    decay_array = np.convolve(IRF, a1*np.exp(-x/lifetime1)+(1-a1)*np.exp(-x/lifetime2))
    decay_array = decay_array[:int((0.5*len(decay_array)+1))]
    decay_array = decay_array*(number_of_photons/np.sum(decay_array)) # adapt decay to #photons
    decay_array = decay_array+0.01*np.max(decay_array)

    
    # Convert data into array of "photon time arrival"
    array_of_arrival_times = np.array([]) # raw data is usually obtained like this
    
    # Add Poisson noise to artificial data
    decay_array = np.random.poisson(decay_array)
    array_of_arrival_times = np.repeat(x, decay_array) # creates array of time arrival
    
        
    return array_of_arrival_times

#%%
def simulate_data_func(monoexp_sim, Simulate_IRF, filename_IRF, FWHM_IRF, lifetime_sim, a1_sim, number_of_photons_sim, Number_of_repeats):
    
    time_range = np.arange(start = 0, stop = 5E-8, step = 1E-11) # bins for the histogram of photon arrival times
    pulse_width = (FWHM_IRF/2.4)*1E-12 # rough approximation of sigma (1/2.4 FWHM) in s  
    
    # IRF data 
        
    if Simulate_IRF == True:
        
        # Parameters
        
        init_pulse =1.5E-9 # time at which the pulse appears
        bins_histogram_data_IRF = time_range[:-1]+0.5*(time_range[1]-time_range[0])
        values_histogram_data_IRF = generate_IRF(pulse_width, init_pulse, bins_histogram_data_IRF)
                
        
    else:               
        
        if '.dat' in filename_IRF:
            time_IRF = np.array([])
            counts_IRF = np.array([])
            
            IRF_data_list = [i.strip().split() for i in open(filename_IRF).readlines()]
            
            comma_or_points_find = IRF_data_list[0][0].find('.')        
            if comma_or_points_find == -1:
                points = False
            else:
                points = True
            for i in range(len(IRF_data_list)):
                if points == False:
                    time_points_instead_of_comma = IRF_data_list[i][0].replace(',', '.').replace(' ', ';')
                    time_IRF = np.append(time_IRF, np.array(float(time_points_instead_of_comma)))
                else:
                    time_IRF = np.append(time_IRF, np.array(float(IRF_data_list[i][0])))
                counts_IRF = np.append(counts_IRF, np.array(float(IRF_data_list[i][1])))
        
    
            micro_channel_IRF = np.repeat(1E-9*time_IRF, counts_IRF.astype(int))         
            bins_histogram_data_IRF, values_histogram_data_IRF = obtain_histogram_values(micro_channel_IRF, time_range)   
        else:
            micro_channel_IRF = filename_IRF[0]
            bins_histogram_data_IRF, values_histogram_data_IRF = obtain_histogram_values(micro_channel_IRF, time_range)   
           
            
    # data generation
    
    if monoexp_sim == True:
        lifetime = 1E-9*lifetime_sim
        number_of_photons = number_of_photons_sim
        
        array_of_arrival_times_list = list() # since each simulation has different number of photons (Poisson noise), we save them the arrival times in a list
    
        for i in range(Number_of_repeats):
            array_of_arrival_times = generate_data(lifetime, values_histogram_data_IRF, number_of_photons, time_range) # simulates data by convolution (Gaussian pulse)
            array_of_arrival_times_list.append(array_of_arrival_times)  
    else:
        a1 = a1_sim
        lifetime1 = 1E-9*lifetime_sim[0]
        lifetime2 = 1E-9*lifetime_sim[1]
        number_of_photons = number_of_photons_sim
        array_of_arrival_times_list = list() # since each simulation has different number of photons (Poisson noise), we save them the arrival times in a list
    
        for i in range(Number_of_repeats):
            array_of_arrival_times = generate_data_biexp(a1, lifetime1, lifetime2, values_histogram_data_IRF, number_of_photons, time_range) # simulates data by convolution (Gaussian pulse)
            array_of_arrival_times_list.append(array_of_arrival_times)  
        
        
            
        

    
    return array_of_arrival_times_list