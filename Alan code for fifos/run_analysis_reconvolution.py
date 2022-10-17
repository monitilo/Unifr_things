# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:27:36 2021
@author: alanszalai
"""

import numpy as np
from aux_functions import obtain_histogram_values, shift_IRF
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from aux_functions import gauss_function, shift_IRF,generate_IRF, obtain_histogram_values,read_ptu
import scipy.signal
import numpy as np
from wavelet_analysis import parse_file, process_entry
    

def analysis_data(biexp_analysis, Simulate_IRF, Simulate_data, channel_data, 
                         filename_data, filename_IRF, FWHM_IRF, lifetime_sim, 
                         number_of_photons_sim, int_threshold, time1_threshold,
                         time2_threshold, bin_macrotime,microtime1_threshold_var, 
                         microtime2_threshold_var,
                         a1_guess, lifetime_guess, 
                         IRF_pos_guess, array_of_arrival_times_list, fileformat,
                         bin_size_nanotime):
    

    
  
       
        
    pulse_width = (FWHM_IRF/2.4)*1E-12 # rough approximation of sigma (1/2.4 FWHM) in s      
       
    #%% Either for simulated or real data, the time_bin is fixed to 'time_range' to perform the convolution accurately
    
    time_range = np.arange(start = 0, stop = 5E-8, step = 1E-11) # bins for the histogram of photon arrival times
    
    #%% IRF data 
        
    if Simulate_IRF == True:
        
        # Parameters
        
        init_pulse = 8.5E-9 # time at which the pulse appears
        bins_histogram_data_IRF = time_range[:-1]+0.5*(time_range[1]-time_range[0])
        values_histogram_data_IRF = generate_IRF(pulse_width, init_pulse, bins_histogram_data_IRF)


        values_x_histogram_data_IRF = 50*values_histogram_data_IRF
        micro_channel_IRF = np.repeat(bins_histogram_data_IRF, values_x_histogram_data_IRF.astype(int)) # ESTO NO ESTA FUNCIONANDO 
        time_IRF = bins_histogram_data_IRF     
        counts_IRF = values_histogram_data_IRF

        
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
        else:          
            micro_channel_IRF = filename_IRF[0]
            time_IRF = filename_IRF[1]
            counts_IRF = filename_IRF[2]
          
    
#%% if analyzing real data, we can set an intensity threshold and macrotime limits
       
    if Simulate_data == False:
    
        # array_of_arrival_times_list = list()   
        
        if fileformat == '.fifo':
            arr_data = parse_file(filename_data) 
            [dtime_channel_data,micro_channel_data] = process_entry(arr_data,channel_data)
        elif fileformat == '.ptu':
            micro_channel_data,dtime_channel_data = read_ptu(filename_data, channel_data, 'data')
        else:
            data = np.genfromtxt(filename_data, delimiter = ',')
            micro_channel_data = data[:,0]
            dtime_channel_data = data[:,1]
        
        
        # label valid data  
#        dtime_channel_data = dtime_channel_data*1e10
        dtime_sorted = np.sort(dtime_channel_data)
        dtime_min_diff = 0
        i = 0
        while dtime_min_diff == 0:
            dtime_min_diff = np.abs(dtime_sorted[i+1]-dtime_sorted[i])
            i += 1
        macrotime = np.arange(start = 0, stop = np.max(dtime_channel_data), step =  bin_macrotime)
    
        print("BBB", dtime_channel_data.shape, macrotime.shape)
        print("CCC", np.max(dtime_channel_data), bin_macrotime)
        bins_macrotime, values_macrotime = obtain_histogram_values(dtime_channel_data, macrotime)
               
                
        if isinstance(int_threshold, float) and int_threshold>0:
            idx_valid_above_int_threshold = np.where(values_macrotime>int_threshold)
            diff_idx_valid_above_int_threshold = np.diff(idx_valid_above_int_threshold[0])
            idx_idx_off = np.where(diff_idx_valid_above_int_threshold>1)
            idx_off = idx_valid_above_int_threshold[0][idx_idx_off]

            if values_macrotime[-1] < int_threshold:
                idx_off = np.hstack((idx_off,idx_valid_above_int_threshold[0][-1]))
            else:
                idx_off = np.hstack((idx_off,len(bins_macrotime)-1))
                
                
            idx_below_int_threshold = np.setdiff1d(np.arange(0,len(values_macrotime),1),idx_valid_above_int_threshold[0])
            diff_idx_below_int_threshold = np.diff(idx_below_int_threshold)
            idx_idx_on = np.where(diff_idx_below_int_threshold>1)
            idx_on = idx_below_int_threshold[idx_idx_on]

            if values_macrotime[0] > int_threshold:                
                idx_on = np.hstack((0,idx_on))
                
            if bins_macrotime[idx_below_int_threshold[-1]]<bins_macrotime[-1]:
                idx_on = np.hstack((idx_on,idx_below_int_threshold[-1]))
            

            micro_channel_data_i = np.array([])
            for i in range(len(idx_on)):
                micro_channel_data_i =  np.hstack((micro_channel_data_i,
                                                   micro_channel_data[np.where(np.logical_and(dtime_channel_data>bins_macrotime[int(idx_on[i])], 
                                                                                              dtime_channel_data<bins_macrotime[int(idx_off[i])]))]))

            micro_channel_data = micro_channel_data_i
        else:
            if isinstance(time1_threshold,float) and isinstance(time2_threshold,float):  
                micro_channel_data =  micro_channel_data[np.where(np.logical_and(dtime_channel_data>time1_threshold, dtime_channel_data<time2_threshold))]
            idx_on = 0
            idx_off = 0
        # filter microtime using microtime limits
        
        
        if isinstance(microtime1_threshold_var,float):
            micro_channel_data = micro_channel_data[micro_channel_data>1E-9*microtime1_threshold_var]
        if isinstance(microtime2_threshold_var,float):
            micro_channel_data = micro_channel_data[micro_channel_data<1E-9*microtime2_threshold_var]
        

        array_of_arrival_times_list.append(micro_channel_data)      
      
    
        max_time =  np.max(array_of_arrival_times_list[0])  
        max_macrotime = np.max(bins_macrotime)

        
    else:
        macrotime = 1 # Simulations only include the decay curve, and not the macrotime trace
        max_time = np.max(array_of_arrival_times_list[0])  
        bins_macrotime = 1
        values_macrotime = 1
        max_macrotime = 1
        idx_on = 0
        idx_off = 0

            
    #%% Data analysis 
    
    if bin_size_nanotime > 0:
        bin_size = bin_size_nanotime
    else:
        if fileformat == '.ptu':
            bin_size = 0.5E-10
        else:
            bin_size = 1E-11      
    

    time_range_final = np.arange(start = 0, stop = max_time, step = bin_size) # bins for the histogram of photon arrival times 

    bins_histogram_data_IRF, values_histogram_data_IRF = obtain_histogram_values(micro_channel_IRF, time_range_final) # Extract values and bins from the complete histogram
    

    # Analysis for simulated data

    if Simulate_data == True:
        
        idx_simulation = int(filename_data[-1])-1 # We set here the index of the simulation to be analyzed, based on the name of the file
        
        min_lifetime = np.zeros(len(array_of_arrival_times_list)) # lifetime obtained from fit
        min_IRF_pos = np.zeros(len(array_of_arrival_times_list)) # IRF position obtained from fit

        filename = 'lifetime_'+str(np.round(lifetime_sim*1E9, decimals = 2))[0] # name used for files saved after fivalues_macrotimeing        
        min_lifetime, number_of_photons_guess, a1, bg, min_IRF_pos, bins_histogram,values_decay, decay_array_fit, residuals = run_analysis(pulse_width,bins_histogram_data_IRF, values_histogram_data_IRF,
                                                           array_of_arrival_times_list[idx_simulation],
                                                           time_range_final, filename, a1_guess,
                                                           lifetime_guess, IRF_pos_guess, biexp_analysis) # run analysis # CORREGIR HARDCODEADA DE TIME LIST [0]

    # Analysis for experimental data 
        
    else:
          
        min_lifetime = np.zeros(len(array_of_arrival_times_list))
        min_IRF_pos = np.zeros(len(array_of_arrival_times_list))
        array_of_arrival_times = np.array(array_of_arrival_times_list)
        if len(array_of_arrival_times) == 1:
            array_of_arrival_times = array_of_arrival_times[0]

        
            if bin_size_nanotime > 0:
                bin_size = bin_size_nanotime
            else:
                if len(array_of_arrival_times)< 499:
                   bin_size = 3E-10            
                elif (len(array_of_arrival_times) >= 500 and len(array_of_arrival_times) < 999):
                    bin_size = 2E-10
                elif (len(array_of_arrival_times) >= 1000 and len(array_of_arrival_times) < 4999):
                    bin_size = 1E-10
                else:
                    if fileformat == '.ptu':
                        bin_size = 0.5E-10
                    else:
                        bin_size = 0.1E-10  
            
        
        time_range_final = np.arange(start = 0, stop = max_time, step = bin_size) # bins for the histogram of photon arrival times
        


        
        
        bins_histogram_data_IRF, values_histogram_data_IRF = obtain_histogram_values(micro_channel_IRF, time_range_final) # Extract values and bins from the complete histogram

        min_lifetime, number_of_photons_guess, a1, bg, min_IRF_pos, bins_histogram,values_decay, decay_array_fit, residuals = run_analysis(pulse_width,bins_histogram_data_IRF, values_histogram_data_IRF,
                                                   array_of_arrival_times,
                                                   time_range_final, filename_data, a1_guess,
                                                   lifetime_guess, IRF_pos_guess, biexp_analysis) # run analysis       
    
    
#%%   
       
    return min_lifetime, number_of_photons_guess, a1, min_IRF_pos, bins_histogram, values_decay, decay_array_fit, residuals, bins_macrotime, values_macrotime, max_time, max_macrotime, 1E-9*time_IRF, counts_IRF, bg, idx_on, idx_off, bin_size


#%%

def run_analysis(pulse_width, bins_histogram_data_IRF, values_histogram_data_IRF,
                 array_of_arrival_times, time_range, filename, a1_guess, 
                 lifetime_guess, IRF_pos_guess, biexp_analysis):
    
    #%% Load data and subtract background
    
    bins_histogram, values_histogram = obtain_histogram_values(array_of_arrival_times, time_range) # Obtain an histogram of fluorescence lifetime
    
    
    values_decay = values_histogram # it is called wout bg, but it has bg (to do: change name of variable)
    
    
    
    number_of_photons_guess = np.sum(values_decay) # Obtain the number of photons from the sum of values from the corrected histogram

    values_histogram_data_IRF =  values_histogram_data_IRF # Substract background to IRF data

    #%% Obtain initial guess
      
    IRF_guess = values_histogram_data_IRF
    try:
        decay_array_smoothed = scipy.signal.savgol_filter(values_decay, 
                                                      window_length=np.min(np.array([51,len(values_decay)])),
                                                      polyorder=3)
    except:
        decay_array_smoothed = scipy.signal.savgol_filter(values_decay, 
                                                        window_length=len(values_decay)-1,
                                                        polyorder=3)  
    deriv_decay = np.diff(decay_array_smoothed)
    fwhm1_pulse_idx = np.where(deriv_decay == np.max(deriv_decay))
    fwhm1_pulse_seconds = bins_histogram[fwhm1_pulse_idx[0]]
    center_pulse_idx = np.where( decay_array_smoothed == np.max( decay_array_smoothed))
    center_pulse_seconds = bins_histogram[center_pulse_idx[0]]   

    start_pulse_seconds = np.mean([center_pulse_seconds,fwhm1_pulse_seconds])

    
    
    if IRF_pos_guess == 0: # if the initial guess for the position is not definied by user
        IRF_pos_guess = start_pulse_seconds
        check_float = isinstance(IRF_pos_guess, float)
        if check_float == False:
            IRF_pos_guess = IRF_pos_guess[0]
            
    step_check_IRF = 0.01E-9
    range_check_IRF = 0.5E-9
    start_pulse_seconds_possibilities = np.arange(IRF_pos_guess-range_check_IRF, 
                                                  IRF_pos_guess+range_check_IRF,
                                                  step_check_IRF)
    
    bg_guess = np.mean(values_histogram[:50])
    try:
        bg_guess_possibilities = np.arange(0.1*bg_guess, 3*bg_guess, 0.25*bg_guess) 
    except:
        bg_guess_possibilities = np.array([bg_guess, 2*bg_guess]) # in case bg_guess = 0


        
    # Monoexponential 
    if biexp_analysis == False:
        if lifetime_guess == 0: # if initial guess for lifetime is not definied by user
            lifetime_initial_guess = np.median(array_of_arrival_times[array_of_arrival_times>center_pulse_seconds])-center_pulse_seconds
        else:
            lifetime_initial_guess = lifetime_guess

    
    # Biexponential
    else:
        if lifetime_guess[0] == 0: # if initial guess for lifetime is not definied by user    
            lifetime1_biexp_initial_guess = 0.5*(np.median(array_of_arrival_times[array_of_arrival_times>center_pulse_seconds])-
                                                 center_pulse_seconds)
        else:
            lifetime1_biexp_initial_guess = lifetime_guess[0]
        if lifetime_guess[1] == 0:
            lifetime2_biexp_initial_guess = 2*(np.median(array_of_arrival_times[array_of_arrival_times>center_pulse_seconds])-
                                               center_pulse_seconds)
        else:
            lifetime2_biexp_initial_guess = lifetime_guess[1]
        if a1_guess == 0:
            a1_guess = 0.5

        
    #%% Perform fivalues_macrotimeing
    

    # Monoexp
    
    Fit_exponential_including_bg = True # Consider the background as a parameter to fit
    Fit_biexponential_including_bg = False # Consider the background as a parameter to fit
    
    if Fit_exponential_including_bg:
        def func_conv(x, IRF_pos_guess, lifetime, number_of_photons, bg): 
            IRF = shift_IRF(IRF_guess, IRF_pos_guess, bins_histogram)
            decay_array = np.convolve(IRF, np.exp(-x/lifetime))
            decay_array = decay_array[:int((0.5*len(decay_array)+1))]
            decay_array = decay_array*(number_of_photons/np.sum(decay_array)) # adapt decay to #photons
            decay_array = decay_array+bg
            
            return decay_array
        
    else:
        def func_conv(x, IRF_pos_guess, lifetime, number_of_photons): 
            IRF = shift_IRF(IRF_guess, IRF_pos_guess, bins_histogram)
            decay_array = np.convolve(IRF, np.exp(-x/lifetime))
            decay_array = decay_array[:int((0.5*len(decay_array)+1))]
            decay_array = decay_array*(number_of_photons/np.sum(decay_array)) # adapt decay to #photons

            return decay_array
        
        # Biexp

    if Fit_biexponential_including_bg:
        
        def func_conv_biexp(x,IRF_pos_guess, a1, lifetime1,lifetime2, number_of_photons,bg_biexp):    
            IRF = shift_IRF(IRF_guess, IRF_pos_guess, bins_histogram)
            decay_array_biexp = np.convolve(IRF, a1*np.exp(-x/lifetime1)+(1-a1)*np.exp(-x/lifetime2))
            decay_array_biexp = decay_array_biexp[:int((0.5*len(decay_array_biexp)+1))]
            decay_array_biexp = decay_array_biexp*(number_of_photons/np.sum(decay_array_biexp)) # adapt decay to #photons
            decay_array_biexp = decay_array_biexp+bg_biexp

            return decay_array_biexp
        
    else:        
        def func_conv_biexp(x,IRF_pos_guess, a1, lifetime1,lifetime2, number_of_photons):    
            IRF = shift_IRF(IRF_guess, IRF_pos_guess, bins_histogram)
            decay_array_biexp = np.convolve(IRF, a1*np.exp(-x/lifetime1)+(1-a1)*np.exp(-x/lifetime2))
            decay_array_biexp = decay_array_biexp[:int((0.5*len(decay_array_biexp)+1))]
            decay_array_biexp = decay_array_biexp*(number_of_photons/np.sum(decay_array_biexp)) # adapt decay to #photons

            return decay_array_biexp
     


    def parameter_estimation_function(x,data):
        return data - func_conv(bins_histogram, IRF_pos_guess=x[0], lifetime=x[1], number_of_photons=x[2], bg=x[3])   
    def parameter_estimation_function_biexp(x,data):
        return data - func_conv_biexp(bins_histogram,IRF_pos_guess=x[0],
                                a1=x[1], lifetime1=x[2], lifetime2=x[3], number_of_photons=x[4], bg_biexp=x[5])   
    
    def parameter_estimation_no_bg_function(x,data):
        return data - func_conv(bins_histogram, IRF_pos_guess=x[0], lifetime=x[1], number_of_photons=x[2])   

    def parameter_estimation_no_bg_function_biexp(x,data):
        return data - func_conv_biexp(bins_histogram,IRF_pos_guess=x[0],
                                a1=x[1], lifetime1=x[2], lifetime2=x[3], number_of_photons=x[4])    
    
    if biexp_analysis == False:
        residuals_sum = np.zeros((len(start_pulse_seconds_possibilities),))
        for i in range(len(start_pulse_seconds_possibilities)):
            IRF_pos_guess = start_pulse_seconds_possibilities[i]
            if Fit_exponential_including_bg == True:
        
                sol = least_squares(parameter_estimation_function,[IRF_pos_guess,lifetime_initial_guess,
                                                                   number_of_photons_guess, bg_guess],
                    args=(values_decay,),method='lm',jac='2-point',max_nfev=5000)
                
            
                results = sol['x']
                decay_array_fit = func_conv(bins_histogram, results[0], results[1], results[2], results[3])
                bg_fit = results[3]

                residuals_sum[i] = np.sum(np.abs(values_decay-decay_array_fit))

                
            else:
                sol = least_squares(parameter_estimation_no_bg_function,[IRF_pos_guess,lifetime_initial_guess,number_of_photons_guess],
                    args=(values_decay-np.mean(values_decay[:10]),),method='lm',jac='2-point',max_nfev=2000)
                results = sol['x']
                decay_array_fit = func_conv(bins_histogram, results[0], results[1], results[2])   
                residuals_sum[i] = np.sum(np.abs(values_decay-np.mean(values_decay[:10])-decay_array_fit))



        start_pulse_seconds_best_fit_idx = np.where(residuals_sum == np.min(residuals_sum))
        start_pulse_seconds_best_fit =  start_pulse_seconds_possibilities[start_pulse_seconds_best_fit_idx[0]]
        IRF_pos_guess = start_pulse_seconds_best_fit[0]
        
        residuals_sum = np.zeros((len(bg_guess_possibilities,)))
        
        for i in range(len(bg_guess_possibilities)):
            if Fit_exponential_including_bg == True:
                bg_guess = bg_guess_possibilities[i]
                sol = least_squares(parameter_estimation_function,[IRF_pos_guess,lifetime_initial_guess,
                                                                   number_of_photons_guess, bg_guess],
                    args=(values_decay,),method='lm',jac='2-point',max_nfev=5000)
                
            
                results = sol['x']
                decay_array_fit = func_conv(bins_histogram, results[0], results[1], results[2], results[3])
                bg_fit = results[3]
                residuals_sum[i] = np.sum(np.abs(values_decay-decay_array_fit))
                
            else:
                sol = least_squares(parameter_estimation_no_bg_function,[IRF_pos_guess,lifetime_initial_guess,number_of_photons_guess],
                    args=(values_decay-np.mean(values_decay[:10]),),method='lm',jac='2-point',max_nfev=2000)
                results = sol['x']
                decay_array_fit = func_conv(bins_histogram, results[0], results[1], results[2])    
                bg_fit = 0
                residuals_sum[i] = np.sum(np.abs(values_decay-decay_array_fit))

        
        bg_guess_best_fit_idx = np.where(residuals_sum == np.min(residuals_sum))
        bg_guess_best_fit =  bg_guess_possibilities[bg_guess_best_fit_idx[0]]
        bg_guess = bg_guess_best_fit[0]
        

        if Fit_exponential_including_bg == True:

            sol = least_squares(parameter_estimation_function,[IRF_pos_guess,lifetime_initial_guess,
                                                               number_of_photons_guess, bg_guess],
                args=(values_decay,),method='lm',jac='2-point',max_nfev=5000)
            
        
            results = sol['x']
            decay_array_fit = func_conv(bins_histogram, results[0], results[1], results[2], results[3])
            bg_fit = results[3]
        else:
            sol = least_squares(parameter_estimation_no_bg_function,[IRF_pos_guess,lifetime_initial_guess,number_of_photons_guess],
                args=(values_decay-np.mean(values_decay[:10]),),method='lm',jac='2-point',max_nfev=2000)
            results = sol['x']
            decay_array_fit = func_conv(bins_histogram, results[0], results[1], results[2])    
            bg_fit = 0                    
        
        min_lifetime = results[1]
        min_IRF_pos = results[0]
        a1 = 0
        
        
    else:
        method_biexp = 'trf'
        bounds_nobg = ([0,0,0,0,0],[1,1,1E-8,1E-7,np.inf]) 
        bounds_bg = ([0,0,0,0,0,0],[1,1,1E-8,1E-7,np.inf,np.inf]) 
        residuals_sum = np.zeros((len(start_pulse_seconds_possibilities),))
        for i in range(len(start_pulse_seconds_possibilities)):
            IRF_pos_guess = start_pulse_seconds_possibilities[i]
            if Fit_biexponential_including_bg == True:
                sol = least_squares(parameter_estimation_function_biexp,
                                          [IRF_pos_guess,a1_guess, lifetime1_biexp_initial_guess,
                                           lifetime2_biexp_initial_guess,number_of_photons_guess,
                                           bg_guess],
                    args=(values_decay,),method=method_biexp,jac = '2-point',max_nfev=5000,    
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, 
                    diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,  verbose=0,
                    bounds=bounds_bg)    
        
                results = sol['x']   
                decay_array_fit = func_conv_biexp(bins_histogram,results[0], 
                                                        results[1], results[2],
                                                        results[3], results[4],
                                                        results[5])
                min_IRF_pos = results[0]
                min_lifetime = np.array([results[2], results[3]])
                min_IRF_pos = results[0]
                a1 = results[1]
                bg_fit = results[5]
                residuals_sum[i] = np.sum(np.abs(values_decay-decay_array_fit))
            else:
                
                sol = least_squares(parameter_estimation_no_bg_function_biexp,
                                          [IRF_pos_guess,a1_guess, lifetime1_biexp_initial_guess,
                                           lifetime2_biexp_initial_guess,number_of_photons_guess],
                    args=(values_decay-np.mean(values_decay[:10]),),method=method_biexp,jac = '2-point',max_nfev=2000,    
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, 
                    diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,  verbose=0,
                    bounds=bounds_nobg)       
        
                results = sol['x']   
                decay_array_fit = func_conv_biexp(bins_histogram,results[0], 
                                                        results[1], results[2],
                                                        results[3], results[4])
                min_IRF_pos = results[0]
                min_lifetime = np.array([results[2], results[3]])
                min_IRF_pos = results[0]
                a1 = results[1]
                bg_fit = 0
                residuals_sum[i] = np.sum(np.abs(values_decay-np.mean(values_decay[:10])-decay_array_fit))
            
                
        start_pulse_seconds_best_fit_idx = np.where(residuals_sum == np.min(residuals_sum))
        start_pulse_seconds_best_fit =  start_pulse_seconds_possibilities[start_pulse_seconds_best_fit_idx[0]]
        IRF_pos_guess = start_pulse_seconds_best_fit[0]
        
        residuals_sum = np.zeros((len(bg_guess_possibilities,)))
        for i in range(len(bg_guess_possibilities)):
            if Fit_biexponential_including_bg == True:
                bg_guess = bg_guess_possibilities[i]
                sol = least_squares(parameter_estimation_function_biexp,
                                          [IRF_pos_guess,a1_guess, lifetime1_biexp_initial_guess,
                                           lifetime2_biexp_initial_guess,number_of_photons_guess,
                                           bg_guess],
                    args=(values_decay,),method=method_biexp,jac = '2-point',max_nfev=5000,    
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, 
                    diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,  verbose=0,
                    bounds=bounds_bg)    
        
                results = sol['x']   
                decay_array_fit = func_conv_biexp(bins_histogram,results[0], 
                                                        results[1], results[2],
                                                        results[3], results[4],
                                                        results[5])
                min_IRF_pos = results[0]
                min_lifetime = np.array([results[2], results[3]])
                min_IRF_pos = results[0]
                a1 = results[1]
                bg_fit = results[5]
                residuals_sum[i] = np.sum(np.abs(values_decay-decay_array_fit))
            else:
                bg_guess = bg_guess_possibilities[i]
                sol = least_squares(parameter_estimation_no_bg_function_biexp,
                                          [IRF_pos_guess,a1_guess, lifetime1_biexp_initial_guess,
                                           lifetime2_biexp_initial_guess,number_of_photons_guess],
                    args=(values_decay-np.mean(values_decay[:10]),),method=method_biexp,jac = '2-point',max_nfev=5000,    
                    ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, 
                    diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,  verbose=0,
                    bounds=bounds_nobg)       
        
                results = sol['x']   
                decay_array_fit = func_conv_biexp(bins_histogram,results[0], 
                                                        results[1], results[2],
                                                        results[3], results[4])
                min_IRF_pos = results[0]
                min_lifetime = np.array([results[2], results[3]])
                min_IRF_pos = results[0]
                a1 = results[1]
                bg_fit = 0
                residuals_sum[i] = np.sum(np.abs(values_decay-decay_array_fit))
        
        bg_guess_best_fit_idx = np.where(residuals_sum == np.min(residuals_sum))
        bg_guess_best_fit =  bg_guess_possibilities[bg_guess_best_fit_idx[0]]
        bg_guess = bg_guess_best_fit[0]       
        
        
        if Fit_biexponential_including_bg:
            sol = least_squares(parameter_estimation_function_biexp,
                                      [IRF_pos_guess,a1_guess, lifetime1_biexp_initial_guess,
                                       lifetime2_biexp_initial_guess,number_of_photons_guess,
                                       bg_guess],
                args=(values_decay,),method=method_biexp,jac = '2-point',max_nfev=2000,    
                ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, 
                diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,  verbose=0,
                bounds=bounds_bg)    
        
            results = sol['x']   
            decay_array_fit = func_conv_biexp(bins_histogram,results[0], 
                                                    results[1], results[2],
                                                    results[3], results[4],
                                                    results[5])
            min_IRF_pos = results[0]
            min_lifetime = np.array([results[2], results[3]])
            min_IRF_pos = results[0]
            a1 = results[1]
            bg_fit = results[5]
                
            
        else:
            sol = least_squares(parameter_estimation_no_bg_function_biexp,
                                      [IRF_pos_guess,a1_guess, lifetime1_biexp_initial_guess,
                                       lifetime2_biexp_initial_guess,number_of_photons_guess],
                args=(values_decay-np.mean(values_decay[:10]),),method=method_biexp,jac = '2-point',max_nfev=5000,    
                ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='linear', f_scale=1.0, 
                diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None,  verbose=0,
                bounds=bounds_nobg)       
    
            results = sol['x']   
            decay_array_fit = func_conv_biexp(bins_histogram,results[0], 
                                                    results[1], results[2],
                                                    results[3], results[4])
            min_IRF_pos = results[0]
            min_lifetime = np.array([results[2], results[3]])
            min_IRF_pos = results[0]
            a1 = results[1]
            bg_fit = 0
                
    
    
    residuals = values_decay-decay_array_fit

    #%%
    return min_lifetime, number_of_photons_guess, a1, bg_fit, min_IRF_pos, bins_histogram, values_decay, decay_array_fit, residuals