# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:35:48 2020

@author: Cibion2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from scipy.optimize import curve_fit
import os
from tkinter import Tk, filedialog

from scipy import optimize
from scipy import stats,signal
from scipy.stats import expon

def exponential(x,a,londa):
    return (a/londa)*np.exp(-x/londa)

def doble_exponential(x, a1, londa1, a2, londa2, offset):
    return (a1/londa1)*np.exp(-x/londa1)+(a2/londa2)*np.exp(-x/londa2)
#%%

total_files = 1  #elijo cuantos archivos juntos quiero analizar por ejemplo si tengo trazas de 1 o mas videos
names = []
for i in range(0, total_files):
    root = Tk()
    trace_file = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.")))
    names.append(trace_file)
    root.withdraw()
    folder = os.path.dirname(trace_file)
    trace_name = os.path.basename(trace_file)

#%% elijo que data analizar si pongo 'total' es todas las que quiero
data_number = 'total'
data = []

if data_number == 'total':
    for j in range(0, total_files):
        archive = np.loadtxt(names[j])
        if j == 1:
            print(archive.shape)
            data.extend(archive*2)
        else:
            data.extend(archive)
else:
    data = np.loadtxt(names[data_number]) 


data = np.asarray(data)#/1000 #para pasar de ms a s

#%% primero miro el histograma total de la data
Events_total = data


binwidth = 2*scipy.stats.iqr(data)*(data.shape[0])**(-1/3)
if binwidth < 0.10:
    binwidth = 0.101
else:
    binwidth = 0.3#2*scipy.stats.iqr(data)*(data.shape[0])**(-1/3)
    
bindwidth = 10
    
bins = np.arange(0, (int(max(Events_total)/binwidth)+2)*binwidth, binwidth)

#entries, bin_edges, patches = plt.hist(Events, bins, normed=False, histtype = 'step', cumulative=False, label=['1 N='+np.str(len(Events))], log=False)
entries, bin_edges = np.histogram(Events_total, bins)
bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
errors = np.sqrt(entries)

bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
errors = np.sqrt(entries)
plt.bar(bin_edges[:-1], entries, align = 'edge', width = binwidth)
plt.grid()
plt.yscale('log')

#%% corto la data

SACO_BIN = 1


bins = np.arange(0, (int(max(data)/binwidth)+2)*binwidth, binwidth)

entries, bin_edges = np.histogram(data, bins)
bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
errors = np.sqrt(entries)
plt.bar(bin_edges[:-1], entries, align = 'edge', width = binwidth, fill=False)
plt.grid()
plt.yscale('log')


bin_fit = bin_middles[entries > 0] #esto es para sacar los valores que son cero y ajustar
entries_fit = entries[entries > 0]
errors_fit = errors[entries > 0]

#proba10 = Tau_tentativo*(np.log(c) + np.log(1-np.exp(-binwidth/Tau_tentativo)) - np.log(10))
proba10 = bin_fit[np.where(entries<10)[0][1]]

#bin_fit_cut = bin_fit[bin_fit < proba10] 
#entries_fit_cut = entries_fit[bin_fit < proba10]
#errors_fit_cut = errors_fit[bin_fit < proba10]

bin_fit_cut = bin_fit[SACO_BIN:]
entries_fit_cut = entries_fit[SACO_BIN:]
errors_fit_cut = errors_fit[SACO_BIN:]

#plt.figure()
plt.plot(bin_fit,entries_fit, label='1')

plt.errorbar(bin_fit, entries_fit, yerr=errors_fit, fmt="none")
plt.grid()
plt.yscale('log')


Tau_tentativo = np.mean(data)
parameters_1, cov_matrix = curve_fit(exponential, bin_fit_cut, entries_fit_cut, p0 = [np.max(entries)-np.min(entries), Tau_tentativo], maxfev = 20000) 
parameters, cov_matrix = curve_fit(exponential, bin_fit_cut, entries_fit_cut, p0 = parameters_1, absolute_sigma = True, sigma = errors_fit_cut, maxfev = 20000) 
plt.plot(bin_middles, exponential(bin_middles, *parameters), color = 'C3')
plt.plot(bin_fit_cut, exponential(bin_fit_cut, *parameters), '-.')
plt.yscale('log')
plt.ylim((0.9, max(entries)+1000))

londa_medida = parameters[1]
error_ajuste = np.sqrt(cov_matrix[1,1])
print('SIN BIN', SACO_BIN)
print('londa exp', londa_medida)
print('error londa exp', error_ajuste)
print(len(data))


ene_ajustado = round(len(data) - entries[0]) + int(round(exponential(bin_middles, *parameters)[0]))#si trabajo sacando solo un bin
ene_corto = len(data) - ene_ajustado

#%% calculo error a parir de una simulacion de una biexponencial perfecta

#total_londas_exp = []
#total_err_londas_exp = []
#taus_total = []
#ene = []
#
#for c in np.array([len(data)]):#([ 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050]):
#    print(c)
#    londas_exp = []
#    error_londas_exp = []
#    taus = [] 
#    
#    T_ON = londa_medida
#    T_ON_short = 0.1
#    T_OFF = londa_medida*10
#
#    ene.append(int(c*(T_ON + T_OFF)/(T_ON_short + T_OFF))+c)
#    
#    for i in range(0, 10000):
##        plt.close('all')
#
#        
#        Events_long = stats.expon.rvs(scale = T_ON, size = ene_ajustado)
##        Events_short = stats.expon.rvs(scale = T_ON_short, size = int(c*(T_ON + T_OFF)/(T_ON_short + T_OFF)))
#        Events_short = stats.expon.rvs(scale = T_ON_short, size = ene_corto)
#        Events = np.concatenate((Events_long, Events_short))
#
##        plt.hist(Events,50)
#        Tau_tentativo = np.mean(Events)
##        print(np.mean(Events))
#        
##        plt.hist(Events,50)
##        print('Mean', np.mean(Events))
#        
#        binwidth = 2*scipy.stats.iqr(Events)*(len(Events))**(-1/3) #Freedman Diaconis rule
#        bins = np.arange(0, (int(max(Events)/binwidth)+2)*binwidth, binwidth)
#        
##        entries, bin_edges, patches = plt.hist(Events, bins, normed=False, histtype = 'step', cumulative=False, label=['1 N='+np.str(len(Events))], log=False)
#        entries, bin_edges = np.histogram(Events, bins)
#        bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
#        errors = np.sqrt(entries)
#        
##        plt.bar(bin_edges[:-1], entries, align = 'edge', width = binwidth, fill=False)
##        plt.grid()
##        plt.yscale('log')
#        
#        bin_fit = bin_middles[entries > 0] #esto es para sacar los valores que son cero y ajustar
#        entries_fit = entries[entries > 0]
#        errors_fit = errors[entries > 0]
#        
#        
#        #corto la data para hacer el ajuste
##        proba10 = Tau_tentativo*(np.log(c) + np.log(1-np.exp(-binwidth/Tau_tentativo)) - np.log(10))
##        proba10 = 3*Tau_tentativo
#        proba10 = bin_fit[np.where(entries<8)[0][0]]
#
#        
#        bin_fit_cut = bin_fit#[bin_fit < proba10] 
#        entries_fit_cut = entries_fit#[bin_fit < proba10]
#        errors_fit_cut = errors_fit#[bin_fit < proba10]
#        
#        bin_fit_cut = bin_fit_cut[SACO_BIN:]
#        entries_fit_cut = entries_fit_cut[SACO_BIN:]
#        errors_fit_cut = errors_fit_cut[SACO_BIN:]
#        
#        
##        plt.figure()
##        plt.plot(bin_fit,entries_fit, label='1')
##        plt.errorbar(bin_fit, entries_fit, yerr=errors_fit, fmt="none")
##        plt.grid()
#        
#        
#        #ajuste exponencial
#        parameters_1, cov_matrix = curve_fit(exponential, bin_fit_cut, entries_fit_cut, p0 = [np.max(entries)-np.min(entries), Tau_tentativo], maxfev = 20000) 
##        plt.plot(bin_middles, exponential(bin_middles, *parameters_1))
#        parameters, cov_matrix = curve_fit(exponential, bin_fit_cut, entries_fit_cut, p0 = parameters_1, absolute_sigma = True, sigma = errors_fit_cut, maxfev = 20000) 
##        plt.plot(bin_middles, exponential(bin_middles, *parameters), color = 'C3')
##        plt.plot(bin_fit_cut, exponential(bin_fit_cut, *parameters))
##        plt.yscale('log')
##        print('londa exp', parameters[1])
##        print('error londa exp', np.sqrt(cov_matrix[1,1]))
#        
#        if cov_matrix[1,1] < 1:
#            taus.append(np.mean(Events))
#            londas_exp.append(parameters[1])
#            error_londas_exp.append(np.sqrt(cov_matrix[1,1]))
#    
#    
#    total_londas_exp.append(londas_exp)
##    total_londas_lin.append(londas_lin)
#    total_err_londas_exp.append(error_londas_exp)
##    total_err_londas_lin.append(error_londas_lin)
#    taus_total.append(taus)
#    
#
##plt.figure(1)
#means_exp = []
#means_short = []
#means_taus = []
#std_exp = []
#std_short = []
#std_taus = []
#    
#for  j in range(0, len(taus_total)):
#    means_exp.append(np.mean(total_londas_exp[j]))
##    means_short.append(np.mean(total_londas_lin[j]))
#    means_taus.append(np.mean(taus_total[j]))
#    std_exp.append(np.std(total_londas_exp[j])) 
##    std_short.append(np.std(total_londas_lin[j])) 
#    std_taus.append(np.std(taus_total[j])) 
#
#print('SIN BIN', SACO_BIN)
#print('error final', std_exp)
#print('londa exp', londa_medida)
#print('error londa exp', error_ajuste)
#print(len(data))
