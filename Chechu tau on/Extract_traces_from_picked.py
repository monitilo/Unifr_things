# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:04:41 2019

@author: Cecilia Zaza

Este programa extrae las trazas generadas desde las localizaciones hechas en el picasso.
Genera trazas ficticias desde las localizaciones, donde hay localizacion pone el numero
de fotones y donde no hay, va un cero. No hay corrección de drift porque las localizaciones 
ya estan corregidas por drift. Cuando te pide un archivo hay que poner la salida del
picasso que termina con "...picked.hdf5"
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog

root = Tk()
video_file = filedialog.askopenfilename(filetypes=(("", "*.hdf5"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(video_file)
video_name = os.path.basename(video_file)
#%%
filename = video_file
number_frames = 20000
with h5py.File(filename, 'r') as f:
    print('hola')
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

#    # Get the data
    data = list(f[a_group_key])
    
#%%
frame = np.zeros([len(data)])
x = np.zeros([len(data)])
y = np.zeros([len(data)])
photons = np.zeros([len(data)])
sx = np.zeros([len(data)])
sy = np.zeros([len(data)])
bg = np.zeros([len(data)])
lpx = np.zeros([len(data)])
lpy = np.zeros([len(data)])
ellipticity = np.zeros([len(data)])
net_gradient = np.zeros([len(data)])
group = np.zeros([len(data)])

for i in range(0, len(data)):
    frame[i] = data[i][0]
    x[i] = data[i][1]
    y[i] = data[i][2]
    photons[i] = data[i][3]
    sx[i] = data[i][4]
    sy[i] = data[i][5]
    bg[i] = data[i][6]
    lpx[i] = data[i][7]
    lpy[i] = data[i][8]
    ellipticity[i] = data[i][9]
    net_gradient[i] = data[i][10]
    group[i] = data[i][11]
#%%
traces = np.zeros([number_frames, int(group[-1])+1])
mean_frame_value_total = []
std_frame_value_total = []
frames_group = []

#%% si hay lio con los indices en la filter file, se arregla en este bloque
for k in range(0, int(group[-1])+1): #k barre para cada pick
    print(k)
    trace_start = np.where(group == k)[0][0]
    trace_end = np.where(group == k)[0][-1]
    mean_frame_value = np.mean(frame[trace_start:trace_end])
    std_frame_value = np.std(frame[trace_start:trace_end])
    mean_frame_value_total.append(mean_frame_value)
    std_frame_value_total.append(std_frame_value)
    for m in range(trace_start, trace_end):
        traces[int(frame[m]), k] = photons[m]
        
#%%
#
np.savetxt(folder+'\\'+'TRACES_'+video_name[:-4]+'txt',traces, fmt = '%.3e')
print('Termine baby :D')

#%%

