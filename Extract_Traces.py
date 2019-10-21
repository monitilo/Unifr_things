# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:59 2019

@author: Cecilia Zaza

Este programa toma un archivo de localizaciones realizadas con FIJI o Image J 
y extrae las trazas de esas localizaciones. TODO EN nm!! se divide por el 
pixel size (linea 39) para pasarlo a pixeles
-elegir el video a analizar
-para que tome bien las localizaciones realizadas a mano, se debe guardar el 
archivo con el nombre: locs_two_dots.txt en la misma carpeta que donde se 
encuentra el video. 
- El programa devuelvo un archivo .txt que se va a guardar en la misma carpeta
donde esta el video, con el mismo nombre del video y adelante traces. 
Este archivo tiene tantas columnas como localizaciones hechas a mano en el 
image J. 
Cada columna es la intensidad de cada traza. 
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
from skimage import io
from tkinter import Tk, filedialog

root = Tk()
video_file = filedialog.askopenfilename(filetypes=(("", "*.tiff"), ("", "*.tif")))
root.withdraw()
folder = os.path.dirname(video_file)
video_name = os.path.basename(video_file)

#%%
#video_file = 'F://2019//2019-06-26 (Origami Kristina+goos nps, next day)//B+ 5nM 10 Mer//W3_B+5nM10mer_532nm_1mW_100ms_5MHz_0.9us_EM5//W3_B+5nM10mer_532nm_1mW_100ms_5MHz_0.9us_EM5.tif'
locs_two_dots = np.loadtxt(folder+'/'+'locs_two_dots.txt',delimiter='\t',skiprows=1)
#locs_two_dots = np.loadtxt('F://2019//2019-06-26 (Origami Kristina+goos nps, next day)//B+ 5nM 10 Mer//W3_B+5nM10mer_532nm_1mW_100ms_5MHz_0.9us_EM5//locs_twopoints.txt', delimiter=',',skiprows=1)

pixel_size = 0.133 #um
x_pos = locs_two_dots[:,3]/pixel_size
y_pos = locs_two_dots[:,4]/pixel_size
#%%
video = io.imread(video_file)
plt.figure()
plt.imshow(video[0])
roi_size = 3
plt.plot(x_pos,y_pos,'ro', markersize=4)

traces = np.zeros((video.shape[0],x_pos.shape[0]))
for i in range(0,x_pos.shape[0]): #para cada posicion seleccionada
    initial_x = x_pos[i]-roi_size//2
    initial_y = y_pos[i]-roi_size//2
    end_x = x_pos[i]+roi_size//2
    end_y = y_pos[i]+roi_size//2
    for j in range(0,video.shape[0]): #suma la intensidad de los pixeles del roi
        roi_intensity = 0
        for m in range(int(initial_x),int(end_x)+1):#barre en el roi
            for n in range(int(initial_y),int(end_y)+1):
                roi_intensity = roi_intensity+video[j][n,m]
                #print(roi_intensity)
        traces[j,i] = roi_intensity
    print(str(int(i*100/(x_pos.shape[0]-1)))+'%')
    
#%%    
np.savetxt(folder+'\\'+'TRACES_'+video_name[:-4]+'.txt',traces)