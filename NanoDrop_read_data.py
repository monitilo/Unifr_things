# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:32:22 2019

@author: chiarelg

PARA la data del Nanodrop
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 NPs 21.10.19 + concentrated/UV-Vis 10_21_2019 8_01_01 AM.tsv'
a = 1333
skip = 9
N = 50
#c = ['g','r','b','k','g','r','b']
alldata=np.zeros((N,a-3-skip))
allxaxis=np.copy(alldata)

dos = 1
try:
    for i in range(N):
        print(i)
        dos = 2*dos
        spectrum = np.loadtxt(file, delimiter='\t', skiprows = skip+i*a, max_rows = a-skip) # 1334
        alldata[i,:] = spectrum[:,1]
        waveleng = spectrum[:,0]
        plt.plot(spectrum[:,0], spectrum[:,1]) #, c[i])
        plt.xlim((340,500))
    
        
        left = np.where(waveleng==340)[0][0]
        right = np.where(waveleng==500)[0][0]
        
        maxpeak = np.where(spectrum[left:right,1]==np.max(spectrum[left:right,1]))[0][0]
        maxvalue = spectrum[maxpeak+left,1]
    #        plt.ylim((0,maxvalue*1.2))
        if maxvalue < 0.1:
            print("\n BLANK!!!! \n")
            dos = 1
        print( "maxpeak = ", waveleng[maxpeak+left], "\n maxvalue y=", maxvalue, dos)
        print("\n concentration 40 (nM)=", maxvalue * dos /33.6 )
        print("\n concentration 50 (nM)=", maxvalue * dos /53.7 )

except:
     print("el i maximo es", i)
#plt.figure()
#plt.grid()
#plt.plot(spectrum[:,0], spectrum[:,1])
#plt.plot(waveleng, alldata[6,:],'g')
#plt.plot(waveleng, alldata[10,:],'r')
#plt.xlim((340,500))
#for i in range(2):
#    print(i+16)
#    plt.plot(waveleng, alldata[i+16,:])