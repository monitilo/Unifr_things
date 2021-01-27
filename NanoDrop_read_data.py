# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:32:22 2019

@author: chiarelg

PARA la data del Nanodrop
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
#file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 NP y Antennas/UV-Vis 1_15_2020 6_46_40 AM german antennas.tsv'
#file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 Gold NPs/UV-Vis 1_15_2020 8_29_14 AM gold antennas german.tsv'

file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 Nicole/UV-Vis 1_27_2021 1_43_01 PM.tsv'

a = 1333  # cantidad de puntos.
skip = 10  # header y texto entre espectros.
N = 5  # Cantidad de espectros esperados
#c = ['g','r','b','k','g','r','b']
alldata=np.zeros((N,a-2-skip))
allxaxis=np.copy(alldata)

#labels = ["BLank", "NP recovered/10", "Dimer", "Monomer", "NP diluted 1:10"]
labels = ["60", "40", "Si", "Blank", "Si"]

dos = 1

datatosave = []
#try:
for i in range(N):
    print(i)
    dos = 2*dos
    spectrum = np.loadtxt(file, delimiter='\t', skiprows = skip+i*a, max_rows = a-skip) # 1334
    print(spectrum)
    datatosave.append(np.array(spectrum[:,1]))
    alldata[i,:] = spectrum[:,1]
    waveleng = spectrum[:,0]
    plt.figure(file)
    if i == 4:
        plt.plot(spectrum[:,0], spectrum[:,1], label=labels[i]) #, c[i])
    else:
        plt.plot(spectrum[:,0], spectrum[:,1], label=labels[i]) #, c[i])
#    plt.xlim((300,700))
#    plt.ylim((-0.1, 2))

    left = np.where(waveleng==340)[0][0]
    right = np.where(waveleng==600)[0][0]
    
#    maxpeak = np.where(spectrum[left:right,1]==np.max(spectrum[left:right,1]))[0][0]
#    maxvalue = spectrum[maxpeak+left,1]
#        plt.ylim((0,maxvalue*1.2))
    plt.legend()

#    if maxvalue < 0.1:
#        print("\n BLANK!!!! \n")
#        dos = 1

#    print( "maxpeak = ", waveleng[maxpeak+left], "\n maxvalue y=", maxvalue, dos)
#    print("\n concentration 40 (nM)=", maxvalue /3.36 )
#    print("\n concentration 50 (nM)=", maxvalue /1.935 )
#except:
#     print("el i maximo es", i)

#plt.figure()
#plt.grid()
#plt.plot(spectrum[:,0], spectrum[:,1])
#plt.plot(waveleng, alldata[6,:],'g')
#plt.plot(waveleng, alldata[10,:],'r')
#plt.xlim((340,500))
#for i in range(2):
#    print(i+9)
#    plt.plot(waveleng, alldata[i+16,:])


datatosave_name = "27.01_spectros.txt"
datatosave_waveleng = "27.01_Waveleng_spectros1.txt"
np.savetxt(datatosave_waveleng, np.array(waveleng).T, delimiter="    ", newline='\r\n')
np.savetxt(datatosave_name, np.array(datatosave).T, delimiter="    ", newline='\r\n')
