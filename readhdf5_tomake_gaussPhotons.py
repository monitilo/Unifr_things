# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:38:37 2021

@author: chiarelg
"""

#%% For the foton analysis I need to open hdf5

import numpy as np
import matplotlib.pyplot as plt
import time as time

filename_inflake = 'C:/Origami testing Widefield/2021-06-10_MoS2_samples_456_BSA_test/4_100ms_130nmpix_mode1/DNA_PAINT_1mW_9merAtto4881nM_trolox_glox_in_150ul_1xTAE12_2/DNA_PAINT_1mW_picked_IN flake2.hdf5'
finelame_outflake = 'C:/Origami testing Widefield/2021-06-10_MoS2_samples_456_BSA_test/4_100ms_130nmpix_mode1/DNA_PAINT_1mW_9merAtto4881nM_trolox_glox_in_150ul_1xTAE12_2/DNA_PAINT_1mW_picked_out of flake2.hdf5'

import h5py
#filename = "file.hdf5"
tic = time.time()
with h5py.File(filename_inflake, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data_inflake = list(f[a_group_key])

print( time.time()-tic)
tac = time.time()
with h5py.File(finelame_outflake, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data_outflake = list(f[a_group_key])

print( time.time()-tac)
files = [data_inflake,data_outflake]

#%% 

parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]

samples = ["in flake", "out flake"]

finaldata = dict()

for l in range(len(samples)):
    data = files[l]
    tic = time.time()
    
    alldata = dict()
    alldata[parameters[0]] = []
    alldata[parameters[1]] = []
    alldata[parameters[2]] = []
    alldata[parameters[3]] = []
    alldata[parameters[4]] = []
    alldata[parameters[5]] = []
    alldata[parameters[6]] = []
    alldata[parameters[7]] = []
    alldata[parameters[8]] = []
    alldata[parameters[9]] = []
    alldata[parameters[10]] = []
    alldata[parameters[11]] = []
    
    for j in range(len(data)):
    #    frame.append(data[j][0])
        alldata[parameters[0]].append(data[j][0])  # Frames
        alldata[parameters[1]].append(data[j][1])  # x
        alldata[parameters[2]].append(data[j][2])  # y
        alldata[parameters[3]].append(data[j][3])  # photons
        alldata[parameters[4]].append(data[j][4])  # sx
        alldata[parameters[5]].append(data[j][5])  # sy
        alldata[parameters[6]].append(data[j][6])  # bg
        alldata[parameters[7]].append(data[j][7])  # lpx
        alldata[parameters[8]].append(data[j][8])  # lpy
        alldata[parameters[9]].append(data[j][9])  # ellipticity
        alldata[parameters[10]].append(data[j][10])  # net_gradient
        alldata[parameters[11]].append(data[j][11])  # group


    print( time.time()-tic)

    h1 = plt.hist(alldata["photons"], bins=200, range=(0,3000))

    finaldata[samples[l]] = alldata


#%%

bines = 50
hin = plt.hist2d(finaldata[samples[0]]['x'],finaldata[samples[0]]["y"], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
plt.colorbar(hin[3])
plt.show()

#hout = plt.hist2d(finaldata[samples[1]]["x"],finaldata[samples[1]]["y"], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
#plt.colorbar(hout[3])
#plt.show()
#h2 = plt.hist2d(x,y, bins=len(x)//3000, range=([0,154], [0,154]))
#plt.xlim([40,60])
#plt.ylim([100,120])

##%% center gauss of 5.3 x 4.2 um
"""  ((x-xc)/a)**2 + ((y-yc)/b)**2 = r**2

xc = yc = 77 pix  (center of gaussian laser)
2*a = 5.3 um = 5300/130 = 40.769230 pix (sigma x laser)
2*b = 4.2 um = 4200/130 = 32.307692 pix (sigma y laser)

if((((A-77)/20)^2+((B-77)/16)^2)<=1, C)
if(1<(((A-75)/20)^2+((B-75)/16)^2)&&(((A-75)/20)^2+((B-75)/16)^2)<=4, C)

"""

xc = 77
yc = 77
pixsize = 130
sigmax_laser = 5300
sigmay_laser = 4200
a = 0.5*(int(sigmax_laser/pixsize))
b = 0.5*(int(sigmay_laser/pixsize))

s=0

x = np.array(finaldata[samples[s]]["x"])
y = np.array(finaldata[samples[s]]["y"])

tic = time.time()

x1circle = []
y1circle = []
phot1circle = []

for i in range(len(x)):

    if (((x[i]-xc)/a)**2 + ((y[i]-yc)/b)**2) <= 1:
        x1circle.append(x[i])
        y1circle.append(y[i])
        phot1circle.append(finaldata[samples[s]]["photons"][i])

print( time.time()-tic)

hist2d = plt.hist2d(x1circle, y1circle, bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
plt.colorbar(hist2d[3])
plt.show()


tic = time.time()

x2circle = []
y2circle = []
phot2circle = []
for i in range(len(x)):

    if 1 < (((x[i]-xc)/a)**2 + ((y[i]-yc)/b)**2) <= 4:
        x2circle.append(x[i])
        y2circle.append(y[i])
        phot2circle.append(finaldata[samples[s]]["photons"][i])

print( time.time()-tic)

hist2d = plt.hist2d(x2circle, y2circle, bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
plt.colorbar(hist2d[3])
plt.show()

h1 = plt.hist(phot1circle, bins=60, alpha=0.5, range=(0,3500))
h2 = plt.hist(phot2circle, bins=180, alpha=0.5, range=(0,3500))


#tic = time.time()
#
#xnew = []
#for i in range(len(x)):
#    if (x[i]-xc/a)**2 <=1:
#        xnew[i].append(x[i])
#
#ynew = []
#for i in range(len(y)):
#    if (y[i]-yc/b)**2 <=1:
#        ynew[i].append(y[i])
#
#print( time.time()-tic)

#%%

