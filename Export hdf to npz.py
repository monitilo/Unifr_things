# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:38:37 2021

@author: chiarelg
"""

#%% For the foton analysis I need to open hdf5

import numpy as np
import matplotlib.pyplot as plt
import time as time
import os

#folder_path = "C:/Projects/FLAKES/Figuras Intensidad circles"

filename = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_2/Automatic_127 origamis.hdf5'

#DATAFROM = "Test"


names = [filename]  #, filename_inflake2, finelame_outflake2]
#samples = ["Flake2_complete", "flake2_origamis"]  # , "glass3_complete", "glass3_origamis"]

import h5py

tic = time.time()

files = dict()
for n in range(len(names)):
    with h5py.File(names[n], "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
    
        # Get the data
        data_inflake = list(f[a_group_key])
        files[n] = list(f[a_group_key])
    print("time one load = ", time.time()-tic)

#tac = time.time()
#with h5py.File(finelame_outflake, "r") as f:
#    # List all groups
#    print("Keys: %s" % f.keys())
#    a_group_key = list(f.keys())[0]
#
#    # Get the data
#    data_outflake = list(f[a_group_key])

print("time total load = ", time.time()-tic)
#files = [data_inflake,data_outflake]
#files = [data_outflake]

#%% 

#nsamples = (files[l][-1][11]+1)//3
#print("{} different origamis".format(nsamples))
#
#samples = []
#for i in range(1,nsamples+1):
#    samples.append("origami_{}".format(i))


parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]

samples = [1]
finaldata = dict()

for l in range(len(samples)):
    data = files[l]
    tic = time.time()

    alldata = dict()
    alldata[parameters[0]]  = np.zeros([len(data)])
    alldata[parameters[1]]  = np.zeros([len(data)])
    alldata[parameters[2]]  = np.zeros([len(data)])
    alldata[parameters[3]]  = np.zeros([len(data)])
    alldata[parameters[4]]  = np.zeros([len(data)])
    alldata[parameters[5]]  = np.zeros([len(data)])
    alldata[parameters[6]]  = np.zeros([len(data)])
    alldata[parameters[7]]  = np.zeros([len(data)])
    alldata[parameters[8]]  = np.zeros([len(data)])
    alldata[parameters[9]]  = np.zeros([len(data)])
    alldata[parameters[10]] = np.zeros([len(data)])
    alldata[parameters[11]] = np.zeros([len(data)])

    for j in range(len(data)):
    #    frame.append(data[j][0])
        alldata[parameters[0]][j]  = (data[j][0])  # Frames
        alldata[parameters[1]][j]  = (data[j][1])  # x
        alldata[parameters[2]][j]  = (data[j][2])  # y
        alldata[parameters[3]][j]  = (data[j][3])  # photons
        alldata[parameters[4]][j]  = (data[j][4])  # sx
        alldata[parameters[5]][j]  = (data[j][5])  # sy
        alldata[parameters[6]][j]  = (data[j][6])  # bg
        alldata[parameters[7]][j]  = (data[j][7])  # lpx
        alldata[parameters[8]][j]  = (data[j][8])  # lpy
        alldata[parameters[9]][j]  = (data[j][9])  # ellipticity
        alldata[parameters[10]][j] = (data[j][10])  # net_gradient
        alldata[parameters[11]][j] = (data[j][11])  # group


    print( time.time()-tic)

    h1 = plt.hist(alldata["photons"], bins=200, range=(0,3000))




    finaldata[samples[l]] = alldata
plt.show()

#%%

nametosave = names[0][:-5] + ".npz"
with open(nametosave,"w") as f:
    np.savez(nametosave, **alldata)


#%%  To load use:

#nametoload = names[0][:-5] + ".npz"
#with open(nametoload,"r") as f:
#    data = np.load(nametoload)