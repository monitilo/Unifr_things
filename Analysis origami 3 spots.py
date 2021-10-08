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


parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]

names = ['C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_2/Automatic_85 origamis.hdf5']


tic = time.time()
nametoload = names[0][:-5] + ".npz"
with open(nametoload,"r") as f:
    data = np.load(nametoload)  #,allow_pickle=True)

print("time one load = ", time.time()-tic)
hout = plt.hist2d(data["x"],data["y"], bins=500)
plt.colorbar(hout[3])


#%%
pick_list = data["group"]
frame = data["frame"]
photons = data["photons"]
pick_number = np.unique(pick_list)
locs_of_picked = np.zeros(len(pick_number))
photons_of_groups = dict()

tic = time.time()
for i in range(len(pick_number)):
    pick_id = pick_number[i]
    index_picked = np.where(pick_list == pick_id)
    frame_of_picked = frame[index_picked]
    locs_of_picked[i] = len(frame_of_picked)
    photons_of_groups[i] = photons[index_picked]
    
    plt.hist(photons_of_groups[i],alpha=0.5, label="pick {}".format(pick_id))
    plt.legend()
    plt.xlabel("Photons")

print("time goruping the photons = ", time.time()-tic)

#%%
for i in range(3):
    plt.hist(photons_of_groups[i],bins=50, label="pick {}".format(i), range=(0,3000), alpha=0.5)
    plt.legend()
    plt.xlabel("Photons")



#%%
#for l in range(len(samples)):

folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center"

DATAFROM = "Sample22_2lvlBiotin2_automatic"

nsamples = int((data["group"][-1]+1)//3)
print("{} different origamis".format(nsamples))

samples = []
for i in range(1,nsamples+1):
    samples.append("origami_{}".format(i))
subgroups = ["left", "center", "right"]*nsamples

multi3 = ((np.arange(nsamples)+1)*3)-1

Norigami = 0

for i in range(len(subgroups)):
    h1 = plt.hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
                  label=(str(i)+subgroups[i]+" "+str(len(photons_of_groups[i]))+" locs"))
    if i in multi3: 
#            print(i)
        Norigami += 1
        figure_name = '{}_{}'.format(DATAFROM, Norigami)
        plt.title(figure_name)
        plt.legend()
        plt.xlabel("photons")
        plt.gca().get_xticklabels()[-2].set_color('red')
#            plt.xticks()
        figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
#        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#        plt.show()
        plt.close()
print("saved {} origamis in: \n".format(nsamples), folder_path_save)

#%% Filter one: keep only more than 400 locs

good_picks = []

threshold = 400
for i in range(len(photons_of_groups)):
    if len(photons_of_groups[i]) > threshold:
        good_picks.append(i)

 
       
plt.plot(good_picks,'*', label="N={}".format(len(good_picks)//3))
plt.legend()
plt.title("origamis with enough locs")
plt.show()
print(len(good_picks)//3)

#%% Filter two, the sides have to be simetric in locs , ± 20 %

delta = 0.5
simetric_picks = []
multiples3 = (np.arange(nsamples))*3
for i in multiples3:
    if i in good_picks and i+1 in good_picks:
        print(i,"i")
        if len(photons_of_groups[i]) < len(photons_of_groups[i+2])*(1+delta) and len(photons_of_groups[i]) > len(photons_of_groups[i+2])*(1-delta):
            simetric_picks.append(i)
            simetric_picks.append(i+1)
            simetric_picks.append(i+2)

plt.plot(simetric_picks,'*', label="N={}".format(len(simetric_picks)//3))
plt.legend()
plt.title("origamis with simetric laterals")
plt.show()
print(len(simetric_picks)//3)
# =============================================================================
#%%

#for l in range(len(samples)):

folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center"

DATAFROM = "Sample22_2lvlBiotin2_Filtered_{}_{}%".format(threshold, int(delta*100))
#
#nsamples = int((data["group"][-1]+1)//3)
#print("{} different origamis".format(nsamples))
#
#samples = []
#for i in range(1,nsamples+1):
#    samples.append("origami_{}".format(i))
#subgroups = ["left", "center", "right"]*nsamples

multi3 = ((np.arange(nsamples)+1)*3)-1
good_origamis = []
for i in range(len(photons_of_groups)):
    if i in simetric_picks:
        h1 = plt.hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
                      label=(str(i)+subgroups[i]+" "+str(len(photons_of_groups[i]))+" locs"))
        if i in multi3: 
    #            print(i)
            Norigami = (i+1)//3
            figure_name = '{}_{}'.format(DATAFROM, Norigami)
            plt.title(figure_name)
            plt.legend()
            plt.xlabel("photons")
            plt.gca().get_xticklabels()[-2].set_color('red')
    #            plt.xticks()
            figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.show()
            plt.close()
            good_origamis.append(i)
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)


#%% Now do analysis of the good ones only

mean_values = np.zeros(len(photons_of_groups))
good_origamis = []
for i in range(len(photons_of_groups)):
    if i in simetric_picks:
        h1 = plt.hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
                      label=(str(i)+subgroups[i]+" "+str(len(photons_of_groups[i]))+" locs"))
        mean_values[i] = np.mean(photons_of_groups[i])
        
        if i in multi3:
            Norigami = (i+1)//3
#            figure_name = '{}_{}'.format(DATAFROM, Norigami)
#            plt.title(figure_name)
#            plt.legend()
#            plt.xlabel("photons")
#            plt.gca().get_xticklabels()[-2].set_color('red')
#    #            plt.xticks()
##            figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
##            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.show()
            plt.close()
            good_origamis.append(i)

plt.plot(simetric_picks, mean_values[simetric_picks],'*-')
plt.show()

plt.hist(mean_values[good_picks], density=True,alpha=0.5)
plt.hist(mean_values[good_origamis], density=True, color="red",alpha=0.5)
plt.show()
#%%





