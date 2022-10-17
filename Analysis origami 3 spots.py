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

SAVE_FIGS = False

folder_path_save = "C:\Projects\Super resolution\Photons quenching two color super res\Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1__1\Histograms/New"
#folder_path_save = 'C:/Projects/FLAKES/Origami photons laterals vs center/Sample 22_ 2lvlBio _2 INFLAKE'


parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]

#file = 'C:/Origami testing Widefield/2021-05-11 3spots CFret/Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1__1/36 origamis info.hdf5'

#file = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_2/Automatic_OUT flake 72 origamis.hdf5'

from tkinter import Tk, filedialog

root = Tk()
nametoload = filedialog.askopenfilename(filetypes=(("", "*.npz"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(nametoload)
only_name = os.path.basename(nametoload)

#tic = time.time()
#nametoload = file[:-5] + ".npz"
#with open(nametoload,"r") as f:
#    data = np.load(nametoload)  #,allow_pickle=True)
data = np.load(nametoload)
#print("time one load = ", time.time()-tic)
hout = plt.hist2d(data["x"],data["y"], bins=500)
plt.colorbar(hout[3])


#%%
pick_list = data["group"]
frame = data["frame"]
photons = data["photons"]
pick_number = np.unique(pick_list)
locs_of_picked = np.zeros(len(pick_number))
photons_of_groups = dict()
frame_of_groups = dict()
print("amount of frames =", len(np.unique(frame)))
tic = time.time()
for i in range(len(pick_number)):
    pick_id = pick_number[i]
    index_picked = np.where(pick_list == pick_id)
    frame_of_picked = frame[index_picked]
    locs_of_picked[i] = len(frame_of_picked)
    photons_of_groups[i] = photons[index_picked]
    frame_of_groups[i] = frame[index_picked]

i=0
plt.hist(photons_of_groups[i],alpha=0.5, label="pick {}".format(pick_id))
plt.legend()
plt.xlabel("Photons")

print("time goruping the photons = ", time.time()-tic)

#%%

#for i in range(3):
#    plt.hist(frame_of_groups[i],bins=50, label="pick {}".format(i), range=(0,15000), alpha=0.5)
#    plt.legend()
#    plt.xlabel("Photons")



#%%
#for l in range(len(samples)):


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
    h1 = plt.hist(photons_of_groups[i], bins=30, density=True, alpha=0.5,
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

#%% Filter one: keep only more than "threshold" locs

good_picks = []

threshold = 1 
photon_treshold = 5500  #np.max(photons_of_groups[0])
for i in range(len(photons_of_groups)):
    if len(np.where(photons_of_groups[i]<photon_treshold)[0]) > threshold:
        good_picks.append(i)

 
       
plt.plot(good_picks,'*', label="N={}".format(len(good_picks)//3))
plt.legend()
plt.title("origamis with enough locs")
plt.show()
print(len(good_picks)//3)

#%% Filter two, the sides have to be simetric in locs , ± 50 % for example

delta = 100
simetric_picks = []
multiples3 = (np.arange(nsamples))*3
for i in multiples3:
    if i in good_picks and i+1 in good_picks and i+2 in good_picks:
#        print(i,"i")
        if len(photons_of_groups[i]) < len(photons_of_groups[i+2])*(1+delta) and len(photons_of_groups[i]) > len(photons_of_groups[i+2])*(1-delta):
            simetric_picks.append(i)
            simetric_picks.append(i+1)
            simetric_picks.append(i+2)

plt.plot(simetric_picks,'*', label="N={}".format(len(simetric_picks)//3))
plt.legend()
plt.title("origamis with simetric laterals")
plt.show()
plt.close()
print(len(simetric_picks)//3)
# =============================================================================

##%%
#np.mean(photons_of_groups[3])
#np.mean(photons_of_groups[3][np.where(photons_of_groups[3]<1500)])


#%% RAW images

#for l in range(len(samples)):



#DATAFROM = "OutFlake_Filtered_{}locs_{}%delta".format(threshold, int(delta*100))
#DATAFROM = "OUTFlake_RAW"
for density in [True, False]:

    
    frame_limit = 15000 # 15k tipically is the max
    
    if frame_limit < 15000:
        DATAFROM = "Raw_Epi FRETx4 only first {} frames".format(frame_limit)
        framebool = True
    else:
        DATAFROM = "Raw_Epi FRETx4 density={}".format(density)
        framebool = False
    
    
    
#    folder_path_save = "C:\Projects\Super resolution\Photons quenching two color super res\Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1__1\Histograms"
    
    #
    #nsamples = int((data["group"][-1]+1)//3)
    #print("{} different origamis".format(nsamples))
    #
    #samples = []
    #for i in range(1,nsamples+1):
    #    samples.append("origami_{}".format(i))
    #subgroups = ["left", "center", "right"]*nsamples
    
    #mean_values = np.zeros(len(photons_of_groups))
    multi3 = ((np.arange(nsamples)+1)*3)-1
    good_origamis = []
    
    #photon_treshold = 6000
    for i in range(len(photons_of_groups)):
    
        if i in simetric_picks:
    
            if framebool ==  True:
                photons_to_plot = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
            else:
                photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 
    
            h1 = plt.hist(photons_to_plot, bins=50,
                          range=(0,photon_treshold), density=density, alpha=0.5,edgecolor='black', linewidth=0.1,
                          label=(str(i)+subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))
    
    
    #        mean_values[i] = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)])
            if i in multi3: 
        #            print(i)
                Norigami = (i+1)//3
                figure_name = '{}_{}'.format(Norigami, DATAFROM)
                plt.title(figure_name)
                plt.legend()
                plt.xlabel("photons")
                plt.gca().get_xticklabels()[-2].set_color('red')
        #            plt.xticks()
                figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
#                plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
#                plt.show()
                
                good_origamis.append(Norigami)
            plt.close()
                
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)

#%%

#interesting_picks = [81,84,87]  # only the left
interesting_picks = [12]

density = False

color = dict()
color["left"] = "blue"
color["center"] = "orange"
color["right"] = "green"

frame_limit = 15000 # 15k tipically is the max

if frame_limit < 15000:
    DATAFROM = "Separate sidesaa FRETx4 only first {} frames".format(frame_limit)
    framebool = True
else:
    DATAFROM = "Separate sidesaa FRETx4 "
    framebool = False

#folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/FRETx4/Labteck 05-11-21/Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1"

for i in interesting_picks:
    aux = i
    for j in range(3):
        i = aux+j
        
        if framebool == True:
            photons_to_plot = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
        else:
            photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 

        if subgroups[i] == "left":  # "right":
                    h1 = plt.hist(photons_to_plot, bins=50,
                      range=(0,photon_treshold), density=density, alpha=0.8,
                      label=(str(i)+subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])
        else:
            h1 = plt.hist(photons_to_plot, bins=50,
                          range=(0,photon_treshold), density=density, alpha=0.2,
                          label=(str(i)+subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])

    Norigami = (i+1)//3
    figure_name = '{}_{}_{}'.format(Norigami, DATAFROM,subgroups[i])
    plt.title(figure_name)
    plt.legend()
    plt.xlabel("photons")
    plt.gca().get_xticklabels()[-2].set_color('red')
#    figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
#    plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
    plt.show()
    plt.close()
    good_origamis.append(Norigami)
                
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)

#%% Image for paper


#interesting_picks = [81,84,87]  # only the left
interesting_picks = [27] # 15,27, 39?, 78?, 93?, 102, 108, 120!

photon_treshold = 5500

density = False

SAVE_FIGS = True
SAVE_FIGS = False

bines = 50

color = dict()
color["left"] = "blue"
color["center"] = "orange"
color["right"] = "green"


maxnumber = 1500

photons_to_dict = dict()

frame_limit = 15000 # 15k tipically is the max

if frame_limit < 15000:
    DATAFROM = " FRETx4 only first {} frames".format(frame_limit)
    framebool = True
else:
    DATAFROM = " FRETx4 "
    framebool = False

folder_path_save = "C:\Projects\Super resolution\Photons quenching two color super res\Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1__1\Histograms/New"

for i in interesting_picks:
    aux = i
    for j in range(3):
        i = aux+j
        
        if framebool == True:
            photons_to_plot = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
        else:
            photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 

        if subgroups[i] == "left":  # "right":

                    h1 = plt.hist(photons_to_plot, bins=bines, 
                      range=(0,photon_treshold), density=density, alpha=1,
                      label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])
        elif subgroups[i] == "right":
            h1 = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density, alpha=1,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])
        else:
            h1 = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density, alpha=1,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])

#        np.savetxt("ori27_first5k_{}_part.txt".format(subgroups[i]), photons_to_plot)


    Norigami = (i+1)//3
    figure_name = '{}_{}'.format(Norigami, DATAFROM)
#    plt.title(figure_name)
    plt.legend(loc="upper right")
    plt.xlabel("photons")
    plt.ylabel("# Events")
#    plt.ylim(0, 120)
#    plt.gca().get_xticklabels()[-2].set_color('red')
    figure_path = os.path.join(folder_path_save, '%s.pdf' % figure_name)
    if SAVE_FIGS == True:
        plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
#    plt.show()
#        plt.close()
    good_origamis.append(Norigami)
    plt.savefig('AAAAAorigami{}_all frames.pdf'.format(interesting_picks[0])) 
    plt.show()

                
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)

frame_limit = 5000 # 15k tipically is the max

if frame_limit < 15000:
    DATAFROM = "V2 free FRETx4 only first {} frames".format(frame_limit)
    framebool = True
else:
    DATAFROM = " free FRETx4 "
    framebool = False


for i in interesting_picks:
    aux = i
    for j in range(3):
        i = aux+j
        
        if framebool == True:
            photons_to_plot = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
        else:
            photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 

        if subgroups[i] == "left":  # "right":
                    h1 = plt.hist(photons_to_plot, bins=bines,
                      range=(0,photon_treshold), density=density, alpha=1,
                      label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])
        elif subgroups[i] == "right":
            h1 = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density, alpha=1,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])
        else:
            h1 = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density, alpha=1,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])        
#        np.savetxt("ori27_full_{}_part.txt".format(subgroups[i]), photons_to_plot)

        
    Norigami = (i+1)//3
    figure_name = '{}_{}'.format(Norigami, DATAFROM)
#    plt.title(figure_name)
    plt.legend()
    plt.xlabel("photons")
    plt.ylabel("# Events")
#    plt.ylim(0, 120)
#    plt.gca().get_xticklabels()[-2].set_color('red')
    figure_path = os.path.join(folder_path_save, '%s.pdf' % figure_name)
    if SAVE_FIGS == True:
        plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
#    plt.show()
#        plt.close()
    good_origamis.append(Norigami)
    plt.savefig('AAAorigami{}_first 5k frames.pdf'.format(interesting_picks[0])) 
    
print("saved {} origamis in: \n".format(len(interesting_picks)), folder_path_save)



#%%
#%% Image for paper V2


#interesting_picks = [81,84,87]  # only the left
interesting_picks = [27]

photon_treshold = 5500

density = False

stacked=True

bines = 35

color = dict()
color["left"] = "blue"
color["center"] = "orange"
color["right"] = "green"

frame_limit = 15000 # 15k tipically is the max

if frame_limit < 15000:
    DATAFROM = " free FRETx4 only first {} frames".format(frame_limit)
    framebool = True
else:
    DATAFROM = " free FRETx4 "
    framebool = False

folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/FRETx4/Labteck 05-11-21/Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1"

frame_limit = 5000

for i in interesting_picks:
    aux = i
    for j in range(3):
        i = aux+j
        
        photons_to_plot_start = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
        photons_to_plot_after = photons_of_groups[i][np.where(frame_of_groups[i]>frame_limit)]  # 
        photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 

        if subgroups[i] == "left":  # "right":
                h1 = plt.hist(photons_to_plot_start, bins=bines, 
                      range=(0,photon_treshold), density=density, alpha=0.8, stacked=True,
                      label=(subgroups[i]+"; "+str(len(photons_to_plot_start))+" locs"))  #, color = color[subgroups[i]])
#
                h1 = plt.hist(photons_to_plot_after, bins=bines, 
                      range=(0,photon_treshold), density=density, alpha=0.8, stacked=True, color="red",
                      label=(subgroups[i]+"; "+str(len(photons_to_plot_after))+" locs"))  #, color = color[subgroups[i]])
        else:
                h1 = plt.hist(photons_to_plot, bins=bines,
                              range=(0,photon_treshold), density=density, alpha=0.2,stacked=True,
                              label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])

    Norigami = (i+1)//3
    figure_name = '{}_{}'.format(Norigami, DATAFROM)
    plt.title(figure_name)
    plt.legend(loc="upper right")
    plt.xlabel("photons")
    plt.ylabel("# Events")
#    plt.ylim(0, 120)
#    plt.gca().get_xticklabels()[-2].set_color('red')
    figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
#    plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
    plt.show()
#        plt.close()
    good_origamis.append(Norigami)
                
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)
#%%
density = False
i = 27
photons_to_plot_start = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
photons_to_plot_after = photons_of_groups[i][np.where(frame_of_groups[i]>frame_limit)]  # 
photons_to_plot = photons_of_groups[i+1][np.where(photons_of_groups[i+1]<photon_treshold)]  # 
photons_to_plot2 = photons_of_groups[i+2][np.where(photons_of_groups[i+2]<photon_treshold)]  # 
#
#h0 = plt.hist([photons_to_plot_start,photons_to_plot_after], bins=bines, color=["cornflowerblue","tab:blue"],  #cornflowerblue, tab:blue, blue
#                      range=(0,photon_treshold), density=density, alpha=0.5, stacked=True,edgecolor='black', linewidth=0.5,
#                      label=(subgroups[i]+"; "+str(len(photons_to_plot_start))+" locs"))  #, color = color[subgroups[i]])
#plt.figure()
h1 = plt.hist(photons_to_plot_start, bins=bines,
                              range=(0,photon_treshold), density=density, alpha=0.5,stacked=True, color="cornflowerblue",edgecolor='black', linewidth=0.5,
                              label=(subgroups[i]+"; "+str(len(photons_to_plot_start))+" locs"))
h2 = plt.hist(photons_to_plot_after, bins=bines,
                              range=(0,photon_treshold), density=density, alpha=0.5,stacked=True, color="blue",edgecolor='black', linewidth=0.5,
                              label=(subgroups[i]+"; "+str(len(photons_to_plot_after))+" locs"))
h3 = plt.hist(photons_to_plot, bins=bines,
                              range=(0,photon_treshold), density=density, alpha=0.2,stacked=False, color="green",edgecolor='black', linewidth=0,
                              label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))
h4 = plt.hist(photons_to_plot2, bins=bines,
                              range=(0,photon_treshold), density=density, alpha=0.2,stacked=False, color="red",edgecolor='black', linewidth=0,
                              label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))
#%% Now do analysis of the good ones only
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # # # # # # # 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


#%%

SAVE_FIGS = True
SAVE_FIGS = False  



DATAFROM = "Epi_FRETx4_{}locs_{}%delta".format(threshold, int(delta*100))


folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/FRETx4"


left_picks = np.arange(0,len(photons_of_groups),3)
#photon_treshold = 4000
mean_values = np.zeros(len(photons_of_groups))
counting = 0

mean_shift = 1.5
delta2 = 10.2


for i in left_picks:  # in multiples3:
    if i in simetric_picks[:-2]:
        left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)])
        right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<photon_treshold)])
        if left_mean > right_mean*(1-delta2) and left_mean < right_mean*(1+delta2):
            counting += 1
#            print("i",i)
            new_photon_treshold = (np.max([left_mean,right_mean])) * mean_shift
            center_mean = np.mean(photons_of_groups[i+1][np.where(photons_of_groups[i+1]<new_photon_treshold)])
            new_left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<new_photon_treshold)])
            new_right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<new_photon_treshold)])

#            fig, axs = plt.subplots(2)
            
            
#            for j in range(1,2):
            j=0
            plt.hist(photons_of_groups[i+j], bins=100, density=True, alpha=0.2, range=(0,photon_treshold),
              label=(str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs"))
            j=1
            plt.hist(photons_of_groups[i+j], bins=100, density=True, alpha=0.7, range=(0,photon_treshold),
              label=(str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs"))
            j=2
            plt.hist(photons_of_groups[i+j], bins=100, density=True, alpha=0.2, range=(0,photon_treshold),
              label=(str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs"))

            plt.axvline(new_left_mean, color="blue", label=("left = {:.0f}".format(new_left_mean)))
            plt.axvline(center_mean, color="orange", label=( "center {:.0f}".format(center_mean)), alpha=0.5)
            plt.axvline(new_right_mean, color="green", label=( "right {:.0f}".format(new_right_mean)))
                
            plt.legend()
    #            axs[0].hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
    #              label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i]))+" locs"))
    #            axs[0].hist(photons_of_groups[i+1], bins=30, range=(0,3000), density=True, alpha=0.5,
    #              label=(str(i+1)+subgroups[i+1]+"; "+str(len(photons_of_groups[i+1]))+" locs"))
    #            axs[0].hist(photons_of_groups[i+2], bins=30, range=(0,3000), density=True, alpha=0.5,
    #              label=(str(i+2)+subgroups[i+2]+"; "+str(len(photons_of_groups[i+2]))+" locs"))
    #            axs[0].legend()
    
#            axs[1].set_ylabel("photon mean")
#            axs[1].plot(["left","center","right"],[new_left_mean, center_mean, new_right_mean],'*-',
#                     label= "center_photons_treshold= " +str(int(new_photon_treshold)))
    
    #        print(i,"i")
            Norigami = ((i+1)//3)+1
            figure_name = '{}_{}_MeanPhotons_{}cutted'.format(Norigami, DATAFROM, photon_treshold)
            plt.title(figure_name)
#            fig.suptitle(figure_name)
#            axs[1].legend()
#            plt.ylabel("mean photons")
    
            figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
            if SAVE_FIGS == True:
                plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.show()
            plt.close()

print("saved {} origamis out of {} \nin: ".format(counting, nsamples), folder_path_save)
print(nametoload)

#%% Exporting histograms for ilustrator


#interesting_picks = [81,84,87]  # only the left
interesting_picks = [120] # 15,27, 39?, 78?, 93?, 102, 108, 120!

photon_treshold = 4500

density = False

SAVE_FIGS = True
#SAVE_FIGS = False

bines = 18

blue = (79/255,134/255,198/255)
green = (104/255,153/255,93/255) 
yellow = (222/255,157/255,38/255)

darker_blue = (51/255,102/255,153/255)
darker_green = (104/255,153/255,93/255) 
darker_yellow = (222/255,157/255,38/255)

color = dict()
color["left"] = yellow # "#DE9D26"
color["center"] =  blue # '#4F86C6'
color["right"] =  green #"#68995D"

edgecolor = dict()
edgecolor["left"] = "#996633"  #"orange" #"#DE9D26"
edgecolor["center"] = "#336699"  # "blue" #  '#4F86C6'
edgecolor["right"] =  "#336633"  # "#344D2E" #"black" # "#68995D"


photons_to_dict = dict()

norm = "norm"
DATAFROM = " FRETx4 "
framebool = False

density = None
alfa = dict()
alfas = [0.9, 0.5, 0.5]  # left center rigth
for i in range(len(alfas)):
    alfa[subgroups[i]] = alfas[i]
    
folder_path_save = "C:\Projects\Super resolution\Photons quenching two color super res\Ch2_PAINTepi_532nm_13mW(40F4)_spli2channels_100ms_1x1__1\Histograms/New"


frame_limit = 5000
DATAFROM = "Presentation FRETx4 all frames"
framebool = False

for i in interesting_picks:
    aux = i
    for j in range(3):
        i = aux+j
        
        if framebool == True:
            photons_to_plot = photons_of_groups[i][np.where(frame_of_groups[i]> frame_limit)]
        else:
            photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 

        if subgroups[i] == "left":  # "right":

                    hleft = plt.hist(photons_to_plot, bins=bines, 
                      range=(0,photon_treshold), density=density,
                      label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"),color=color[subgroups[i]],width=10,edgecolor=edgecolor[subgroups[i]],linewidth=1)  #, color = color[subgroups[i]])

        elif subgroups[i] == "right":
            hright = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"),color=color[subgroups[i]],width=10,edgecolor=edgecolor[subgroups[i]],linewidth=1)  #, color = color[subgroups[i]])



        else:
            hcenter = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"),color=color[subgroups[i]],width=10,edgecolor=edgecolor[subgroups[i]],linewidth=1)  #, color = color[subgroups[i]])

#        np.savetxt("ori27_first5k_{}_part.txt".format(subgroups[i]), photons_to_plot)
    plt.close()

    Norigami = (i+1)//3
# =============================================================================
#     figure_name = '{}_{}'.format(Norigami, DATAFROM)
# #    plt.title(figure_name)
#     plt.legend(loc="upper right")
#     plt.xlabel("Photons")
#     plt.ylabel("# Events")
# #    plt.ylim(0, 120)
# #    plt.gca().get_xticklabels()[-2].set_color('red')
#     figure_path = os.path.join(folder_path_save, '%s.pdf' % figure_name)
#     if SAVE_FIGS == True:
#         plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
#     plt.show()
# #        plt.close()
#     good_origamis.append(Norigami)
# #    plt.savefig('AAAAAorigami{}_all frames.pdf'.format(interesting_picks[0])) 
#     plt.show()
# =============================================================================
    
    figure_name = 'presentation {}_{}_NORM'.format(Norigami, DATAFROM)
    
    i=aux
    i+=1
    stepright = abs(hright[1][0]-hright[1][1] )/2
    plt.bar(hright[1][:-1]+stepright,hright[0]/max(hright[0]),width=stepright*2, alpha=alfa[subgroups[i]],color=color[subgroups[i]],edgecolor=edgecolor[subgroups[i]])
    i+=1
    stepcenter = abs(hcenter[1][0]-hcenter[1][1] )/2
    plt.bar(hcenter[1][:-1]+stepcenter,hcenter[0]/max(hcenter[0]),width=stepcenter*2, alpha=alfa[subgroups[i]],color=color[subgroups[i]],edgecolor=edgecolor[subgroups[i]])
    i=aux
    stepleft = abs(hleft[1][0]-hleft[1][1] )/2
    plt.bar(hleft[1][:-1]+stepleft,hleft[0]/max(hleft[0]),width=stepleft*2, alpha=alfa[subgroups[i]],color=color[subgroups[i]],edgecolor=edgecolor[subgroups[i]])
    
    
    # plt.legend(loc="upper right")
    plt.xlabel("Photons")
    plt.ylabel("Normalized # Events")
    plt.title("full video")
    figure_path = os.path.join(folder_path_save, '%s.pdf' % figure_name)
    if SAVE_FIGS == True:
        plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')

    plt.show()

                
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)

frame_limit = 5000
DATAFROM = "Presentation FRETx4 only first {} frames".format(frame_limit)
framebool = True


for i in interesting_picks:
    aux = i
    for j in range(3):
        i = aux+j
        
        if framebool == True:
            photons_to_plot = photons_of_groups[i][np.where(frame_of_groups[i]< frame_limit)]
        else:
            photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)]  # 

        if subgroups[i] == "left":  # "right":

                    hleft = plt.hist(photons_to_plot, bins=bines, 
                      range=(0,photon_treshold), density=density,
                      label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])

        elif subgroups[i] == "right":
            hright = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])



        else:
            hcenter = plt.hist(photons_to_plot, bins=bines,
                          range=(0,photon_treshold), density=density,
                          label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"))  #, color = color[subgroups[i]])

        plt.close()
    Norigami = (i+1)//3
# =============================================================================
#     figure_name = '{}_{}'.format(Norigami, DATAFROM)
# #    plt.title(figure_name)
#     plt.legend()
#     plt.xlabel("photons")
#     plt.ylabel("# Events")
# #    plt.ylim(0, 120)
# #    plt.gca().get_xticklabels()[-2].set_color('red')
#     figure_path = os.path.join(folder_path_save, '%s.pdf' % figure_name)
#     if SAVE_FIGS == True:
#         plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
#     plt.show()
# =============================================================================

    figure_name = '{}_{}_NORM'.format(Norigami, DATAFROM)

    i=aux
    i+=1
    stepright = abs(hright[1][0]-hright[1][1] )/2
    plt.bar(hright[1][:-1]+stepright,hright[0]/max(hright[0]),width=stepright*2, alpha=alfa[subgroups[i]],color=color[subgroups[i]],edgecolor=edgecolor[subgroups[i]])
    i+=1
    stepcenter = abs(hcenter[1][0]-hcenter[1][1] )/2
    plt.bar(hcenter[1][:-1]+stepcenter,hcenter[0]/max(hcenter[0]),width=stepcenter*2, alpha=alfa[subgroups[i]],color=color[subgroups[i]],edgecolor=edgecolor[subgroups[i]])
    i=aux
    stepleft = abs(hleft[1][0]-hleft[1][1] )/2
    plt.bar(hleft[1][:-1]+stepleft,hleft[0]/max(hleft[0]),width=stepleft*2, alpha=alfa[subgroups[i]],color=color[subgroups[i]],edgecolor=edgecolor[subgroups[i]])
    

    # plt.legend()
    plt.title("first 5 k frames #{} ".format(Norigami))
    plt.xlabel("Photons")
    plt.ylabel("Normalized # Events")
    figure_path = os.path.join(folder_path_save, '%s.pdf' % figure_name)
    if SAVE_FIGS == True:
        plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')

    plt.show()
#        plt.close()
    good_origamis.append(Norigami)
#    plt.savefig('AAAorigami{}_first 5k frames.pdf'.format(interesting_picks[0])) 
    
print("saved {} origamis in: \n".format(len(interesting_picks)), folder_path_save)
#%% Try to make the events from the locs

#photons_of_groups[]

#frame_of_groups[]

interesting_picks = [27,28,29]
binding_photons = dict()

for i in interesting_picks:
    binding_photons[i] = [photons_of_groups[i][0]]
    
    for j in range(len(frame_of_groups[i])-1):
        if frame_of_groups[i][j+1] < frame_of_groups[i][j]+3:
            binding_photons[i][-1] = binding_photons[i][-1] + photons_of_groups[i][j]
        else:
            binding_photons[i].append(photons_of_groups[i][j])

    print("raw", np.mean(photons_of_groups[i]),"grouped", np.mean(binding_photons[i]))
    print("\n",len(photons_of_groups[i]),len(binding_photons[i]))

#%%
for i in interesting_picks:
    #photons_to_plot = photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)] 
    binding_photons[i] = np.array(binding_photons[i])
    toplot =np.where(binding_photons[i] > 2)[0]
    toplot =np.where(binding_photons[i][toplot] < 2000000)[0]
    print("raw", np.mean(photons_of_groups[i][toplot]),"grouped", np.mean(binding_photons[i][toplot]))
    plt.hist(binding_photons[i][toplot], alpha=0.5, label=subgroups[i]) #, bins=50, range=(0,5e+3) )
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Photons per Event")
    plt.ylabel("log scale Events")
#    plt.show()

#    plt.ylim(0,10)
#    plt.hist(photons_of_groups[i],alpha=0.1)

#plt.hist(binding_photons)



