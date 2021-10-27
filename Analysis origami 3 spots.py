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

folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center"
#folder_path_save = 'C:/Projects/FLAKES/Origami photons laterals vs center/Sample 22_ 2lvlBio _2 INFLAKE'


parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]

file = 'C:/Origami testing Widefield/2021-10-22 Staples seal test 1/C_Core&Staples_532nm_12mW(40F4)_2x2_100ms_10kf_slow_Circular_1/25 origamis info.hdf5'

#file = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_2/Automatic_OUT flake 72 origamis.hdf5'


tic = time.time()
nametoload = file[:-5] + ".npz"
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

#for i in range(3):
#    plt.hist(photons_of_groups[i],bins=50, label="pick {}".format(i), range=(0,3000), alpha=0.5)
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

#%% Filter one: keep only more than 400 locs

good_picks = []

threshold = 100
photon_treshold = 10000
for i in range(len(photons_of_groups)):
    if len(np.where(photons_of_groups[i]<photon_treshold)[0]) > threshold:
        good_picks.append(i)

 
       
plt.plot(good_picks,'*', label="N={}".format(len(good_picks)//3))
plt.legend()
plt.title("origamis with enough locs")
plt.show()
print(len(good_picks)//3)

#%% Filter two, the sides have to be simetric in locs , ± 50 %

delta = 10
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
print(len(simetric_picks)//3)
# =============================================================================

##%%
#np.mean(photons_of_groups[3])
#np.mean(photons_of_groups[3][np.where(photons_of_groups[3]<1500)])


#%%

#for l in range(len(samples)):



#DATAFROM = "OutFlake_Filtered_{}locs_{}%delta".format(threshold, int(delta*100))
#DATAFROM = "OUTFlake_RAW"
DATAFROM = "Raw_Epi FRETx4"
folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/FRETx4"

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

photon_treshold = 6000
for i in range(len(photons_of_groups)):
    if i in simetric_picks:
        h1 = plt.hist(photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)], bins=50,
                      range=(0,photon_treshold), density=False, alpha=0.5,
                      label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i]))+" locs"))
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
#            plt.savefig(figure_path,  dpi = 300, bbox_inches='tight')
            plt.show()
#            plt.close()
            good_origamis.append(Norigami)
print("saved {} origamis in: \n".format(len(simetric_picks)//3), folder_path_save)

#%% Now do analysis of the good ones only

#mean_values = np.zeros(len(photons_of_groups))

left_picks = np.arange(0,len(photons_of_groups),3)
#center_picks = np.arange(1,len(photons_of_groups),3)
#right_picks = np.arange(2,len(photons_of_groups),3)

#multiples3 = (np.arange(nsamples))*3 is above

#photon_treshold = 10000
mean_values = np.zeros(len(photons_of_groups))
counting = 0
for i in left_picks:  # in multiples3:
    if i in simetric_picks[:-2]:
        left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)])
        right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<photon_treshold)])
        if left_mean > right_mean*0.8 and left_mean < right_mean*1.2:
            counting += 1
#            print("i",i)
            new_photon_treshold = (np.max([left_mean,right_mean]))*1.3
            center_mean = np.mean(photons_of_groups[i+1][np.where(photons_of_groups[i+1]<new_photon_treshold)])
        
            plt.plot(["left","center","right"],[left_mean, center_mean, right_mean],'*-',
                     label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i])))+" locs" + "\n" +
                     (str(i+1)+subgroups[i+1]+"; "+str(len(photons_of_groups[i+1])))+" locs"+ "\n" +
                     (str(i+2)+subgroups[i+2]+"; "+str(len(photons_of_groups[i+2])))+" locs" + "\n" +
                     "center_photons_treshold= " +str(int(new_photon_treshold)))
    
    #        print(i,"i")
            Norigami = ((i+1)//3)+1
            figure_name = '{}_{}_MeanPhotons_{}cutted'.format(DATAFROM, Norigami,photon_treshold)
            plt.title(figure_name)
            plt.legend()
            plt.ylabel("mean photons")
    
            figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
#            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#            plt.show()
            plt.close()

print("saved {} mean origamis out of {} \nin: ".format(counting, nsamples), folder_path_save)




#%%

SAVE_FIGS = True
SAVE_FIGS = False  



DATAFROM = "Epi_FRETx4_{}locs_{}%delta".format(threshold, int(delta*100))


folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/"


left_picks = np.arange(0,len(photons_of_groups),3)
#photon_treshold = 10000
mean_values = np.zeros(len(photons_of_groups))
counting = 0

mean_shift = 1.5
delta2 = 0.2


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
            plt.show()
            plt.close()

print("saved {} origamis out of {} \nin: ".format(counting, nsamples), folder_path_save)
print(file)


#%%

# =============================================================================
# 
# #%%
# # =============================================================================
# # 
# # fig, ax1 = plt.subplots()
# # 
# # # These are in unitless percentages of the figure size. (0,0 is bottom left)
# # left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
# # ax2 = fig.add_axes([left, bottom, width, height])
# # 
# # 
# # ax1.hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
# #               label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i]))+" locs"))
# # ax1.tick_params(axis='y')
# # 
# # color = 'tab:blue'
# # ax2.set_ylabel("photon mean", color=color)  # we already handled the x-label with ax1
# # ax2.plot(["left","center","right"],[left_mean, center_mean, right_mean],'*-',
# #          label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i])))+" locs" + "\n" +
# #          (str(i+1)+subgroups[i+1]+"; "+str(len(photons_of_groups[i+1])))+" locs"+ "\n" +
# #          (str(i+2)+subgroups[i+2]+"; "+str(len(photons_of_groups[i+2])))+" locs" + "\n" +
# #          "center_photons_treshold= " +str(int(new_photon_treshold)))
# # ax2.tick_params(axis='y', labelcolor=color)
# # 
# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # 
# # plt.show()
# # 
# # =============================================================================
# 
# # %%
# 
# # =============================================================================
# # fig, axs = plt.subplots(2)
# # 
# # fig.suptitle('aa ')
# # 
# # axs[0].hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
# #               label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i]))+" locs"))
# # 
# # #axs[0].tick_params(axis='x', labelcolor="red")  #   
# # plt.gca().get_xticklabels()[1].set_color('blue')
# # 
# # axs[1].set_ylabel("photon mean")
# # axs[1].plot(["left","center","right"],[left_mean, center_mean, right_mean],'*-',
# #          label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i])))+" locs" + "\n" +
# #          (str(i+1)+subgroups[i+1]+"; "+str(len(photons_of_groups[i+1])))+" locs"+ "\n" +
# #          (str(i+2)+subgroups[i+2]+"; "+str(len(photons_of_groups[i+2])))+" locs" + "\n" +
# #          "center_photons_treshold= " +str(int(new_photon_treshold)))
# # =============================================================================
# #axs[1].tick_params(axis='y')
# #%%
# 
# 
# #%% Until here it was the circular stuff for the gaussian
#         # Now is fitting to get the max of the photons in each of them
# #import sys
# #new_path = 'C:/Users/chiarelG/switchdrive/German data analysis/code'
# #if new_path not in sys.path:
# #    sys.path.append(new_path)
# #sys.path
# 
# from fiting_functions import plot_histo_fit
# 
# 
# 
# 
# gauss1 = np.zeros(len(photons_of_groups))
# gauss2 = np.zeros(len(photons_of_groups))
# 
# 
# photon_treshold = 10000
# for s in range(len(photons_of_groups)):
# #    print("s", s)
#     bines = len(photons_of_groups[s]) // 50
#     (gauss1[s], gauss2[s]) = plot_histo_fit(photons_of_groups[s][np.where(photons_of_groups[s]<photon_treshold)], bines, name="a", shift=0.9, double=True,ploting=False)
#     plt.show()
# #    plt.close()
# 
# 
# bines = 50    
# h1 = plt.hist(gauss1, bins=bines, alpha=0.5)
# h2 = plt.hist(gauss2, bins=bines, alpha=0.5)
# 
# 
# 
# #%%
# #SAVE_FIGS = True
# #SAVE_FIGS = False
# 
# from fiting_functions import plot_histo_fit
# 
# DATAFROM = "OUTFlake_{}locs_{}%delta".format(threshold, int(delta*100))
# 
# 
# folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/Sample 22_2lvlBio"
# 
# 
# left_picks = np.arange(0,len(photons_of_groups),3)
# photon_treshold = 50000
# mean_values = np.zeros(len(photons_of_groups))
# counting = 0
# 
# mean_shift = 1.3
# delta2 = 0.2
# 
# 
# gauss1 = np.zeros(len(photons_of_groups))
# gauss2 = np.zeros(len(photons_of_groups))
# 
# for i in left_picks:  # in multiples3:
#     if i in simetric_picks[:-2]:
#         left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)])
#         right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<photon_treshold)])
#         if left_mean > right_mean*(1-delta2) and left_mean < right_mean*(1+delta2):
#             counting += 1
# #            print("i",i)
#             new_photon_treshold = (np.max([left_mean,right_mean])) * mean_shift
#             center_mean = np.mean(photons_of_groups[i+1][np.where(photons_of_groups[i+1]<new_photon_treshold)])
#             new_left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<new_photon_treshold)])
#             new_right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<new_photon_treshold)])
# 
#             fig, axs = plt.subplots(2)
#             
#             for j in range(3):
#                 bines = len(photons_of_groups[s]) // 50
#                 label = (str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs")
#                 (gauss1[i+j], gauss2[i+j]) = plot_histo_fit(photons_of_groups[j][np.where(photons_of_groups[j]<new_photon_treshold)], bines, name=label, shift=0.9, double=True,ploting=False)
# 
#                 axs[0].hist(photons_of_groups[i+j], bins=100, density=False, alpha=0.5, range=(0,photon_treshold),
#                   label=(str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs"))
#             axs[0].legend()
# #            axs[0].hist(photons_of_groups[i], bins=30, range=(0,3000), density=True, alpha=0.5,
# #              label=(str(i)+subgroups[i]+"; "+str(len(photons_of_groups[i]))+" locs"))
# #            axs[0].hist(photons_of_groups[i+1], bins=30, range=(0,3000), density=True, alpha=0.5,
# #              label=(str(i+1)+subgroups[i+1]+"; "+str(len(photons_of_groups[i+1]))+" locs"))
# #            axs[0].hist(photons_of_groups[i+2], bins=30, range=(0,3000), density=True, alpha=0.5,
# #              label=(str(i+2)+subgroups[i+2]+"; "+str(len(photons_of_groups[i+2]))+" locs"))
# #            axs[0].legend()
#     
#             axs[1].set_ylabel("photon mean")
#             axs[1].plot(["left","center","right"],[new_left_mean, center_mean, new_right_mean],'*-',
#                      label= "center_photons_treshold= " +str(int(new_photon_treshold)))
#             axs[1].plot(["left","center","right"],[gauss1[i], gauss1[i+1], gauss1[i+2]],'*-',
#                      label= "Gauss1 ")
#             axs[1].plot(["left","center","right"],[gauss2[i], gauss2[i+1], gauss2[i+2]],'*-',
#                      label= "Gauss2 ")
#     
#     #        print(i,"i")
#             Norigami = ((i+1)//3)+1
#             figure_name = '{}_{}_MeanPhotons_{}cutted'.format(Norigami, DATAFROM, photon_treshold)
# #            plt.title(figure_name)
#             fig.suptitle(figure_name)
#             axs[1].legend()
#             plt.ylabel("mean photons")
#     
#             figure_path = os.path.join(folder_path_save, '%s.png' % figure_name)
# #            if SAVE_FIGS == True:
# #                plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
#             plt.show()
#             plt.close()
# 
# print("saved {} mean origamis out of {} \nin: ".format(counting, nsamples), folder_path_save)
# print(file)
# 
# plt.figure("GAuss")
# plt.plot(gauss1)
# 
# # %%
# 
# from fiting_functions import plot_histo_fit
# 
# DATAFROM = "OUTFlake_{}locs_{}%delta".format(threshold, int(delta*100))
# 
# 
# folder_path_save = "C:/Projects/FLAKES/Origami photons laterals vs center/Sample 22_2lvlBio"
# 
# 
# left_picks = np.arange(0,len(photons_of_groups),3)
# photon_treshold = 50000
# mean_values = np.zeros(len(photons_of_groups))
# counting = 0
# 
# mean_shift = 1.5
# delta2 = 0.25
# 
# 
# gauss1 = np.zeros(len(photons_of_groups))
# gauss2 = np.zeros(len(photons_of_groups))
# 
# for i in left_picks:  # in multiples3:
#     if i in simetric_picks[:-2]:
#         left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<photon_treshold)])
#         right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<photon_treshold)])
#         if left_mean > right_mean*(1-delta2) and left_mean < right_mean*(1+delta2):
#             counting += 1
# #            print("i",i)
#             new_photon_treshold = (np.max([left_mean,right_mean])) * mean_shift
#             center_mean = np.mean(photons_of_groups[i+1][np.where(photons_of_groups[i+1]<new_photon_treshold)])
#             new_left_mean = np.mean(photons_of_groups[i][np.where(photons_of_groups[i]<new_photon_treshold)])
#             new_right_mean = np.mean(photons_of_groups[i+2][np.where(photons_of_groups[i+2]<new_photon_treshold)])
#             
#             
#             for j in range(3):
#                 plt.figure("before")    
#                 plt.hist(photons_of_groups[i+j], bins=100, density=True, alpha=0.5, range=(0,new_photon_treshold),
#                   label=(str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs"))
#                 plt.legend()
#                 plt.title("before")
#                 
#                 plt.figure("After")
#                 bines = len(photons_of_groups[i+j]) // 50
#                 label = (str(i+j)+subgroups[i+j]+"; "+str(len(photons_of_groups[i+j]))+" locs")
#                 (gauss1[i+j], gauss2[i+j]) = plot_histo_fit(photons_of_groups[i+j][np.where(photons_of_groups[i+j]<new_photon_treshold)],
#                  bines, name=label, shift=0.9, double=True,ploting=True)
# 
#             plt.xlabel("photon treshold="+str(new_photon_treshold))
#             plt.show()
# 
# 
# #%%
# =============================================================================




