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

folder_path = "C:/Projects/FLAKES/Figuras Intensidad circles"

#filename_inflake = 'C:/Origami testing Widefield/2021-06-10_MoS2_samples_456_BSA_test/4_100ms_130nmpix_mode1/DNA_PAINT_1mW_9merAtto4881nM_trolox_glox_in_150ul_1xTAE12_2/DNA_PAINT_1mW_picked_IN flake2.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2021-06-10_MoS2_samples_456_BSA_test/4_100ms_130nmpix_mode1/DNA_PAINT_1mW_9merAtto4881nM_trolox_glox_in_150ul_1xTAE12_CONTROL_1/DNA_PAINT_1mW_488CONTROL_allorigamispicked.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2021-07-01 slide for basel/Slide_600pM_15min_oriBasel_50pM_10min_NP_30nM_imagerQ_scanSlow_16mW(30F1)_circular_1/aaa_all origamis in image.hdf5'

#DATAFROM = "Sample22_Flake1"
#filename_inflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_1/sample22_488_picked_centers.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_1/sample22_488_picked_laterals.hdf5'

#DATAFROM = "Sample22_Flake2"
#filename_inflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_2/sample22_488_picked_centers.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_2/sample22_488_picked_laterals.hdf5'

#DATAFROM = "Sample22_Flake3"
#filename_inflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_3/sample22_488_picked_centers.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_3/sample22_488_picked_laterals.hdf5'

#DATAFROM = "Sample22_Glass"
#filename_inflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_no_flake_1/sample22_488_picked_centers.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2021-07-02 Flake 22 biotin/sample22_488_1638uW_tirf2540_imager1nM_trolox-glox_ultimate_2lvl_biotin_no_flake_1/sample22_picked_laterals.hdf5'

#filename_inflake = 'C:/Origami testing Widefield/2029-09-23 Sample 35 NO-ssDNA Pyrene/Flake2_640nm_2.3mW_circular_optosplit_1/Flake2_cutted_forPicasso_locs.hdf5'
#finelame_outflake = 'C:/Origami testing Widefield/2029-09-23 Sample 35 NO-ssDNA Pyrene/Flake2_640nm_2.3mW_circular_optosplit_1/Origamis picked info.hdf5'
#
#filename_inflake2 = 'C:/Origami testing Widefield/2029-09-23 Sample 35 NO-ssDNA Pyrene/glass3_640nm_2.3mW_circular_optosplit_1/Glass3_cuttedforPicasso_locs.hdf5'
#finelame_outflake2 = 'C:/Origami testing Widefield/2029-09-23 Sample 35 NO-ssDNA Pyrene/glass3_640nm_2.3mW_circular_optosplit_1/origamis picked info.hdf5'

DATAFROM = "Test"


names = [filename_inflake, finelame_outflake ]  #, filename_inflake2, finelame_outflake2]
#samples = ["Flake2_complete", "flake2_origamis"]  # , "glass3_complete", "glass3_origamis"]
samples = ["centers", "laterals"]  # , "glass3_complete", "glass3_origamis"]

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

parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]

#samples = ["in flake", "out flake"]
#samples = ["out flake control"]
#samples = ["Fret 3 spots"]
#samples = ["center", "laterals"]

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
#        alldata[parameters[11]][j] = (data[j][11])  # group


    print( time.time()-tic)

    h1 = plt.hist(alldata["photons"], bins=200, range=(0,3000))




    finaldata[samples[l]] = alldata
plt.show()

#%%

#h7 = plt.hist(alldata["sx"], bins=200, range=(0.5,3))

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

Circle0 =  5.3 x  4.2 um Diameter ==> 41 x 32 pix aprox
Circle1 = 10.6 x  8.4 um Diameter ==> 82 x 64 pix aprox
Circle2 = 15.9 x 12.6 um Diameter ==> 123 x 96 pix aprox
Circle3 = 21.2 x 16.8 um Diameter ==> 164 x 128 pix aprox
Circle4 = 26.5 x 21.0 um Diameter ==> 205 x 160 pix aprox
"""

xc = 77 # 335//2  # 77
yc = 77 # 256//2  # 77
pixsize = 130  # 130
sigmax_laser = 5300  #  5300
sigmay_laser = 4200  #  4200
a = 1.0*(int(sigmax_laser/pixsize))
b = 1.0*(int(sigmay_laser/pixsize))

bines = 50
#hin = plt.hist2d(finaldata[samples[0]]['x'],finaldata[samples[0]]["y"], bins=bines, range=([0,xc*2], [0,yc*2]), cmin=0, cmax=4000)
#plt.colorbar(hin[3])
#plt.show()


N_circles = 3

x_circles = []
y_circles = []
photons_circles = []
bg_circles = []
for c in range(N_circles):
    x_circles.append("x_circle_{}".format(c))
    y_circles.append("y_circle_{}".format(c))
    photons_circles.append("photon_circle_{}".format(c))
    bg_circles.append("bg_circle_{}".format(c))

s=0
for s in range(len(samples)):
    hin = plt.hist2d(finaldata[samples[s]]['x'],finaldata[samples[s]]["y"], bins=bines, range=([0,xc*2], [0,yc*2]), cmin=0, cmax=4000)
    plt.colorbar(hin[3])
#    plt.title("figure {}".format(samples[s]))
    figure_name = '{}_{}_locs_complete'.format(DATAFROM, samples[s])
    plt.title(figure_name)
    figure_path = os.path.join(folder_path, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.close()
    

    x = np.array(finaldata[samples[s]]["x"])
    y = np.array(finaldata[samples[s]]["y"])
    
    
    #parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]


    for c in range(N_circles):
        tic = time.time()
    
        finaldata[samples[s]][x_circles[c]] = []
        finaldata[samples[s]][y_circles[c]] = []
        finaldata[samples[s]][photons_circles[c]] = []
        finaldata[samples[s]][bg_circles[c]] = []
        
        for i in range(len(x)):

            if c**2 < (((x[i]-xc)/a)**2 + ((y[i]-yc)/b)**2) <= (c+1)**2:
                finaldata[samples[s]][x_circles[c]].append(finaldata[samples[s]]["x"][i])
                finaldata[samples[s]][y_circles[c]].append(finaldata[samples[s]]["y"][i])
                finaldata[samples[s]][photons_circles[c]].append(finaldata[samples[s]]["photons"][i])
                finaldata[samples[s]][bg_circles[c]].append(finaldata[samples[s]]["bg"][i])

        print("\n s=",s," c=",c,";time=", time.time()-tic, ":")


        hist2d = plt.hist2d(finaldata[samples[s]][x_circles[c]], finaldata[samples[s]][y_circles[c]], bins=bines, range=([0,xc*2], [0,yc*2]), cmin=0, cmax=4000)
        plt.colorbar()
#        plt.title("Circle (start in cero) {}".format(c))
        figure_name = '{}_{}_locs_circle{}'.format(DATAFROM, samples[s],c)
        plt.title(figure_name)
#        figure_path = folder_path + "/" + figure_name + ".png"
        figure_path = os.path.join(folder_path, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()

#bines = 50
#hin = plt.hist2d(finaldata[samples[0]]['x'],finaldata[samples[0]]["y"], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
#plt.colorbar()
#plt.show()
#hist2d = plt.hist2d(x_circle[0]+x_circle[1]+ x_circle[2]+x_circle[3], y_circle[0]+y_circle[1]+y_circle[2]+y_circle[3], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
#plt.colorbar()
#plt.show()


##%%
#for c in range(N_circles):
#    hin = plt.hist(finaldata[samples[0]][photons_circles[c]], bins=60, alpha=0.5, range=(0,3500), label=photons_circles[c])
#plt.legend()
#plt.show()
#for c in range(N_circles):
#    hout = plt.hist(finaldata[samples[1]][photons_circles[c]], bins=60, alpha=0.5, range=(0,3500), label=photons_circles[c])
#plt.legend()
#plt.show()
        
        
#%% Until here it was the circular stuff for the gaussian

#s, c = 0, 1
for c in range((N_circles)):
    for s in range(1,len(samples)-1):
        hist2d = plt.hist2d(finaldata[samples[s]][x_circles[c]], finaldata[samples[s]][y_circles[c]], bins=bines, range=([0,xc*2], [0,yc*2]), cmin=0, cmax=1000)
        plt.colorbar()
        plt.show()
    
    
#%%

#s, c = 0, 0
rango = (0, 5000)
bines= 100
#h1 = plt.hist(finaldata[samples[s]][photons_circles[c]], bins=200, range=rango, alpha=0.5)
#s, c = 1, 0


for c in range((N_circles)):
    figure_name = '{}_Photons_circle{}'.format(DATAFROM,c)
    plt.figure(figure_name)
    plt.title(figure_name)
    for s in range(len(samples)):  #range(1,len(samples)-1):
        h1 = plt.hist(finaldata[samples[s]][photons_circles[c]], bins=bines, range=rango, alpha=0.5, density=True, label="sample_{}".format(samples[s]))
        plt.legend()

    figure_path = os.path.join(folder_path, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.close()


#%%

# =============================================================================
# 
# 
# #%% Until here it was the circular stuff for the gaussian
#         # Now is fitting to get the max of the photons in each of them
# import sys
# new_path = 'C:/Users/chiarelG/switchdrive/German data analysis/code'
# if new_path not in sys.path:
#     sys.path.append(new_path)
# #sys.path
# 
# from fiting_functions import plot_histo_fit
# 
# #if __name__ == '__main__':
# #    print("aa")
# #a = os.path.abspath('junk.txt')
# #os.path.dirname(a)
# #os.path.basename(a)
# #import glob
# #b = glob.glob('*.py')
# #b[1]
# 
# data = finaldata[samples[0]][photons_circles[0]]
# 
# bines = 100
# gauss1 = np.zeros((N_circles))
# gauss2 = np.zeros((N_circles))
# gauss3 = np.zeros((N_circles))
# gauss4 = np.zeros((N_circles))
# 
# for c in range(2):  # ,N_circles):
#     print(c)
# 
#     (gauss1[c], gauss2[c]) = plot_histo_fit(finaldata[samples[0]][photons_circles[c]],
#                                 bines, photons_circles[c])
#     plt.xlim([0,4000])
#     plt.show()
# 
# #for c in range(N_circles):
#     try:
#         (gauss3[c], gauss4[c]) = plot_histo_fit(finaldata[samples[1]][photons_circles[c]], bines, photons_circles[c])
#     finally:
#         print("wtf?")
# plt.xlim([0,4000])
# plt.show()
# 
# print("\n", gauss1,"\n", gauss2)
# print("\n", gauss3,"\n", gauss4)
# 
# #plt.plot(gauss1,".-",label=samples[0])
# ##plt.plot(gauss2,".-",label="gauss2")
# #plt.plot(gauss3,".-",label=samples[1])
# ##plt.plot(gauss4,".-",label="gauss4")
# #plt.legend()
# 
# 
# #%%
# bines = 100
# gauss5 = np.zeros(len(samples))
# gauss6 = np.zeros(len(samples))
# 
# for s in range(len(samples)):
#     (gauss5[s], gauss6[s]) = plot_histo_fit(finaldata[samples[s]][photons_circles[2]], bines, samples[s])
# 
# print("\n", gauss5,"\n")
# #%%
# # =============================================================================
# # import numpy as np
# # import matplotlib.pyplot as plt
# # 
# # import pylab as plb
# # from scipy.optimize import curve_fit
# # from scipy import asarray as ar,exp
# # 
# # #N=[50,200,200]
# # largos = []
# # for c in range(N_circles):
# #     largos.append(len(finaldata[samples[s]][photons_circles[c]]))
# # largos = np.array(largos)
# # 
# # bines = 100
# # auxe = largos / bines
# # largos/auxe
# # 
# # nbins = auxe #[10]*4 # [10, 10, 10, 10, 10, 10, 10]
# # 
# # c=0
# # fit_max = np.zeros((N_circles))
# # fit_min = np.zeros((N_circles))
# # 
# # for c in range(N_circles):
# #     N = nbins[c]
# #     bins = np.linspace(0,int(np.max(phothons_circle[c])), N)
# #     
# #     plt.hist(phothons_circle[c], bins=bins, alpha = 0.5, label="Photons_circle {}".format(c))# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
# #     y,x = np.histogram(phothons_circle[c], bins=bins)  #len(nozeros)//N
# #     x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
# #     
# #     n = len(x)                          #the number of data
# #     mean = sum(x*y)/sum(y)                   #note this correction
# #     sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction
# #     
# #     def gaus(x,a,x0,sigma):
# #         return a*exp(-(x-x0)**2/(2*sigma**2))
# #     
# #     popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma/100])
# #     perr = np.sqrt(np.diag(pcov))
# #     
# #     
# #     #plt.plot(x,y,'b+:',label='data')
# #     X = np.linspace(x[0], x[-1], 500)
# #     plt.plot()
# #     plt.plot(X,gaus(X,*popt),'g',lw=2, label='1G fit')
# #     plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
# #     plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
# #     plt.legend()
# #     plt.title('hist')
# #     plt.xlabel('Counts kHz')
# #     plt.ylabel("total points ={} in {} bins".format(len(phothons_circle[c]), N))
# #     plt.xlim([0,3000])
# #     #    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
# #     print(popt[1], popt[2], "1")
# #     #plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
# #     #plt.xlim(0, int(np.max(nozeros)))
# #     #plt.show()
# #         
# #     
# #     def gauss(x,mu,sigma,A):
# #         return A*exp(-(x-mu)**2/2/sigma**2)
# #     
# #     def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
# #         return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
# #     
# #     
# #     expected = (popt[1],abs(popt[2]),popt[0],
# #                 0.5*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])
# #     
# #     
# #     try:
# #         params,cov = curve_fit(bimodal,x,y,expected)
# #         sigma = np.sqrt(np.diag(cov))
# #     
# #         if params[0] < 0 or params[3] < 0:
# #             print("bad adjusts 2G")
# #         else:
# #             #X = np.linspace(x[0]-50, x[-1]+50, 5000)
# #             plt.plot(X,bimodal(X,*params),color='orange',lw=3,label='2G model')
# #             plt.legend()
# #             plt.vlines((params[0], params[3]), color=('r','b'), ymin=0,ymax=0.5*popt[0])
# # #        print(params,'\n',sigma)
# #         #print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
# #     except:
# #         params = ["no"]*6
# #     
# # #    print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
# # #    print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
# # #              "\n",(params[3],"±",params[4]), "*", params[5])
# #     fit_max[c] = int(params[0])
# #     fit_min[c] = int(params[3])
# # 
# # print(fit_max)
# # print("\n",fit_min)
# # =============================================================================
# 
# #%% Open scans from labview:
# import numpy as np
# import matplotlib.pyplot as plt
# 
# data = np.loadtxt('C:/Data Confocal/CONFOCAL/2021-07-22 flake 25/4_scan G0,7/image in txt.txt')
# print(data.shape)
# 
# 
# fig = plt.figure(figsize=(8, 6))
# 
# ax = fig.add_subplot(111)
# ax.set_title('colorMap')
# plt.imshow(data)
# ax.set_aspect('equal')
# #
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# plt.colorbar(orientation='vertical')
# plt.show()
# 
# #%%
# 
# 
# =============================================================================
