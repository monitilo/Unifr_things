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

Circle0 =  5.3 x  4.2 um Diameter ==> 41 x 32 pix aprox
Circle1 = 10.6 x  8.4 um Diameter ==> 82 x 64 pix aprox
Circle2 = 15.9 x 12.6 um Diameter ==> 123 x 96 pix aprox
Circle3 = 21.2 x 16.8 um Diameter ==> 164 x 128 pix aprox
Circle4 = 26.5 x 21.0 um Diameter ==> 205 x 160 pix aprox
"""

xc = 77
yc = 77
pixsize = 130
sigmax_laser = 5300
sigmay_laser = 4200
a = 0.5*(int(sigmax_laser/pixsize))
b = 0.5*(int(sigmay_laser/pixsize))

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
    
        print("\n c=",c,";time=", time.time()-tic, ":")
    
    
        hist2d = plt.hist2d(finaldata[samples[s]][x_circles[c]], finaldata[samples[s]][y_circles[c]], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
        plt.colorbar()
        plt.show()

#bines = 50
#hin = plt.hist2d(finaldata[samples[0]]['x'],finaldata[samples[0]]["y"], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
#plt.colorbar()
#plt.show()
#hist2d = plt.hist2d(x_circle[0]+x_circle[1]+ x_circle[2]+x_circle[3], y_circle[0]+y_circle[1]+y_circle[2]+y_circle[3], bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
#plt.colorbar()
#plt.show()


#%%
for c in range(N_circles):
    hin = plt.hist(finaldata[samples[0]][photons_circles[c]], bins=60, alpha=0.5, range=(0,3500), label=photons_circles[c])
plt.legend()
plt.show()
for c in range(N_circles):
    hout = plt.hist(finaldata[samples[1]][photons_circles[c]], bins=60, alpha=0.5, range=(0,3500), label=photons_circles[c])
plt.legend()
plt.show()

#%%
from
data = finaldata[samples[0]][photons_circles[0]]

bines = 100

(gauss1, gauss2) = plot_histo_fit(data, bines)

#%%
import numpy as np
import matplotlib.pyplot as plt

import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

#N=[50,200,200]
largos = []
for c in range(N_circles):
    largos.append(len(finaldata[samples[s]][photons_circles[c]]))
largos = np.array(largos)

bines = 100
auxe = largos / bines
largos/auxe

nbins = auxe #[10]*4 # [10, 10, 10, 10, 10, 10, 10]

c=0
fit_max = np.zeros((N_circles))
fit_min = np.zeros((N_circles))

for c in range(N_circles):
    N = nbins[c]
    bins = np.linspace(0,int(np.max(phothons_circle[c])), N)
    
    plt.hist(phothons_circle[c], bins=bins, alpha = 0.5, label="Photons_circle {}".format(c))# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
    y,x = np.histogram(phothons_circle[c], bins=bins)  #len(nozeros)//N
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    
    n = len(x)                          #the number of data
    mean = sum(x*y)/sum(y)                   #note this correction
    sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction
    
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma/100])
    perr = np.sqrt(np.diag(pcov))
    
    
    #plt.plot(x,y,'b+:',label='data')
    X = np.linspace(x[0], x[-1], 500)
    plt.plot()
    plt.plot(X,gaus(X,*popt),'g',lw=2, label='1G fit')
    plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
    plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
    plt.legend()
    plt.title('hist')
    plt.xlabel('Counts kHz')
    plt.ylabel("total points ={} in {} bins".format(len(phothons_circle[c]), N))
    plt.xlim([0,3000])
    #    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
    print(popt[1], popt[2], "1")
    #plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
    #plt.xlim(0, int(np.max(nozeros)))
    #plt.show()
        
    
    def gauss(x,mu,sigma,A):
        return A*exp(-(x-mu)**2/2/sigma**2)
    
    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
    
    
    expected = (popt[1],abs(popt[2]),popt[0],
                0.5*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])
    
    
    try:
        params,cov = curve_fit(bimodal,x,y,expected)
        sigma = np.sqrt(np.diag(cov))
    
        if params[0] < 0 or params[3] < 0:
            print("bad adjusts 2G")
        else:
            #X = np.linspace(x[0]-50, x[-1]+50, 5000)
            plt.plot(X,bimodal(X,*params),color='orange',lw=3,label='2G model')
            plt.legend()
            plt.vlines((params[0], params[3]), color=('r','b'), ymin=0,ymax=0.5*popt[0])
#        print(params,'\n',sigma)
        #print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
    except:
        params = ["no"]*6
    
#    print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
#    print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
#              "\n",(params[3],"±",params[4]), "*", params[5])
    fit_max[c] = int(params[0])
    fit_min[c] = int(params[3])

print(fit_max)
print("\n",fit_min)

#%%

bines = 100
#auxe = largos / bines
#largos/auxe

#nbins = auxe #[10]*4 # [10, 10, 10, 10, 10, 10, 10]

c=0

def plot_histo_fit(vector, bines):
    import numpy as np
    import matplotlib.pyplot as plt
    
    import pylab as plb
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp
    N = len(vector)/bines
    bins = np.linspace(0,int(np.max(vector)), N)
    
    plt.hist(vector, bins=bins, alpha = 0.5, label="Photons_circle {}".format(c))# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
    y,x = np.histogram(vector, bins=bins)  #len(nozeros)//N
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    
    n = len(x)                          #the number of data
    mean = sum(x*y)/sum(y)                   #note this correction
    sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction
    
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma/100])
    perr = np.sqrt(np.diag(pcov))
    
    
    #plt.plot(x,y,'b+:',label='data')
    X = np.linspace(x[0], x[-1], 500)
    plt.plot()
    plt.plot(X,gaus(X,*popt),'g',lw=2, label='1G fit')
    plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
    plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
    plt.legend()
    plt.title('hist')
    plt.xlabel('Counts kHz')
    plt.ylabel("total points ={} in {} bins".format(len(vector), N))
    plt.xlim([0,3000])
    #    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
    print(popt[1], popt[2], "1")
    #plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
    #plt.xlim(0, int(np.max(nozeros)))
    #plt.show()
        
    
    def gauss(x,mu,sigma,A):
        return A*exp(-(x-mu)**2/2/sigma**2)
    
    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
    
    
    expected = (popt[1],abs(popt[2]),popt[0],
                0.5*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])
    
    
    try:
        params,cov = curve_fit(bimodal,x,y,expected)
        sigma = np.sqrt(np.diag(cov))
    
        if params[0] < 0 or params[3] < 0:
            print("bad adjusts 2G")
        else:
            #X = np.linspace(x[0]-50, x[-1]+50, 5000)
            plt.plot(X,bimodal(X,*params),color='orange',lw=3,label='2G model')
            plt.legend()
            plt.vlines((params[0], params[3]), color=('r','b'), ymin=0,ymax=0.5*popt[0])
#        print(params,'\n',sigma)
        #print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
    except:
        params = ["no"]*6
    
#    print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
#    print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
#              "\n",(params[3],"±",params[4]), "*", params[5])
    return (int(params[0]), int(params[3]))
