# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:58:32 2019

@author: Santiago
"""

import numpy as np
import matplotlib.pyplot as plt
#%% Otro intente. El maximo lo encuentra de izq a der, y el minimo empezando por la der

#'C:/Analizando Imagenes/Time Trace(s)-MML-50mW_3 Automatic132.csv',
#            'C:/Analizando Imagenes/Time Trace(s) 50mW 100ms Automaticpick (179).csv',
#
allfiles = ['C:/Analizando Imagenes/Time Trace(s)-MML-50mW_3.csv',
            'C:/Analizando Imagenes/Time Trace(s)-MML-50mW_2.csv',
            'C:/Analizando Imagenes/Time Trace(s)-MML-50mW_1.csv',
            'C:/Analizando Imagenes/Time Trace(s)-50mW_100ms-hand.csv']#,
#            'C:/Analizando Imagenes/Time Trace(s)-50mW_200ms-hand.csv']

PLOT = False
#allfiles = ['C:/Analizando Imagenes/50nmANTENNAS_50mW_hand.csv']
#allfiles = ['C:/Analizando Imagenes/Time Trace(s)-MML-50mW_3.csv']
alldata = {}
for filen in range (len(allfiles)):
    print(allfiles[filen])
    traces = np.loadtxt(allfiles[filen], delimiter=',', skiprows=1)
    [Nframes, Ntracesbad] = traces.shape
    Ntraces = Ntracesbad  - 2  # The two final colums of imagej are another things
    
    
    maxs = []
    means = []
    mins = []
    
    realmaxs = []
    realmins = []
    
    hipart = {}
    lowpart = {}
    
    lessmax = 0.97
    moremin = 1.03
    
    N =  Ntraces
    
    highpart = {}
    lowpart = {}
    
    for i in range(N):
        print(i)
        if PLOT:        
            plt.figure(i)
            plt.plot(traces[:,i], '*b')
    
        maxs.append(np.max(traces[:,i]))
        means.append(np.mean(traces[:,i]))
        mins.append(np.min(traces[:,i]))
    
    
        H = True
        L = True
        mm = []
        ml = []
        hipart[i] = [traces[1,i]]
        lowpart[i] = [traces[-1,i]]
        for j in range(1, Nframes-1):
    
            if H:
                mm.append(traces[j,i])
    
                if (np.mean(np.array(mm)))*lessmax < traces[j+1,i]:
                    hipart[i].append(traces[j+1,i])
    
                else:
                    if (np.mean(np.array(mm)))*lessmax < traces[j+2,i]:
                        hipart[i].append(traces[j+1,i])
                    else:
                        if (np.mean(np.array(mm)))*lessmax < traces[j+3,i]:
                            hipart[i].append(traces[j+1,i])
                        else:
                            # Finished the high part
                            H = False
    
            if L:
                ml.append(traces[-j,i])
    
                if traces[-j-1,i] < (np.mean(np.array(ml)))*moremin:
                    lowpart[i].append(traces[-j-1,i])
                    
                else:
                    if traces[-j-2,i] < (np.mean(np.array(ml)))*moremin:
                        lowpart[i].append(traces[-j-1,i])
                    else:
                        if traces[-j-3,i] < (np.mean(np.array(ml)))*moremin:
                            lowpart[i].append(traces[-j-1,i])
                        else:
                            L = False
                
    #    print(len(hipart[i]))
    #    print(len(lowpart[i]))
    #    plt.plot(hipart[i], 'r')
    #    plt.plot(lowpart[i], 'g')
    
        fixedhighpartaux = np.array(hipart[i])
        fixedhighpart = np.concatenate((fixedhighpartaux, np.nan*np.zeros(Nframes-len(fixedhighpartaux))),axis=0)
    
        fixedlowpartaux = np.array(np.flip(lowpart[i]))
        fixedlowpart = np.concatenate((np.nan*np.zeros(Nframes-len(fixedlowpartaux)), fixedlowpartaux,),axis=0)
    
        realmaxs.append(np.mean(hipart[i]))
        realmins.append(np.mean(lowpart[i]))
        
        if PLOT:
            plt.plot(fixedhighpart, '--r')
            plt.plot(fixedlowpart, '--g')    
    
    
    
    #plt.figure("max,mean and min")
    #plt.plot(maxs, 'r')
    #plt.plot(means, 'b')
    #plt.plot(mins, 'g')
    #plt.plot(realmaxs, '.-m')
    #plt.plot(realmins, '*-k')
    
    #plt.plot(np.array(realmaxs)-np.array(realmins), '*m')
    
#    mu = np.mean(np.array(realmaxs)-np.array(realmins))
#    sigma = np.sqrt(((len(realmaxs)-1)**(-1))*np.sum(((np.array(realmaxs)-np.array(realmins))-mu)**2))
#    #se = sigma/np.sqrt(Nframes)
#    
#    print(mu,"+-", sigma)
    #plt.figure("hist")
    #plt.hist(np.array(realmaxs)-np.array(realmins))
    alldata[filen] = (np.array(realmaxs)-np.array(realmins))

histogram = []
#allhistograms = []
plt.figure("allhistograms")
for p in range(len(alldata)):
#    allhistograms.append(alldata[p])
    histogram = np.concatenate((np.array(histogram), alldata[p]))
    plt.hist(alldata[p], label=len(alldata[p]))
    plt.legend()

#plt.hist(allhistograms)

plt.figure("one big histogram")
plt.hist(histogram, int(len(histogram)/2), color ='m')
plt.title((len(histogram)))
plt.grid()
print(len(histogram))

mu = np.mean(np.array(histogram))
sigma = np.sqrt(((len(histogram)-1)**(-1))*np.sum((histogram-mu)**2))
#se = sigma/np.sqrt(Nframes)

print(mu,"+-", sigma)

#%% Prueba bonita a mano
     
traces = np.loadtxt(filen, delimiter=',', skiprows=1)


[Nframes, Ntracesbad] = traces.shape
Ntraces = Ntracesbad  - 2  # The two final colums of imagej are another things


maxs = []
means = []
mins = []

realmaxs = []
realmins = []

hipart = {}
lowpart = {}

lessmax = 0.97
moremin = 1.1

N = Ntraces

highpart = {}
lowpart = {}

for i in range(N):
    print(i)

    plt.figure(i)
    plt.plot(traces[:,i], '*b')

#    plt.xlim((340,500))
    maxs.append(np.max(traces[:,i]))
    means.append(np.mean(traces[:,i]))
    mins.append(np.min(traces[:,i]))

   
    H = True

    mm = []
    hipart[i] = [traces[0,i]]
    lowpart[i] = []
    for j in range(1, Nframes-1):

        if H:
            mm.append(traces[j,i])

            if (np.mean(np.array(mm)))*lessmax < traces[j+1,i] < (np.mean(np.array(mm)))*moremin:
                hipart[i].append(traces[j-1,i])

            else:
                if (np.mean(np.array(mm)))*lessmax < traces[j+2,i] < (np.mean(np.array(mm)))*moremin:
                    hipart[i].append(traces[j-1,i])
                else:
                    #lowpart starts
                    H = False

        if not H:
            lowpart[i].append(traces[j,i])
            
#    print(len(hipart[i]))
#    print(len(lowpart[i]))
#    plt.plot(hipart[i], 'r')
#    plt.plot(lowpart[i], 'g')

    fixedhighpartaux = np.array(hipart[i])
    fixedhighpart = np.concatenate((fixedhighpartaux, np.nan*np.zeros(Nframes-len(fixedhighpartaux))),axis=0)

    fixedlowpartaux = np.array(lowpart[i])
    fixedlowpart = np.concatenate((np.nan*np.zeros(Nframes-len(fixedlowpartaux)), fixedlowpartaux,),axis=0)

    realmaxs.append(np.mean(hipart[i]))
    realmins.append(np.mean(lowpart[i]))

    plt.plot(fixedhighpart, '--r')
    plt.plot(fixedlowpart, '--g')    



plt.figure("max,mean and min")
plt.plot(maxs, 'r')
plt.plot(means, 'b')
plt.plot(mins, 'g')
plt.plot(realmaxs, '.-m')
plt.plot(realmins, '*-k')

#plt.plot(np.array(realmaxs)-np.array(realmins), '*m')

mu = np.mean(np.array(realmaxs)-np.array(realmins))
sigma = np.sqrt(((Ntraces-1)**(-1))*np.sum(((np.array(realmaxs)-np.array(realmins))-mu)**2))
#se = sigma/np.sqrt(Nframes)

print(mu,"+-", sigma)
plt.figure("hist0")
plt.hist(np.array(realmaxs)-np.array(realmins))

#%%
#data5 = np.array(realmaxs)-np.array(realmins)

muchadata = np.concatenate((np.array(data*0.5), np.array(data2),np.array(data3), np.array(data4),np.array(data5)))
plt.hist(muchadata)
plt.grid()
print(len(muchadata))
#%%
muu = np.mean(muchadata)
sigmaa = np.sqrt(((len(muchadata)-1)**(-1))*np.sum((muchadata-mu)**2))
print(muu,"+-", sigmaa)

#%% La segunda columna resta
filen = 'C:/Analizando Imagenes/Time Trace(s) 20mW_200ms-(resta la segunda).csv'

#c = ['g','g','g','g','g','k','.r','.r','.r','k','.b','.b','.b','.b','k','c','c','c','k','m','m','m']
#alldata=np.zeros((N,a-3-skip))
#allxaxis=np.copy(alldata)

traces = np.loadtxt(filen, delimiter=',', skiprows=1)

[Nframes, Ntraces] = traces.shape

maxs = []
means = []
mins = []

realmaxs = []
realmins = []

highpart = {}
lowpart = {}

lessmax = 0.9
moremin = 1.08

N = 2 # Ntraces-3

for j in range(int(N/2)):
    i = 6 # j*2
    print(i)
     
    plt.figure(i)
    plt.plot(traces[:,i], 'b')
    plt.plot(traces[:,i]-traces[:,i+1], '.g')
    plt.plot(traces[:,i+1], '--r')

#    plt.xlim((340,500))
    maxs.append(np.max(traces[:,i]))
    means.append(np.mean(traces[:,i]))
    mins.append(np.min(traces[:,i]))
    
#    print(maxs[i], maxs[i]*lessmax)
#    print(mins[i], mins[i]*moremin


mu = np.mean(np.array(realmaxs)-np.array(realmins))
sigma = np.sqrt((Nframes**(-1))*np.sum(((np.array(realmaxs)-np.array(realmins))-mu)**2))
#se = sigma/np.sqrt(Nframes)

print(mu,"+-", sigma*4)



#%% Analisis de muchos puntos
filen = 'C:/Analizando Imagenes/Time Trace(s) Prueba1.csv'

#c = ['g','g','g','g','g','k','.r','.r','.r','k','.b','.b','.b','.b','k','c','c','c','k','m','m','m']
#alldata=np.zeros((N,a-3-skip))
#allxaxis=np.copy(alldata)

traces = np.loadtxt(filen, delimiter=',', skiprows=1)

[Nframes, Ntraces] = traces.shape

maxs = []
means = []
mins = []

realmaxs = []
realmins = []

highpart = {}
lowpart = {}

lessmax = 0.9
moremin = 1.08

N = Ntraces-3

for i in range(N):
    print(i)
    
    plt.figure(i)
#    plt.plot(traces[:,i], '.b')
#    plt.xlim((340,500))
    maxs.append(np.max(traces[:,i]))
    means.append(np.mean(traces[:,i]))
    mins.append(np.min(traces[:,i]))
    
#    print(maxs[i], maxs[i]*lessmax)
#    print(mins[i], mins[i]*moremin
    highpart[i] = []
    lowpart[i] = []
    
    for j in range(Nframes):  # Puedo cambiar esto con un while para que tome todos los puntos hasta que baje.
        # y luego tome los siguentes (incluso hasta que baje de nuevo). y voy promediando

        if maxs[i]*lessmax < traces[j,i] <= maxs[i]:
            highpart[i].append(traces[j,i])

        if mins[i] <= traces[j,i] < mins[i] * moremin:
            lowpart[i].append(traces[j,i])

    fixedhighpartaux = np.array(highpart[i])
    fixedhighpart = np.concatenate((fixedhighpartaux, np.nan*np.zeros(Nframes-len(fixedhighpartaux))),axis=0)

    fixedlowpartaux = np.array(lowpart[i])
    fixedlowpart = np.concatenate((np.nan*np.zeros(Nframes-len(fixedlowpartaux)), fixedlowpartaux,),axis=0)

    realmaxs.append(np.mean(highpart[i]))
    realmins.append(np.mean(lowpart[i]))

#    plt.plot(fixedhighpart, '--r')
#    plt.plot(fixedlowpart, '--g')    

#    plt.plot(np.concatenate((np.array(highpart[i]), np.array(lowpart[i]))), '--r')

#    plt.plot(highpart[i])
#    plt.plot(lowpart[i])

#plt.figure()
#plt.grid()
#plt.plot(traces[:,0])
    
plt.figure("max,mean and min")
plt.plot(maxs, 'r')
plt.plot(means, 'b')
plt.plot(mins, 'g')
plt.plot(realmaxs, '.-')
plt.plot(realmins, '.-')

#plt.plot(np.array(realmaxs)-np.array(realmins), '*m')

mu = np.mean(np.array(realmaxs)-np.array(realmins))
sigma = np.sqrt((Nframes**(-1))*np.sum(((np.array(realmaxs)-np.array(realmins))-mu)**2))
#se = sigma/np.sqrt(Nframes)

print(mu,"+-", sigma*4)
plt.figure("hist")
plt.hist(np.array(realmaxs)-np.array(realmins))

#%% Morgan measure. 50C nanodrop
file_green = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 Dimers 09.10.19 (50C)(MORGANE)/UV-Vis 10_8_2019 9_35_14 AM.tsv'
a = 1333
skip = 9
N = 22
c = ['g','g','g','g','g','k','.r','.r','.r','k','.b','.b','.b','.b','k','c','c','c','k','m','m','m']
alldata=np.zeros((N,a-3-skip))
allxaxis=np.copy(alldata)

for i in range(N):
    print(i)
    
    spectrum = np.loadtxt(file_green, delimiter='\t', skiprows = skip+i*a, max_rows = a-skip) # 1334
    alldata[i,:] = spectrum[:,1]
    waveleng = spectrum[:,0]
    plt.plot(spectrum[:,0], spectrum[:,1], c[i])
    plt.xlim((340,500))
    
    left = np.where(waveleng==340)[0][0]
    right = np.where(waveleng==500)[0][0]
    
    maxpeak = np.where(spectrum[left:right,1]==np.max(spectrum[left:right,1]))[0][0]
    maxvalue = spectrum[maxpeak+left,1]
    
    print( "maxpeak = ", waveleng[maxpeak+left], "\n maxvalue y=", maxvalue)
    
plt.figure()
plt.grid()
#plt.plot(spectrum[:,0], spectrum[:,1])
plt.plot(waveleng, alldata[6,:],'g')
plt.plot(waveleng, alldata[10,:],'r')
#plt.xlim((340,500))
#for i in range(2):
#    print(i+16)
#    plt.plot(waveleng, alldata[i+16,:])

#%%
file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 NPs 21.10/UV-Vis 10_21_2019 2_22_04 AM.tsv'
a = 1333
skip = 9
N = 7
c = ['g','r','b','k','g','r','b']
alldata=np.zeros((N,a-3-skip))
allxaxis=np.copy(alldata)

for i in range(N):
    print(i)
    
    spectrum = np.loadtxt(file, delimiter='\t', skiprows = skip+i*a, max_rows = a-skip) # 1334
    alldata[i,:] = spectrum[:,1]
    waveleng = spectrum[:,0]
    plt.plot(spectrum[:,0], spectrum[:,1], c[i])
    plt.xlim((340,500))

    
    left = np.where(waveleng==340)[0][0]
    right = np.where(waveleng==500)[0][0]
    
    maxpeak = np.where(spectrum[left:right,1]==np.max(spectrum[left:right,1]))[0][0]
    maxvalue = spectrum[maxpeak+left,1]
    plt.ylim((0,maxvalue*1.2))

    print( "maxpeak = ", waveleng[maxpeak+left], "\n maxvalue y=", maxvalue)
    
#plt.figure()
#plt.grid()
#plt.plot(spectrum[:,0], spectrum[:,1])
#plt.plot(waveleng, alldata[6,:],'g')
#plt.plot(waveleng, alldata[10,:],'r')
#plt.xlim((340,500))
#for i in range(2):
#    print(i+16)
#    plt.plot(waveleng, alldata[i+16,:])

# %% Ver las trazas juntas rojo y verde (FRET)
Ef = []
for i in range(1):
    print(i)
    file_green = 'C:/Origami testing Widefield/2019-10-07 COPIAR/Ori_for_antennas-10mW_532nm-Zoom-200ms_1/82_41.csv' #str(i) + "_traza_verde" + ".txt"
    file_red = 'C:/Origami testing Widefield/2019-10-07 COPIAR/Ori_for_antennas-10mW_532nm-Zoom-200ms_2/43_26.csv' #str(i) + "_traza_rojo" + ".txt"
    file_bkg_green = 'C:/Origami testing Widefield/2019-10-07 COPIAR/Ori_for_antennas-10mW_532nm-Zoom-200ms_1/82_41-Noice.csv' #str(i) + "_bkg_verde" + ".txt"
    file_bkg_red = 'C:/Origami testing Widefield/2019-10-07 COPIAR/Ori_for_antennas-10mW_532nm-Zoom-200ms_2/43_26-Noice.csv' # str(i) + "_bkg_rojo" + ".txt"

    file_green = 'C:/Origami testing Widefield/2019-10-04/40nm-antennas_3fmol_10mW-532nM_1/161_302.csv'
    file_bkg_green = 'C:/Origami testing Widefield/2019-10-04/40nm-antennas_3fmol_10mW-532nM_1/161_302-Noice.csv'
    
    green = np.loadtxt(file_green, delimiter=',', skiprows =1)
    red = np.loadtxt(file_red, delimiter=',', skiprows =1)
    bkg_verde = np.loadtxt(file_bkg_green, delimiter=',', skiprows = 1)
    bkg_rojo = np.loadtxt(file_bkg_red, delimiter=',', skiprows =1)


#    bkg2 = np.mean(bkg_verde[:,1])
#    bkg1 = np.mean(bkg_rojo[:,1])
#    if bkg1 -10 < bkg2 < bkg1 +10:
##        print("vamo lo pibe (bkg bien)")
#        bkg = (bkg1+bkg2)/2
#    else:
#        print("El bkg es distinto", bkg1, bkg2)

    plt.figure(i)
    plt.grid()

#    plt.plot(green[:,1],'g')
#    plt.plot(red[:,1],'r')
#    
#    plt.plot(green[:,1]-bkg_verde[:,1],'g')
    plt.plot(red[:,1]-bkg_rojo[:,1], 'r')
#    plt.plot(bkg_verde[:,1], '--y')
#    plt.plot(bkg_rojo[:,1], linestyle = '--', color ='orange')
    plt.xlim=(0,9)


#    Idf = np.mean(green[0:20,1]-bkg_verde[0:20,1])
#    Id = np.mean(green[98:,1]-bkg_verde[98:,1])
##
#    Idf = np.mean(green[0:20,1]-bkg_verde[0:20,1])
#    Id  = np.max(green[:,1])-np.max(bkg_verde[:,1])

#    Ef.append((1-Idf/Id)*100)
#    print(Ef[i-1], "%")
#print("\"promediando\"",i,"veces; tengo Ef=", np.mean(Ef))



# %% algunos blinkings lindos

for i in range(1,3):
    file_green = "Blinking_verde_" + str(i) + ".txt"
    file_red = "Blinking_rojo_" + str(i) + ".txt"
    file_bkg_green = "Blinking_bkgverde_" + str(i) + ".txt"
    file_bkg_red = "Blinking_bkgrojo_" + str(i) + ".txt"


    green = np.loadtxt(file_green, skiprows =1)
    red = np.loadtxt(file_red, skiprows=1)
    bkg_verde = np.loadtxt(file_bkg_green, skiprows = 1)
    bkg_rojo = np.loadtxt(file_bkg_red, skiprows = 1)


    plt.figure(990+i)
    plt.grid()
    plt.plot(red[:,1]-bkg_rojo[:,1], 'r')
    plt.plot(green[:,1]-bkg_verde[:,1],'g')
    plt.plot(bkg_verde[:,1], '--y')
    plt.plot(bkg_rojo[:,1], linestyle = '--', color ='orange')

# %% Data del curso para los pibes INTERCALADA

for i in range(1,2):
    i=2
    file_green = "medicion_" + str(i) + "_verde" + ".txt"
    file_red = "medicion_" + str(i) + "_rojo" + ".txt"


    green = np.loadtxt(file_green, skiprows =1)
    red = np.loadtxt(file_red, skiprows=1)
    largo = len(green[:,1])
    bkg_verde = np.mean(green[largo-100:,1])
    bkg_rojo = np.mean(red[largo-100:,1])


    plt.figure(990+i)
    plt.grid()
    plt.plot((red[:,1]))#-bkg_rojo), 'r')
    plt.plot(green[:,1])#-bkg_verde,'g')
#    plt.plot(bkg_verde[:,1], '--y')
#    plt.plot(bkg_rojo[:,1], linestyle = '--', color ='orange')



# %% Data del curso para los pibes SOLO VERDE

for i in range(1,5):
#    i=1
    file_green = "medicion_" + str(i) + "_solo_verde" + ".txt"
    file_red = "medicion_" + str(i) + "_solo_rojo" + ".txt"


    green = np.loadtxt(file_green, skiprows =1)
    red = np.loadtxt(file_red, skiprows=1)
    largo = len(green[:,1])
    bkg_verde = np.mean(green[largo-50:,1])
    bkg_rojo = np.mean(red[largo-50:,1])


    plt.figure(220+i)
    plt.grid()
    plt.plot((red[:,1]-bkg_rojo), 'r')
    plt.plot(green[:,1]-bkg_verde,'g')

# %%
    
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import matplotlib.patches 
from PIL import Image

TIFF = np.array(Image.open('C:/Origami testing Widefield/2019-10-10/oriGreen_reference_AT542-2fmol-200pM-45min_incubation_finish-50mW_532nm-100ms_1/oriGreen_reference_AT542-2fmol-200pM-45min_incubation_finish-50mW_532nm-100ms_1_MMStack_Pos0.ome.tif'))

plt.figure(figsize=(10,10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

#size        = 500 #width and height of image in pixels
#peak_height = 100 # define the height of the peaks
#num_peaks   = 20
#noise_level = 50
threshold   = 102

np.random.seed(3)

#set up a simple, blank image (Z)
#x = np.linspace(0,size,size)
#y = np.linspace(0,size,size)

#X,Y = np.meshgrid(x,y)
#Z = X*0
Z=TIFF

##now add some peaks
#def gaussian(X,Y,xo,yo,amp=100,sigmax=4,sigmay=4):
#    return amp*np.exp(-(X-xo)**2/(2*sigmax**2) - (Y-yo)**2/(2*sigmay**2))
#
#for xo,yo in size*np.random.rand(num_peaks,2):
#    widthx = 5 + np.random.randn(1)
#    widthy = 5 + np.random.randn(1)
#    Z += gaussian(X,Y,xo,yo,amp=peak_height,sigmax=widthx,sigmay=widthy)

##of course, add some noise:
#Z = Z + scipy.ndimage.gaussian_filter(0.5*noise_level*np.random.rand(size,size),sigma=5)    
#Z = Z + scipy.ndimage.gaussian_filter(0.5*noise_level*np.random.rand(size,size),sigma=1)    




t = time.time() #Start timing the peak-finding algorithm

#Set everything below the threshold to zero:
Z_thresh = np.copy(Z)
Z_thresh[Z_thresh<threshold] = 0
print ('Time after thresholding: %.5f seconds'%(time.time()-t))

#now find the objects
labeled_image, number_of_objects = scipy.ndimage.label(Z_thresh)
print ('Time after labeling: %.5f seconds'%(time.time()-t))

peak_slices = scipy.ndimage.find_objects(labeled_image)
print ('Time after finding objects: %.5f seconds'%(time.time()-t))

def centroid(data):
    h,w = np.shape(data)   
    x = np.arange(0,w)
    y = np.arange(0,h)

    X,Y = np.meshgrid(x,y)

    cx = np.sum(X*data)/np.sum(data)
    cy = np.sum(Y*data)/np.sum(data)

    return cx,cy

centroids = []

for peak_slice in peak_slices:
    dy,dx  = peak_slice
    x,y = dx.start, dy.start
    cx,cy = centroid(Z_thresh[peak_slice])
    centroids.append((x+cx,y+cy))

print ('Total time: %.5f seconds\n' %(time.time()-t))

###########################################
#Now make the plots:
for ax in (ax1,ax2,ax3,ax4): ax.clear()
ax1.set_title('Original image')
ax1.imshow(Z,origin='lower')

ax2.set_title('Thresholded image')
ax2.imshow(Z_thresh,origin='lower')

ax3.set_title('Labeled image')
ax3.imshow(labeled_image,origin='lower') #display the color-coded regions

for peak_slice in peak_slices:  #Draw some rectangles around the objects
    dy,dx  = peak_slice
    xy     = (dx.start, dy.start)
    width  = (dx.stop - dx.start + 1)
    height = (dy.stop - dy.start + 1)
    rect = matplotlib.patches.Rectangle(xy,width,height,fc='none',ec='red')
    ax3.add_patch(rect,)

ax4.set_title('Centroids on original image')
ax4.imshow(Z,origin='lower')

for x,y in centroids:
    ax4.plot(x,y,'kx',ms=10)

#ax4.set_xlim(0,size)
#ax4.set_ylim(0,size)

plt.tight_layout
plt.show()

# %% Otro intento mas

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, io #,img_as_float

#im = img_as_float(data.coins())
FILE = 'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif'

tiff = io.imread(FILE)

im = tiff[1]

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=5)

# display results
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()

plt.show()




maxvalues = []
for i in range(len(coordinates[:,0])):
    maxvalues.append(im[coordinates[i,0],coordinates[i,1]])

#print(maxvalues)

nomax = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]

print( "\n \n", "len maxvalues",len(maxvalues),"\n len nomax", len(nomax))

plt.figure("que onda")
plt.title(len(maxvalues))
plt.imshow(im, cmap=plt.cm.gray)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')

for j in range(len(coordinates[:,1])):
    if j not in nomax:
        plt.plot(coordinates[j, 1], coordinates[j, 0], 'b.')

trace = {}

p = 0
for i in range(len(coordinates[:,0])):
    if i not in nomax:

        trace[p] = []
        for f in range(tiff.shape[0]):
            suming = 0
            deletear = []
            try:
                for x in range(-4,5): # goes from (-4,-4) to (4,4)
                    for y in range(-4,5):  # its 81 numbers. 9x9 box

                        suming += tiff[f][coordinates[i+x,0],coordinates[i+y,1]]
                trace[p].append(suming)
            except:
#                print("i",i, "f", f, "p",p,"trace[p]", trace[p])
                deletear.append(p)
#                trace[p].append(np.nan)

        p += 1

for d in deletear:
    print(d)
    del trace[d]

print("len trace",len(trace))
print("len maxvalues-nomax",len(maxvalues)-len(nomax))



#for j in range(5):
#    plt.plot(trace[j])
  
# %% To continue the previous traces.

Nframes = len(trace[0])
Ntraces = len(trace)


realmaxs = []
realmins = []

hipart = {}
lowpart = {}

lessmax = 0.97
moremin = 1.03

N = Ntraces

PLOT = True
graph = np.linspace(1, Ntraces-5, num=10, endpoint=False, dtype=int)
for i in range(N):
#    print(i)
    if PLOT and i in graph:
        plt.figure(i)
        plt.plot(trace[i], '*b')

    H = True
    L = True
    mm = []
    ml = []
    hipart[i] = [trace[i][0]]
    hipart[i].append(trace[i][1])
    
    lowpart[i] = [trace[i][-1]]
    for j in range(1, Nframes-3):

        if H:
            mm.append(trace[i][j])

            if (np.mean(np.array(mm)))*lessmax < trace[i][j+1] < (np.mean(np.array(mm)))*moremin:
                hipart[i].append(trace[i][j+1])

            else:
                if (np.mean(np.array(mm)))*lessmax < trace[i][j+2] < (np.mean(np.array(mm)))*moremin:
                    hipart[i].append(trace[i][j+1])
                else:
                    if (np.mean(np.array(mm)))*lessmax < trace[i][j+3] < (np.mean(np.array(mm)))*moremin:
                        hipart[i].append(trace[i][j+1])
                    else:
                        # Finished the high part
                        H = False

        if L:
            ml.append(trace[i][-j])

            if (np.mean(np.array(ml)))*lessmax < trace[i][-j-1] < (np.mean(np.array(ml)))*moremin:
                lowpart[i].append(trace[i][-j-1])
                
            else:
                if (np.mean(np.array(ml)))*lessmax < trace[i][-j-2] < (np.mean(np.array(ml)))*moremin:
                    lowpart[i].append(trace[i][-j-2])
                else:
                    if (np.mean(np.array(ml)))*lessmax < trace[i][-j-3] < (np.mean(np.array(ml)))*moremin:
                        lowpart[i].append(trace[i][-j-3])
                    else:
                        L = False


    fixedhighpartaux = np.array(hipart[i])
    fixedhighpart = np.concatenate((fixedhighpartaux, np.nan*np.zeros(Nframes-len(fixedhighpartaux))),axis=0)

    fixedlowpartaux = np.array(np.flip(lowpart[i]))
    fixedlowpart = np.concatenate((np.nan*np.zeros(Nframes-len(fixedlowpartaux)), fixedlowpartaux,),axis=0)

    realmaxs.append(np.mean(hipart[i]))
    realmins.append(np.mean(lowpart[i]))
    
    if PLOT and i in graph:
        plt.plot(fixedhighpart, '--r')
        plt.plot(fixedlowpart, '--g')    


finaldata = (np.array(realmaxs)-np.array(realmins))


plt.figure("histogram")
for i in [20, 16, 12, 8, 4]:
    print(i)
    plt.hist(finaldata, int(len(finaldata)/i))
plt.title((len(finaldata)))
plt.grid()
print(len(finaldata))



mu = np.mean(np.array(finaldata))
sigma = np.sqrt(((len(finaldata)-1)**(-1))*np.sum((finaldata-mu)**2))
#se = sigma/np.sqrt(Nframes)

print(mu,"+-", sigma)

plt.figure("histo/16")
plt.title((len(finaldata)))
plt.hist(finaldata, int(len(finaldata)/16), color='m')
plt.axvline(mu, linestyle=':', color='k')
plt.axvline(mu+sigma, linestyle='-.', color='r')
plt.axvline(mu-sigma, linestyle='-.', color='r')
plt.axvline(mu+2*sigma, linestyle='--', color='orange')
plt.axvline(mu-2*sigma, linestyle='--', color='orange')


fixeddata = np.copy(finaldata)
fixeddata[fixeddata>mu + (1.6*sigma)] = np.nan 
fixeddata[fixeddata<mu - (1.6*sigma)] = np.nan 
mu = np.nanmean(np.array(fixeddata))
sigma = np.sqrt(((np.count_nonzero(~np.isnan(fixeddata))-1)**(-1))*np.nansum((fixeddata-mu)**2))
#se = sigma/np.sqrt(Nframes)

print("valor final = ", mu,"+-", sigma)

plt.figure("histo centrado")
plt.hist(finaldata, int(len(finaldata)/16), color='b')
plt.title((np.count_nonzero(~np.isnan(finaldata))))
plt.axvline(mu, linestyle=':', color='k')
plt.axvline(mu+sigma, linestyle='-.', color='r')
plt.axvline(mu-sigma, linestyle='-.', color='r')
plt.axvline(mu+2*sigma, linestyle='--', color='orange')
plt.axvline(mu-2*sigma, linestyle='--', color='orange')


#%%
from take_traces import take_traces

import numpy as np
import matplotlib.pyplot as plt
#FILE = 'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif'
#
allfiles = ['C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif',
            'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_2/1Atto542_1640um_50mW_2_MMStack_Pos0.ome.tif',
            'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_3/1Atto542_1640um_50mW_3_MMStack_Pos0.ome.tif',
            'C:/Origami testing Widefield/2019-10-10/oriGreen_reference_AT542-2fmol-200pM-45min_incubation_finish-50mW_532nm-100ms_1/oriGreen_reference_AT542-2fmol-200pM-45min_incubation_finish-50mW_532nm-100ms_1_MMStack_Pos0.ome.tif']#,
#            'C:/Analizando Imagenes/Time Trace(s)-50mW_200ms-hand.csv']


LA_DATA = []
length = [0]
for i in range(len(allfiles)):
    LA_DATA = np.concatenate((LA_DATA, take_traces(allfiles[i], True)))
    length.append(len(LA_DATA)-length[i])
    print(len(LA_DATA))

plt.figure("LA_DATA")
for i in [20, 16, 12, 8, 4]:
    print(i)
    plt.hist(LA_DATA, int(len(LA_DATA)/i))
plt.title((len(LA_DATA)))
plt.grid()
print(len(LA_DATA))
    

mu = np.mean(np.array(LA_DATA))
sigma = np.sqrt(((len(LA_DATA)-1)**(-1))*np.sum((LA_DATA-mu)**2))
#se = sigma/np.sqrt(Nframes)

#print(mu,"+-", sigma)

plt.figure("histo antes")
plt.title((len(LA_DATA)))
plt.hist(LA_DATA, int(len(LA_DATA)/16), color='m')
plt.axvline(mu, linestyle=':', color='k')
plt.axvline(mu+sigma, linestyle='-.', color='r')
plt.axvline(mu-sigma, linestyle='-.', color='r')
plt.axvline(mu+2*sigma, linestyle='--', color='orange')
plt.axvline(mu-2*sigma, linestyle='--', color='orange')

fixealadata = np.copy(LA_DATA)
fixealadata[fixealadata>mu + (1.6*sigma)] = np.nan 
fixealadata[fixealadata<mu - (1.6*sigma)] = np.nan 

mu2 = np.nanmean(np.array(fixealadata))
sigma2 = np.sqrt(((np.count_nonzero(~np.isnan(fixealadata))-1)**(-1))*np.nansum((fixealadata-mu)**2))

#print(mu2,"+-", sigma2)



plt.figure("histo centrado")
plt.hist(LA_DATA, int(len(LA_DATA)/16), range=(mu2-3*sigma2, mu2+3*sigma2), color='b')
plt.title((len(LA_DATA)))
plt.axvline(mu2, linestyle=':', color='k')
plt.axvline(mu2+sigma2, linestyle='-.', color='r')
plt.axvline(mu2-sigma2, linestyle='-.', color='r')
plt.axvline(mu2+2*sigma2, linestyle='--', color='orange')
plt.axvline(mu2-2*sigma2, linestyle='--', color='orange')

print("mu", mu, "±", sigma, "sigma")
print("\n mu2", mu2, "±", sigma2, "sigma2")


# %% Ejemplo para mirar las trazas de a una

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
ax1.set(facecolor='#FFFFCC')

x = np.arange(0.0, 5.0, 0.01)
y = np.sin(2*np.pi*x) + 0.5*np.random.randn(len(x))

ax1.plot(x, y, '-')
ax1.set_ylim(-2, 2)
ax1.set_title('Press left mouse button and drag to test')

#ax2.set(facecolor='#FFFFCC')
line2, = ax2.plot(x, y, '-')


def onselect(xmin, xmax):
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x) - 1, indmax)

    thisx = x[indmin:indmax]
    thisy = y[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw()

span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
# Set useblit=True on most backends for enhanced performance.


plt.show()

