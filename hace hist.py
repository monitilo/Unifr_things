# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:50:12 2019

@author: ChiarelG
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

# %%
alldata = ["C:/Analizando Imagenes/Power dependent/Quick and bad/Green_reference_2.8mW_intensities447(4).txt",
           "C:/Analizando Imagenes/Power dependent/Quick and bad/Green_reference_4.5mW_intensities1284(0).txt",
           "C:/Analizando Imagenes/Power dependent/Quick and bad/Green_reference_6.0mW_intensities1025(1).txt",
           "C:/Analizando Imagenes/Power dependent/Quick and bad/Green_reference_7.8mW_intensities731(2).txt",
           "C:/Analizando Imagenes/Power dependent/Quick and bad/Green_reference_11.3mW_intensities582(3).txt"]

data = dict()
for i in range(len(alldata)):
    data[i] = np.loadtxt(alldata[i])

# %%

N = 8  #  cantidad de histogramas


histoarray = [N] #  np.linspace(30,10,N, dtype=int)
#for i in range(len(alldata)):
i=0
plt.figure(alldata[i])
#    media = np.mean(data[i])
#    sigma = np.std(data[i])
#    print(media, sigma)
for j in histoarray:
    print(j, len(data[i]))
    plt.hist(data[i], len(data[i])//j)
#        plt.vlines(media, color='k', ymin=0, ymax=np.max(data[i]))
#        plt.vlines((media-sigma, media+sigma),color='r', ymin=0, ymax=np.max(data[i]))

hist, bin_edges = np.histogram(data[i], len(data[i])//j)
x = hist
y = np.copy(x)
for j in range(len(bin_edges)-1):
    y[j] = bin_edges[j] + (bin_edges[j+1]- bin_edges[j])/2

x = np.copy(y)
y = hist
#x = ar(range(10))
#y = ar([0,1,2,3,4,5,4,3,2,1])

n = len(x)                          #the number of data
mean = sum(x*y)/sum(y)                   #note this correction
sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
perr = np.sqrt(np.diag(pcov))

plt.plot(x,y,'b+:',label='data')
plt.plot(x,gaus(x,*popt),'ro:',label='fit')
plt.vlines(popt[1], ymin=0,ymax=10)
plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
plt.legend()
plt.title('hist'+alldata[i])
plt.xlabel('Counts kHz')
plt.ylabel("mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
#    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
print(popt[1], popt[2])
plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
plt.show()

viejopopt = popt
# %%

#selection1 = np.loadtxt('C:/Analizando Imagenes/Power dependent/Quick and bad/selection_Green_reference_11.3mW_60swait_traces-51(0).txt')
#selection2 = np.loadtxt('C:/Analizando Imagenes/Power dependent/Quick and bad/selection_Green_reference_11.3mW_traces-135(1).txt')

#selection1 = np.loadtxt('C:/Analizando Imagenes/Power dependent/Quick and bad/selection_Green_reference_7.8mW_60swait_traces-74(2).txt')
#selection2 = np.loadtxt('C:/Analizando Imagenes/Power dependent/Quick and bad/selection_Green_reference_7.8mW_traces-108(3).txt')

#selection1 = np.loadtxt('C:/Analizando Imagenes/Power dependent/Quick and bad/selection_Green_reference_2.8mW_60swaiti_traces-78(0).txt')
#selection2 = np.loadtxt('C:/Analizando Imagenes/Power dependent/Quick and bad/selection_Green_reference_2.8mW_traces-117(1).txt')

#selection = np.concatenate((selection1, selection2))


#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(5)1.2mW_1_traces-31(0).txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(5)1.2mW_2_traces20.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(5)1.2mW_3_traces56.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(5)1.2mW_4_traces84.txt',]

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(7)2.0mW_1_traces-31.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(7)2.0mW_2_traces-43.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(7)2.0mW_3_traces-28.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(7)2.0mW_4_traces-45.txt']

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_1_traces-18.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_2_traces-23.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_3_traces-28.txt']

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_1_traces-70.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_2_traces-24.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_3_traces-39.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_4_traces-37.txt']

ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_1_traces-47.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_2_traces-40.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_3_traces-32.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_4_traces-29.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_5_traces-19.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_6_traces-20.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_7_traces-32.txt',
          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_8_traces-33.txt',]


selection = np.loadtxt(ladata[0])
for i in range(1,len(ladata)):
    print(len(selection))
    loaded = np.loadtxt(ladata[i])
    selection = np.concatenate((selection, loaded))


#plt.hist(selection[:, 5], len(selection[:, 5])//3)
nozeros = selection[np.nonzero(selection[:, 5]),5][0]

#nozeros = nozeros[np.where(nozeros > 8)]

N = len(nozeros)//8

plt.hist(nozeros, len(nozeros)//N)



y,x = np.histogram(nozeros, len(nozeros)//N)
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

n = len(x)                          #the number of data
mean = sum(x*y)/sum(y)                   #note this correction
sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
perr = np.sqrt(np.diag(pcov))


#plt.plot(x,y,'b+:',label='data')
plt.plot(x,gaus(x,*popt),'ro:',label='1G fit')
plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
#plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
plt.legend()
plt.title('hist')
plt.xlabel('Counts kHz')
plt.ylabel("total points ={}//{}".format(len(nozeros), N))
#    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
print(popt[1], popt[2])
#plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
plt.show()

## %%

#N=N

y,x,_ = plt.hist(nozeros,len(nozeros)//N,alpha=.3,label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

#expected = (60,7,15,85,7,13)  # for 11.3 mW
#expected = (75,7,0.5,50,7,13)  # for 7.8mW
#expected = (10,7,0.5,15,7,13)  # for 2.8mW

#expected = (popt[1],abs(popt[2]),popt[0],15,5,1)  # 1.2mW N~20
#expected = (popt[1],abs(popt[2]),popt[0],
#            1.8*popt[1], 1.5*abs(popt[2]), 0.5*popt[0])  # 2.0mW N~16 (dos picos, ajusta 1 solo)

#expected = (popt[1],abs(popt[2]),popt[0],
#            2*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 3.5mW N~8 (nada funciona)

#expected = (popt[1],abs(popt[2]),popt[0],
#            2*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 6.3mW N~12

expected = (popt[1],abs(popt[2]),popt[0],
            0.4*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 6.3mW N~12

params,cov = curve_fit(bimodal,x,y,expected)
sigma = np.sqrt(np.diag(cov))
plt.plot(x,bimodal(x,*params),color='red',lw=3,label='2D model')
plt.legend()
plt.vlines((params[0], params[3]), color='m', ymin=0,ymax=0.5*popt[0])
print(params,'\n',sigma)
#print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
          "\n",(params[3],"±",params[4]), "*", params[5])
# %%
