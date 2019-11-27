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

selection1 = np.loadtxt('C:/Analizando Imagenes/Single_molecule/selection_Green_reference_11.3mW_60swait_traces-51(0).txt')
selection2 = np.loadtxt('C:/Analizando Imagenes/Single_molecule/selection_Green_reference_11.3mW_traces-135(1).txt')
#
#selection1 = np.loadtxt('C:/Analizando Imagenes/Single_molecule/selection_Green_reference_7.8mW_60swait_traces-74(2).txt')
#selection2 = np.loadtxt('C:/Analizando Imagenes/Single_molecule/selection_Green_reference_7.8mW_traces-108(3).txt')

#selection1 = np.loadtxt('C:/Analizando Imagenes/Single_molecule/selection_Green_reference_2.8mW_60swaiti_traces-78(0).txt')
#selection2 = np.loadtxt('C:/Analizando Imagenes/Single_molecule/selection_Green_reference_2.8mW_traces-117(1).txt')
##
selection = np.concatenate((selection1, selection2))


#plt.hist(selection[:, 5], len(selection[:, 5])//3)
nozeros = selection[np.nonzero(selection[:, 5]),5][0]
plt.hist(nozeros, len(nozeros)//N)

N = len(nozeros)//14

hist, bin_edges = np.histogram(nozeros, len(nozeros)//N)
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

N=N

y,x,_ = plt.hist(nozeros,len(nozeros)//N,alpha=.3,label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected = (60,7,15,85,7,13)  # for 11.3 mW
#expected = (75,7,0.5,50,7,13)  # for 7.8mW
#expected = (10,7,0.5,15,7,13)  # for 2.8mW
params,cov = curve_fit(bimodal,x,y,expected)
sigma = np.sqrt(np.diag(cov))
plt.plot(x,bimodal(x,*params),color='red',lw=3,label='model')
plt.legend()
plt.vlines((params[0], params[3]), color='m', ymin=0,ymax=10)
print(params,'\n',sigma)
#print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
          "\n",(params[3],"±",params[4]), "*", params[5])
# %%
