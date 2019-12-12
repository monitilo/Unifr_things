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

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_1_traces-47.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_2_traces-40.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_3_traces-32.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_4_traces-29.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_5_traces-19.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_6_traces-20.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_7_traces-32.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(20)9.1W_8_traces-33.txt',]

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_1_traces-14.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_2_traces-41.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_3_traces-31.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_4_traces-31.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_5_traces-31.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_6_traces-35.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_7_traces-31.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_8_traces-30.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_9_traces-42.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/selection_ATTO542_1fmol-Pol_circular-532nm_(25)12.0W_10_traces-72.txt']

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_4_traces-14.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_5_traces-14.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_6_traces-16.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_7_traces-19.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(10)3.5mW_8_traces-22.txt']

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_5_traces-18.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_6_traces-22.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_7_traces-14.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_8_traces-19.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(15)6.3mW_9_traces-12.txt']

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(20)8.7mW_9_traces-10.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(20)8.7mW_10_traces-11.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(20)8.7mW_11_traces-15.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(20)8.7mW_12_traces-12.txt',
#          'C:/Analizando Imagenes/Power dependent/New Run circular/New day/selection_ATTO542_1fmol-Pol_circular-532nm_(20)8.7mW_13_traces-18.txt']

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/New day/ATTO542_1fmol-Pol_circular-532nm_(25)11.6W_histogram-99.txt']

#ladata = ['C:/Analizando Imagenes/Single_molecule/Test_histogram-67.txt', # only from one image
#          'C:/Analizando Imagenes/Single_molecule/Test_histogram-72.txt', # only from one image
#ladata = ['C:/Analizando Imagenes/Single_molecule/Test_histogram-351.txt', # only from one image
ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/New day 2 small powers/ATTO542_1fmol-Pol_circular-532nm_(3)0.5W_todos_histogram-88.txt']

#ladata = ['C:/Analizando Imagenes//Power dependent/New Run circular/New day 2 small powers/ATTO542_1fmol-Pol_circular-532nm_(5)1.2W_alldata_histogram-152.txt'] 

#ladata = ['C:/Analizando Imagenes/Power dependent/New Run circular/New day 2 small powers/ATTO542_1fmol-Pol_circular-532nm_(7)2.0W_alldata_histogram-137.txt']


#ladata=['C:/Analizando Imagenes/Single_molecule/1Spot, powers from0.5 to 6.2_histogram-5.txt']
#ladata=['C:/Analizando Imagenes/Single_molecule/Place1-1spot-5_powers_from0.5 to 6.2_histogram-5.txt']
ladata=['C:/Analizando Imagenes/Single_molecule/Place2-1spot-5_powers_from0.5 to 6.2_histogram-4.txt']

x = np.array([ 0.5, 1.2, 2.0, 3.5, 6.2])  # powers
y3 = np.array([ 5.55125   , 11.35770833, 17.75317708, 34.60130208, 53.5978125 ])  # one single spot alive. from place 3
#y1 = np.array([ 5.8315625 , 12.05578125, 11.98088542, 16.62135417, 31.26265625])  # one single spot alive. from place 1
y1 = np.array([ 5.8315625 , 12.05578125, 19.59, 16.62135417, 31.26265625])  # one single spot alive. from place 1
y2 = np.array([ 6.54822917, 12.83020833, 21.851546875 ])
m2 = ((y2[2]-y2[0]) / (x[2]-x[0]))
m1 = ((y1[-1]-y1[-2]) / (x[-1]-x[-2]))
m1_c = ((y1[2]-y1[0]) / (x[2]-x[0]))
m3 = ((y3[-1]-y3[0]) / (x[-1]-x[0]))
b1_c = 0 # m1_c*x[1] - y1[1]
b1 = 0 # m1*x[-1] - y1[-1]
b3 = 0 # m3*x[1] - y3[1]
plt.plot(x,y1, '-ob', label="Place 1", lw=0.3)
plt.plot(x,y3, '-ok', label="Place3", lw=0.3)
plt.plot(x[:3],y2, '-oc', label="Place 2", lw=0.3)
plt.plot(x, m2*x, color='m', label="fit P2")
#plt.plot(x, m1*x+b1, 'r', label="fit all line (P1)")
plt.plot(x, m1_c*x+b1_c, '--r', label="fit first 3 points(P1)")
plt.plot(x, m3*x+b3, color='orange', label="fit P3")
plt.xlabel("Power (mW)")
plt.ylabel("Counts kHz")
plt.legend()

trazas = True
trazas = False


selection = np.loadtxt(ladata[0])
for i in range(1,len(ladata)):
    print(len(selection))
    loaded = np.loadtxt(ladata[i])
    selection = np.concatenate((selection, loaded))

if trazas:
    plt.hist(selection[:, 5], len(selection[:, 5])//3)
    nozeros = selection[np.nonzero(selection[:, 5]),5][0]
#    nozeros = nozeros[np.where(nozeros > 2)]

else:
    nozeros = selection[np.nonzero(selection)]
#    nozeros = nozeros[np.where(nozeros > 1)]
#    nozeros = nozeros[np.where(nozeros < 35)]

#8, 10*, 13
N = 8
# len(nozeros)//12

#bins = np.linspace(0,int(np.max(histo)), N)
bins = np.linspace(0,int(np.max(nozeros)), N)

plt.hist(nozeros, bins=bins, alpha = 0.5)# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
#plt.bar(x,y,20)
#plt.plot(x,y,'r')
#x = np.array(Histo[:,0], dtype=float)
#y = np.array(Histo[:-1,1], dtype=float)
y,x = np.histogram(nozeros, bins=bins)  #len(nozeros)//N
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

n = len(x)                          #the number of data
mean = sum(x*y)/sum(y)                   #note this correction
sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
perr = np.sqrt(np.diag(pcov))


#plt.plot(x,y,'b+:',label='data')
X = np.linspace(x[0], x[-1], 500)
plt.plot()
plt.plot(X,gaus(X,*popt),'g',lw=2, label='1G fit')
plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
#plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
plt.legend()
plt.title('hist')
plt.xlabel('Counts kHz')
plt.ylabel("total points ={} in {} bins".format(len(nozeros), len(nozeros)//N))
#    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
print(popt[1], popt[2], "1")
#plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
#plt.xlim(0, int(np.max(nozeros)))
plt.show()

## %%

#N=N

#y,x,_ = plt.hist(nozeros,len(nozeros)//N,alpha=.3,label='data')
#x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

#expected = (60,7,15,85,7,13)  # for 11.3 mW
#expected = (75,7,0.5,50,7,13)  # for 7.8mW
#expected = (10,7,0.5,15,7,13)  # for 2.8mW

#expected = (popt[1],abs(popt[2]),popt[0],15,5,1)  # 1.2mW N~20

#expected = (popt[1],abs(popt[2]),popt[0],
#            2*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 2.0mW N~16 (dos picos, ajusta 1 solo)

#expected = (popt[1],abs(popt[2]),popt[0],
#            2*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 3.5mW N~8 (nada funciona)

#expected = (popt[1],abs(popt[2]),popt[0],
#            2*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 6.3mW N~12

#expected = (0.5*popt[1],0.5*abs(popt[2]),0.3*popt[0],
#            1*popt[1], 0.5*abs(popt[2]), popt[0])  # 9.1

expected = (popt[1],abs(popt[2]),popt[0],
            1.5*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 

#expected = (29,5,8, 40,5,8)  # 

#expected = (40,5,8, 70,5,8)  # 

expected = (11,2,8, 3,2,1)  # 

try:
#    params,cov = curve_fit(bimodal,x,y,expected)
#    sigma = np.sqrt(np.diag(cov))
    if params[0] < 0 or params[3] < 0:
        print("bad adjusts 2G")
    else:
        #X = np.linspace(x[0]-50, x[-1]+50, 5000)
        plt.plot(X,bimodal(X,*params),color='orange',lw=3,label='2G model')
        plt.legend()
        plt.vlines((params[0], params[3]), color=('r','b'), ymin=0,ymax=0.5*popt[0])
    print(params,'\n',sigma)
    #print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
except:
    params = ["no"]*6
print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
          "\n",(params[3],"±",params[4]), "*", params[5])


