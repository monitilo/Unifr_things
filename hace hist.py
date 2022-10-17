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
plt.plot(x,y1, '-ob', label="Place 1", lw=1.3)
plt.plot(x[:3],y2, '-oc', label="Place 2", lw=1.3)
plt.plot(x,y3, '-ok', label="Place 3", lw=1.3)

#plt.plot(x, m2*x, color='m', label="fit P2")
#plt.plot(x, m1*x+b1, 'r', label="fit all line (P1)")
#plt.plot(x, m1_c*x+b1_c, '--r', label="fit first 3 points(P1)")
plt.plot(x, m3*x+b3, color='orange', label="fit P3", lw='0.7')
plt.xlabel("Power (mW)")
plt.ylabel("Counts kHz")
plt.legend()

trazas = True
trazas = False
#%% DIMER ramp 27.01 by hand
x1 = np.array([ 0.6, 1.4, 3.5, 4.2, 5.7, 7.4, 8.9])  # powers
y1 = np.array([ 1391, 2690, 6918, 7899, 12174, 15668, 16323])/100
y2 = np.array([ 1386, 2933, 7194, 7917, 11039, 13715, 14904])/100
ybg = np.array([ -64, -52, -0.1, 169, 64, -6, 26])/100
m1 = ((y1[-1]-y1[0]) / (x1[-1]-x1[0]))
m2 = ((y2[-1]-y2[0]) / (x1[-1]-x1[0]))
mbg = ((ybg[-1]-ybg[0]) / (x1[-1]-x1[0]))
plt.plot(x1,y1, '-ob', label="Place1", lw=1.3)
plt.plot(x1,y2, '-og', label="Place1", lw=1.3)
plt.plot(x1,ybg, '--', label="Background", lw=1.3)

plt.plot(x1, m1*x1, color='m', label="fit1")
plt.plot(x1, m2*x1, color='c', label="fit2")
plt.plot(x1, mbg*x1, color='r', label="fitbg", lw=0.5)

b1 = y1[-1] - m1*x1[-1]
b2 =y2[-1] - m2*x1[-1]


plt.xlabel("Power (mW)")
plt.ylabel("Counts kHz")
#plt.xlim((0,0.8))
#plt.ylim((0,25))
plt.legend()
plt.show()

print(m1,m2, "\n", b1, b2)

#%% MONOMER ramp 27.01 by hand
x1 = np.array([ 0.6, 1.4, 3.5, 4.2, 5.7, 7.4, 8.9])  # powers
x2 = np.array([ 0.6, 1.4, 3.5, 4.2, 5.7, 7.4])  # powers
y1 = np.array([ 786, 1258, 3374, 4123, 5134, 8363, 9685])/100
y2 = np.array([ 498, 846, 2015, 2337, 2700, 3909])/100
m1 = ((y1[-1]-y1[0]) / (x1[-1]-x1[0]))
m2 = ((y2[-1]-y2[0]) / (x1[-1]-x1[0]))
mbg = ((ybg[-1]-ybg[0]) / (x1[-1]-x1[0]))
plt.plot(x1,y1, '-ob', label="Place1", lw=1.3)
plt.plot(x2,y2, '-og', label="Place1", lw=1.3)

plt.plot(x1, m1*x1, color='m', label="fit1")
plt.plot(x2, m2*x2, color='c', label="fit2")

b1 = y1[-1] - m1*x1[-1]
b2 =y2[-1] - m2*x2[-1]


plt.xlabel("Power (mW)")
plt.ylabel("Counts kHz")
#plt.xlim((0,0.8))
#plt.ylim((0,25))
plt.legend()
plt.show()

print(m1,m2, "\n", b1, b2)
# %% Histo por imagenes

import numpy as np
import matplotlib.pyplot as plt

import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

ladata = ['C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_532nm_4mW_moving_1_histogram-444.txt']
ladata = ['C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_532nm_4mW_moving_final frames_histogram-121.txt']
ladata = ['C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_640nm_7.7mW_moving_satrting frames_histogram-149.txt']
ladata = ['C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_640nm_7.7mW_moving_1_histogram-395.txt']
# =============================================================================
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_0.6mW_histogram-129.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_1.4mW_histogram-121.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_2.0mW_histogram-130.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_3.5mW_histogram-180.txt']
# #ladata =['C:/Analizando Imagenes/Single_molecule/Dimer_5.0mW(5.7last)_histogram-128.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_7.4mW_histogram-7.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_8.9mW_histogram-39.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer_0.7mW_histogram-26.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer_1.4mW_histogram-41.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer_2.0mW_histogram-45.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer_3.5mW_histogram-45.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer_5.0mWTOTAL_histogram-90.txt']
# #ladata = [ 'C:/Analizando Imagenes/Single_molecule/Monomer_8.9mW_histogram-38.txt']
# 
# #ladata =['C:/Analizando Imagenes/Single_molecule/ATTO_3.5_histogram-78.txt']
# 
# #ladata =['C:/Analizando Imagenes/Single_molecule/atto3.5mW-from02_12_19_histogram-77.txt']
# #
# ladata =['C:/Analizando Imagenes/Single_molecule/atto3.5mW-NEW_histogram-186.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer3.5mW-NEW_histogram-109.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer3.5mW-NEW_histogram-146 - +21.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Agdim3.5mW_histogram-92.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/oldsambple-3.5mW_histogram-63.txt']
# #
# #ladata = ['C:/Analizando Imagenes/Single_molecule/oldSample2-3.5mW_histogram-204.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer_0.6mW_histogram-129.txt',
# #           'C:/Analizando Imagenes/Single_molecule/Dimer_1.4mW_histogram-121.txt',
# #           'C:/Analizando Imagenes/Single_molecule/Dimer_2.0mW_histogram-130.txt',
# #           'C:/Analizando Imagenes/Single_molecule/Dimer_3.5mW_histogram-180.txt',
# #           'C:/Analizando Imagenes/Single_molecule/Dimer_5.0mW(5.7last)_histogram-128.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/1-24HBCy5_1.0mW_histogram-9.txt']
# ladata = [ 'C:/Analizando Imagenes/Single_molecule/1-24HBCy5_2.0mW_histogram-24.txt']
# #ladata =['C:/Analizando Imagenes/Single_molecule/2-24HBCy5@SiO_1_2.0mW_histogram-70.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/F-atto542-3.5mW_histogram-166.txt']
# ladata = ['C:/Analizando Imagenes/Single_molecule/attoold_nolabeled-redglitch-3.5mW_histogram-158.txt']
# ladata = ['C:/Analizando Imagenes/Single_molecule/attoold_nolabeled-redglitch-3.5mW_histogram-111.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/oldsambple-3.5mW_histogram-63 +old2 204 +redglitch 158 .txt']
# 
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_1000mM_Dimer_3.5mW_2_histogram-145.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_1000mM_Dimer_3.5mW_histogram-102.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_1000mM_Dimer_3.5mW_2_histogram-145 + 102.txt']
# 
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_500mM_Dimer_3.5mW_2_histogram-132.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_500mM_Dimer_3.5mW_histogram-136.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_500mM_Dimer_3.5mW_2_histogram-132 + 136.txt']
# #ladata = ['C:/Analizando Imagenes/Single_molecule/NaCl_0_Dimer_3.5mW_histogram-79.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/Atto542_alone_3.4mW_50ms_histogram-196 +125.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/OriT-cy5-640nm-6.2mW-TIRF_5_histogram-72.txt']
# 
# ladata= ['C:/Analizando Imagenes/Single_molecule/OriT-cy5-640nm-7.0mW_8_histogram-152.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/OriT-cy5-640nm-7.0mW_ROI15_histogram-79.txt']
# 
# #ladata = ['C:/Analizando Imagenes/Single_molecule/4-Cy5-PLL1_100-640nm_0.9mW_7_histogram-242 + TODO.txt']
# ladata = ['C:/Analizando Imagenes/Single_molecule/3-Cy5@SiO2-2-PLL1_100-640nm_0.9mW_9_histogram-318.txt']
# ladata = ['C:/Analizando Imagenes/Single_molecule/3-Cy5@SiO2-2-PLL1_100-640nm_2.3mW_7_histogram-303.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/No scavenger cy5-origami_histogram-32.txt']
# 
# #25.05.2020
# ladata = ['C:/Analizando Imagenes/Single_molecule/1_cy5alone_PLL_2.1mW_640nm _10_histogram-658.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_5_histogram-271.txt']
# 
# ladata = ['C:/Analizando Imagenes/Single_molecule/1_cy5alone_PLL_2.1mW_640nm _10_histogram-658.txt']
# 
# =============================================================================
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
#    nozeros = nozeros[np.where(nozeros > 30)]
#    nozeros = nozeros[np.where(nozeros < 50)]

#8, 10*, 13
N = 20
# len(nozeros)//12

#bins = np.linspace(0,int(np.max(histo)), N)
bins = np.linspace(0,int(np.max(nozeros)), N)

plt.hist(nozeros, bins=bins, alpha = 0.5, label = ladata[0][-25:-4])# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
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
plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
plt.legend()
plt.title('hist')
plt.xlabel('Counts kHz')
plt.ylabel("total points ={} in {} bins".format(len(nozeros), N))
#    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
print(popt[1], popt[2], "1")
#plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
#plt.xlim(0, int(np.max(nozeros)))
#plt.show()

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
    params,cov = curve_fit(bimodal,x,y,expected)
    sigma = np.sqrt(np.diag(cov))
    a=vaca
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
# %% otra pegada de codigo para varios hist a la vez:

files = ['C:/Analizando Imagenes/Single_molecule/oldsambple-3.5mW_histogram-63 +old2 204 +redglitch 158 .txt',
#         'C:/Analizando Imagenes/Single_molecule/attoold_nolabeled-redglitch-3.5mW_histogram-158.txt',
         'C:/Analizando Imagenes/Single_molecule/F-atto542-3.5mW_histogram-166.txt',
         'C:/Analizando Imagenes/Single_molecule/New_sandwitch_3.5mW_histogram-264.txt',
#         'C:/Analizando Imagenes/Single_molecule/oldsambple-3.5mW_histogram-63 +old2 204.txt',
#         'C:/Analizando Imagenes/Single_molecule/oldsambple-3.5mW_histogram-63.txt',
#         'C:/Analizando Imagenes/Single_molecule/oldSample2-3.5mW_histogram-204.txt',
         'C:/Analizando Imagenes/Single_molecule/ATTO_3.5_histogram-135 - +NEW 186.txt']
#         'C:/Analizando Imagenes/Single_molecule/atto3.5mW-NEW_histogram-186.txt']

#powers = ["old R&2G", "old 1 atto542 sandwich", "NEW atto542 sandwich", "labtek"]
powers = ["532 moving", "532 low", "647N low", "647N moving"]

files = ['C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_532nm_4mW_moving_1_histogram-444.txt',
         'C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_532nm_4mW_moving_final frames_histogram-121.txt',
         'C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_640nm_7.7mW_moving_satrting frames_histogram-149.txt',
         'C:/Analizando Imagenes/Single_molecule/4_boat-incubating-allweekend_640nm_7.7mW_moving_1_histogram-395.txt']

data = dict()
largos = []
for i in range(len(files)):
    data[i] = np.loadtxt(files[i])
    largos.append(len(data[i]))

largos = np.array(largos)

gaussplot = True #False  # True


fits = dict()

a = 0

#for s in structures:

bines = 9
auxe = largos / bines
largos/auxe

nbins = auxe #[10]*4 # [10, 10, 10, 10, 10, 10, 10]
minimun = [0, 0, 0, 0, 0, 0, 0]
maximun = [200, 200, 900, 1100, 110, 1100, 1100]

plt.figure()
for i in range(len(powers)):

    nozeros = data[i]

#    nozeros = dimerdata[powers[0]]
#    for j in powers[1:]:
#        nozeros = np.concatenate((nozeros, dimerdata[j]))
    #nozeros = selection
    #nozeros = selection[np.nonzero(selection)]

#    nozeros = nozeros[np.where(nozeros > minimun[i])]
#    nozeros = nozeros[np.where(nozeros < maximun[i])]

    N = nbins[i]

    bins = np.linspace(0,int(np.max(nozeros)), N)

    plt.hist(nozeros, bins=bins, alpha = 0.5, label=(powers[i])+"-"+str(len(nozeros)))# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N

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

    #plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
    plt.legend()
    plt.title('Histogram 3.5mW diferent samples')# for {} mW'.format(powers[i]))
    plt.xlabel('Counts (kHz)')
    plt.ylabel(" points ")#total points ={} in {} bins".format(len(nozeros), N))
    #    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
    print("\n"+str(powers[i]), " "+str(len(nozeros)))
    print(popt[1], popt[2], "1")
    fits[powers[i]] = (popt[1], popt[2])
#    plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
#    plt.xlim(0, 120)
#    plt.ylim(0,200)
    #plt.show()
    
    if gaussplot:
        plt.plot(X,gaus(X,*popt),'g',lw=2)  #, label='1G fit')
        plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
        def gauss(x,mu,sigma,A):
            return A*exp(-(x-mu)**2/2/sigma**2)
        
        def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
            return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
        
        expected = (popt[1],abs(popt[2]),popt[0],
                    1.5*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 
        
        #expected = (11,2,8, 3,2,1)  # 


        try:
            params,cov = curve_fit(bimodal,x,y,expected)
            sigma = np.sqrt(np.diag(cov))
            a=vaca
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

        print(" 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
#        print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
#                  "\n",(params[3],"±",params[4]), "*", params[5])
plt.show()

#print(fits)
# %% repito todo porque me encanta seguir copiando y pegando codigo.... Para antennas


powers = ["0.6", "1.4", "2.0", "3.5", "5.7", "7.5", "8.9"]
filesD = ['C:/Analizando Imagenes/Single_molecule/Dimer_0.6mW_histogram-129.txt',
         'C:/Analizando Imagenes/Single_molecule/Dimer_1.4mW_histogram-121.txt',
         'C:/Analizando Imagenes/Single_molecule/Dimer_2.0mW_histogram-130.txt',
         'C:/Analizando Imagenes/Single_molecule/Dimer_3.5mW_histogram-180.txt',
#         'C:/Analizando Imagenes/Single_molecule/Dimer3.5mW-NEW_histogram-109.txt',
         'C:/Analizando Imagenes/Single_molecule/Dimer_5.0mW(5.7last)_histogram-128.txt',
         'C:/Analizando Imagenes/Single_molecule/Dimer_7.4mW_histogram-7.txt',
         'C:/Analizando Imagenes/Single_molecule/Dimer_8.9mW_histogram-81.txt']

filesM = ['C:/Analizando Imagenes/Single_molecule/Monomer_0.7mW_histogram-26.txt',
         'C:/Analizando Imagenes/Single_molecule/Monomer_1.4mW_histogram-41.txt',
         'C:/Analizando Imagenes/Single_molecule/Monomer_2.0mW_histogram-45.txt',
         'C:/Analizando Imagenes/Single_molecule/Monomer_3.5mW_histogram-45.txt',
#         'C:/Analizando Imagenes/Single_molecule/Monomer3.5mW-NEW_histogram-146 - +21.txt',
         'C:/Analizando Imagenes/Single_molecule/Monomer_5.0mWTOTAL_histogram-90.txt',
         'C:/Analizando Imagenes/Single_molecule/Monomer_7.4mW_histogram-16.txt',
         'C:/Analizando Imagenes/Single_molecule/Monomer_8.9mW_histogram-38.txt']

filesAT = ['C:/Analizando Imagenes/Single_molecule/ATTO_0.7mW_histogram-153.txt',
           'C:/Analizando Imagenes/Single_molecule/ATTO_1.4mW_histogram-111.txt',
           'C:/Analizando Imagenes/Single_molecule/ATTO_0.7mW_histogram-153.txt',
#           'C:/Analizando Imagenes/Single_molecule/ATTO_4.2mW_histogram-133.txt',
#           'C:/Analizando Imagenes/Single_molecule/atto3.5mW-NEW_histogram-186.txt',
           'C:/Analizando Imagenes/Single_molecule/ATTO_3.5_histogram-135.txt',
           'C:/Analizando Imagenes/Single_molecule/ATTO_5.7mW_histogram-116.txt',
           'C:/Analizando Imagenes/Single_molecule/ATTO_7.5mW_histogram-76.txt',
           'C:/Analizando Imagenes/Single_molecule/ATTO_8.9mW_histogram-116.txt']

#ladata =['C:/Analizando Imagenes/Single_molecule/atto3.5mW-NEW_histogram-186.txt']

#ladata = ['C:/Analizando Imagenes/Single_molecule/Dimer3.5mW-NEW_histogram-109.txt']

#ladata = ['C:/Analizando Imagenes/Single_molecule/Monomer3.5mW-NEW_histogram-146 - +21.txt']


dimerfiles = dict()
dimerdata = dict()
monomerfiles = dict()
monomerdata = dict()
attofiles = dict()
attodata = dict()
largos = dict()
j=0
for i in powers:
    dimerfiles[i] = filesD[j]
    dimerdata[i] = np.loadtxt(dimerfiles[i])
    monomerfiles[i] = filesM[j]
    monomerdata[i] = np.loadtxt(monomerfiles[i])
    attofiles[i] = filesAT[j]
    attodata[i] = np.loadtxt(attofiles[i])
    j+=1

gaussfit = False  # True
structures = ["DIMER", "MONOMER", "ATTO"]
fits = dict()

todaladata = dict()
for s in structures:
    largos[s] = []

for i in powers:
    for s in structures:
        if s == "DIMER":
            todaladata[s, i] = dimerdata[i]
        elif s == "MONOMER":
            todaladata[s, i] = monomerdata[i]
        elif s == "ATTO":
            todaladata[s, i] = attodata[i]
        largos[s].append(len(todaladata[s,i]))

a = 0

#for s in structures:

#s = "DIMER"


for s in structures:
    print(s)
    bines = 5
    auxe = np.ceil(np.array(largos[s]) / bines)
    print(auxe)
    if s == "DIMER":
        nbins = [10, 12, 17, 15, 13, 7, 9]
        minimun = [0, 11, 21, 36, 30, 20, 95]
        maximun = [24, 51, 90, 110, 150, 200, 270]
    
    elif s == "MONOMER":
        nbins =  [9, 14, 13, 17, 11, 7, 8]
        minimun = [0, 1, 19, 45, 100, 40, 50]
        maximun = [20, 55, 70, 90, 200, 180, 270]
    
    elif s == "ATTO":
        nbins =  [23, 15, 24, 17, 15, 11, 16]
        minimun = [0, 11, 2, 3, 4, 5, 6]
        maximun = [300, 40, 900, 1100, 1500, 1700, 250]
    plt.figure()
    for i in range(len(powers)):
        print(s, powers[i])
        nozeros = todaladata[s, powers[i]]
    
    #    nozeros = dimerdata[powers[0]]
    #    for j in powers[1:]:
    #        nozeros = np.concatenate((nozeros, dimerdata[j]))
        #nozeros = selection
        #nozeros = selection[np.nonzero(selection)]
    
        nozeros = nozeros[np.where(nozeros > minimun[i])]
        nozeros = nozeros[np.where(nozeros < maximun[i])]

        N = nbins[i]
        #bins = np.linspace(0,int(np.max(histo)), N)
        bins = np.linspace(0,int(np.max(nozeros)), N)
        
        plt.hist(nozeros, bins=bins, alpha = 0.5, label=(powers[i]+"mW-"+ str(len(nozeros))))# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
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
    
        #plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
        plt.legend()
        plt.title('Histogram for {} '.format(s))
        plt.xlabel('Counts (kHz)')
        plt.ylabel(" points ")#total points ={} in {} bins".format(len(nozeros), N))
        #    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
        print(popt[1], popt[2], "1")
        fits[s,powers[i]] = (popt[1], popt[2])
    #    plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
    #    plt.xlim(0, int(np.max(nozeros)))
        #plt.show()
        
        if gaussfit:
            plt.plot(X,gaus(X,*popt),'g',lw=2, label='1G fit')
            plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
            def gauss(x,mu,sigma,A):
                return A*exp(-(x-mu)**2/2/sigma**2)
            
            def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
                return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
            
            expected = (popt[1],abs(popt[2]),popt[0],
                        1.5*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])  # 
            
            #expected = (11,2,8, 3,2,1)  # 
    
    
            try:
                params,cov = curve_fit(bimodal,x,y,expected)
                sigma = np.sqrt(np.diag(cov))
                a=vaca
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
plt.show()

#print(fits)

# %%
for i in powers:
    print("\n power={}".format(i))
    for s in structures:
        print(fits[s,i], s)
# %% ImageJ Roi size analysis

# change roi by 1 from 5 to 16, and also take 20
roi_size = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 25, 27, 30, 40, 50]
mean_roi = [219, 195.562, 180.56, 165.778, 152.959, 145.344, 138.728, 133.670, 129.810,
            126.951, 124.006, 121.832, 119.924, 118.332, 116.340, 114.812, 112.957,
            112.043, 111.187, 110.336, 108.683, 107.917]
mean_bg = [ 107.889, 107.188, 107.32, 107.972, 107.878, 107.484, 106.951, 106.930, 107.397,
           107.014, 106.438, 106.607, 106.733, 106.770, 106.691, 106.707, 106.588,
           106.696, 106.653, 106.736, 106.690, 106.737]


signal = np.array(mean_roi) - np.array(mean_bg)
signal2 = np.array(mean_roi) - np.mean(mean_bg)

odd = np.arange(0,len(roi_size), 3)
signal = signal[odd]
roi_size = np.array(roi_size)[odd]
x=np.array(roi_size)
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
pont = np.diff(signal)

x2=np.array(x)
x2=(x2[1:]+x2[:-1])/2 # for len(x)==len(y)
diff2 = np.diff(pont)

s,f = 0, len(signal)
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(roi_size[s:f], signal[s:f], '-o', label="signal")

#plt.plot(roi_size[s:f], signal2[s:f], '-o')
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x[s:f], pont[s:f], '-*r', label="derivated")
ax1.set_xlabel ("roi size (pix)")
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylabel ("signal per pixel", color='b')
#plt.title("subtracting a background from another roi close")
ax2.set_ylabel("derivate", color='r')
ax2.tick_params(axis='y', labelcolor='r')
#ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3.tick_params(axis='y', labelcolor='m')
ax3.plot(x2,diff2, '-+m',lw=2, label='diff2')
ax3.set_ylabel("Second derivate", color='m')
ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax3.set_xlabel ("roi size (pix)")

#plt.yscale("log")

# %%  Total amount of intensity analisys (for cy5@SiO2)

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

muestras = ["cy5", "cy5trolox"]

muestra = muestras[1]

promedio = {}
#for muestra in muestras:
if muestra == "cy5":
    f = ['C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _1/1_cy5alone_PLL_2.1mW_640nm _1_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _2/1_cy5alone_PLL_2.1mW_640nm _2_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _3/1_cy5alone_PLL_2.1mW_640nm _3_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _4/1_cy5alone_PLL_2.1mW_640nm _4_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _5/1_cy5alone_PLL_2.1mW_640nm _5_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _6/1_cy5alone_PLL_2.1mW_640nm _6_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _7/1_cy5alone_PLL_2.1mW_640nm _7_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _8/1_cy5alone_PLL_2.1mW_640nm _8_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _9/1_cy5alone_PLL_2.1mW_640nm _9_MMStack_Pos0.ome.tif',
         'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_cy5alone_PLL_2.1mW_640nm _10/1_cy5alone_PLL_2.1mW_640nm _10_MMStack_Pos0.ome.tif']
elif muestra == "cy5trolox":
    f=['C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_20minWait_cy5alone_2.1mW_640nm_1/1_TROLOX_30min_cy5alone_2.1mW_640nm_1_MMStack_Pos0.ome.tif',
       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_1/1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_1_MMStack_Pos0.ome.tif',
       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_2/1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_2_MMStack_Pos0.ome.tif']
#    ,
#       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_1/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_1_MMStack_Pos0.ome.tif',
#       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_2/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_2_MMStack_Pos0.ome.tif',
#       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_3/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_3_MMStack_Pos0.ome.tif',
#       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_4/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_4_MMStack_Pos0.ome.tif',
#       'C:/Origami testing Widefield/2020-05-22 Silica again & trolox/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_5/1_TROLOX_30minUV_105minWait_cy5alone_2.1mW_640nm_5_MMStack_Pos0.ome.tif']
# %%
zprofile = {}

for files in range(len(f)):
    
    data = io.imread(f[files])

    print(data.shape[0], data.shape[1], data.shape[2], "files=", files,"/", len(f))  # Nframes, heigth, length

    zprofile[files] = np.zeros((len(data[:,0,0])))
    for i in range(len(data[:,0,0])):
        zprofile[files][i] = np.mean(data[i,:,:])

    plt.plot(zprofile[files], label=f[files][-80:-50])
plt.legend()


#%%

largos = []
for i in range(len(zprofile)): largos.append(len(zprofile[i]))
print (largos)

promedios = promedio
promedios[muestra] = np.zeros((np.max(largos), np.max(largos))) * np.nan

for j in range(len(zprofile)):
    print(j)
    promedios[muestra][j] = (zprofile[j])

#    plt.plot(promedios[muestra][j])

#minimo = []
#for i in range(len(zprofile)): minimo.append(len(zprofile[i]))
#print (minimo)
#
#promedio[muestra] = zprofile[0][:min(minimo)]
#
#for j in range(1, len(zprofile)):
#    print(j)
#    promedio[muestra] = (zprofile[j][:min(minimo)] + promedio[muestra])
#
#plt.plot(promedio[muestra]/len(zprofile), 'm*')


# %%
for i in range(len(muestras)):
    plt.plot(promedio[muestras[i]])
plt.legend(muestras)



#for i in range(len(muestras)):

plt.plot(promedio[muestras[0]]/10)
plt.plot(promedio[muestras[1]]/3)

# %%

import numpy as np
import matplotlib.pyplot as plt

#for j in range(20):

def str2number(data):
    x=[]
    y=[]
    for i in range(len(data[:,0])):
        x.append(float(data[i,0]))
        y.append(float(data[i,1]))
    return x, y


xdic = {}
ydic = {}

#data = [data1, data2, data3, data4, data5, data6, data7, data8, data9,
#        data10, data11, data12, data13, data14, data15, data16, data17, data18,
#        data19, data20,data21, data22, data23, data24, data25, data26,
#        data27, data28, data29, data30, data31, data32, data33, data34, data35,
#        data36, data37, data38, data39, data40]
#loading = np.load("Tirf day 1.npz")
#data = loading["txtdata"]
data = [dataaaa]
positions = []
for i in range(len(data)):
    xdic[i], ydic[i] = str2number(data[i])
    positions.append(xdic[i][np.where(ydic[i]==np.max(ydic[i]))[0][0]])
    punto = plt.plot(positions[i], ydic[i][np.where(ydic[i]==np.max(ydic[i]))[0][0]], 'm*')

    plt.plot(xdic[i], ydic[i], '-',lw=0.3)
    
#plt.figure()
umeters = np.linspace(1,40,40)
#plt.plot(umeters, positions, '.')
plt.xlabel("z distance")
plt.ylabel("Plane movement")

#print(positions)

aux = ydic[0]
for i in range(len(newpos)):
    newpos[i] = aux[i]-aux[0]
#name = "Tirf day 1"
#np.savez(name, txtdata=data)

#piop = np.load(name+".npz")
#asdsa= piop['txtdata']

# %%

import numpy as np
import matplotlib.pyplot as plt
import time as time

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#name = 'C:/Origami testing Widefield/Tirf_angle_2.729_640nm_13mW_z_3050-1-3090_ all together/profile.txt'
#name = 'C:/Origami testing Widefield/Tirf_angle_2.729_640nm_13mW_z_3050-1-3090_ all together/profile_rectangle.txt'

#name = 'C:/Origami testing Widefield/2020-08-14 TIRF/640nm_5mW_2610Tirf_3064-3084_Z_1 All/profile.txt'
#name = 'C:/Origami testing Widefield/2020-08-14 TIRF/640nm_5mW_2650Tirf_3064-3084_Z_Alltogether/profile.txt'
#name = 'C:/Origami testing Widefield/2020-08-14 TIRF/640nm_5mW_2660Tirf_3064-3084_Z_Alltogether/profile.txt'
#name = 'C:/Origami testing Widefield/2020-08-14 TIRF/640nm_5mW_2680Tirf_3064-3084_Z_1 ALL/profile.txt'
#name = 'C:/Origami testing Widefield/2020-08-14 TIRF/640nm_5mW_2710Tirf_3064-3084_Z_1 ALL/profile.txt'

name = '//physpc07/German/2020-08-25 Paint Quencher/Tirf angle moving/profile.txt'
name = '//physpc07/German/2020-08-25 Paint Quencher/TIRF angle bleach/profile.txt'
macro = np.loadtxt(name)

smoo = 10
maxes = np.zeros(len(macro))
places= []
for i in range(len(macro)):
#    if i > 19:
#    #    maxes.append(np.nanargmax(macro[i,:]))
#        maxes[i] = (np.nanargmax(smooth(macro[i,:-500],smoo)))
#    #    places.append(np.where(macro[i,:]==np.nanargmax(macro[i,:]))[0])
#        plt.plot(smooth(macro[i,:-500],smoo), lw=0.4)
#        punto = plt.plot(maxes[i], np.max(macro[i,:-500]), 'b*')
#    elif i < 5 :
#        print("not using this")
##    #    maxes.append(np.nanargmax(macro[i,:]))
##        maxes[i] = (np.nanargmax(smooth(macro[i,:],smoo)))
##    #    places.append(np.where(macro[i,:]==np.nanargmax(macro[i,:]))[0])
##        plt.plot(smooth(macro[i,:],smoo), lw=0.9)
##        punto = plt.plot(maxes[i], np.max(macro[i,:]), 'ro')
#    else:
#    maxes.append(np.nanargmax(macro[i,:]))
    maxes[i] = (np.nanargmax(smooth(macro[i,:],smoo)))
#    places.append(np.where(macro[i,:]==np.nanargmax(macro[i,:]))[0])
    plt.plot(smooth(macro[i,:],smoo), lw=0.5)
    punto = plt.plot(maxes[i], np.max(macro[i,:]), 'm*')

#
#
maxes = np.array((maxes)) * 0.065
print(maxes)
#print(places)
plt.figure()
plt.plot(maxes)
plt.plot(np.diff(maxes))
fit = np.mean(np.diff(maxes))
print(fit)
print(np.rad2deg(np.arctan(fit)))

# %%

fites = [[-2.90282 , 0.19069], [-3.34674 , 0.10981],[ -3.54351 , 0.08272],[ -3.69757 , 0.0655],
       [ -3.79136, 0.05647]]

for j in range(len(fites)):
    medio = np.rad2deg(np.arctan((fites[j][0])))
    errorup =  np.rad2deg(np.arctan((fites[j][0] +fites[j][1])))
    errodown = np.rad2deg(np.arctan((fites[j][0] -fites[j][1])))
    final = [errodown, medio, errorup]
    error = (final[2]-final[1] + final[1]-final[0])/2
    print([medio,error])


#print(np.rad2deg(np.arctan(fites)))

# %%

good = np.zeros((len(histo), 2))
for i in range(len(histo)):
    for j in range(2):
        good[i,j] = float(histo[i,j].replace(',', '.'))

plt.plot(good[:,0],good[:,1])

#plt.hist(good[:,1],bins=good[:,0])

# %% Nicole Duetta Diagonal Matix

fname = 'C:/Origami testing Widefield/EEM for extinction/EzspecCompatible_SiDBSNP-2mM- for extiction_EEM.txt'

#data = np.genfromtxt(name)

f = open(fname,'r')

data = []
for line in f.readlines():
#    data.append(line.replace('\n','').split(' '))
    data.append(line.replace('\t\t',' ').replace('\t',' ').split(' '))

f.close()

#%%
matrix = dict()

#skip first 2 rows

for i in range(2, 6):#len(data)):
    print(i, "\n")
    print(data[i])

#    matrix[i] = []
#    for j in range(len(data[i])):
#        print(j)
#        matrix[i].append(np.float(data[i][j]))



#print(Ez[0,1])
#
#for i in range(len(Ez)):
#    print(Ez[i])
    
    
#%% Simple code To evaluate diferent angles for the quencher respect to the acceptor
    """ 
    for now is only in 2D,
    I am not sure how to do it in 3D ( or if make sense to lose time doing it)
    """
    
import numpy as np
import matplotlib.pyplot as plt

#angles = np.linspace(40,180-40,10)
angles = np.array([0, 43, 90, 137])

D = 6  # Distance acceptor - Capturing strands
Q = 5  # Distance fully elongated Quencher imager
A = np.deg2rad(angles)  # Angle between D and the quencher

h = Q* np.sin(A) 
r = Q* np.cos(A)  # if the angle is bigger than 90 the quencher is farder away so r is negative

#r = np.sqrt(Q**2 + h**2)
#l = D - r
#X = np.sqrt(h**2 + l**2)
X = np.sqrt(h**2 + (D-r)**2)




plt.plot(np.rad2deg(A),X,'-*')
#plt.plot(np.rad2deg(A),r,'-*')
plt.show()

print("\n", X)

#%% Simple code To evaluate diferent angles for Dye that makes fret
""" 
for now is only in 2D,
I am not sure how to do it in 3D ( or if make sense to lose time doing it)
"""
    
import numpy as np
import matplotlib.pyplot as plt

#angles = np.linspace(40,180-40,10)
angles = np.array([0, 43, 90, 137])

D = 6  # Distance acceptor - Capturing strands
W = 2  # Distance Donor - origami
A = np.deg2rad(angles)  # Angle between origami and Donor

h = W* np.sin(A) 
r = W* np.cos(A)  # if the angle is bigger than 90 the quencher is farder away so r is negative

#r = np.sqrt(Q**2 + h**2)
#l = D - r
#X = np.sqrt(h**2 + l**2)
Y = np.sqrt(h**2 + (D-r)**2)




plt.plot(np.rad2deg(A),Y,'-*')
#plt.plot(np.rad2deg(A),r,'-*')
plt.show()

print("\n", Y)

#%% Now itarating in all the new distances

import numpy as np
import matplotlib.pyplot as plt

#angles = np.linspace(40,180-40,10)
angles = np.array([0, 43, 90, 137, 180])

D = 6  # Distance acceptor - Capturing strands
W = 2  # Distance Donor - origami
A = np.deg2rad(angles)  # Angle between origami and Donor

h = W* np.sin(A) 
r = W* np.cos(A)  # if the angle is bigger than 90 the quencher is farder away so r is negative

#r = np.sqrt(Q**2 + h**2)
#l = D - r
#X = np.sqrt(h**2 + l**2)
Y = np.sqrt(h**2 + (D-r)**2)
print("Y = ", Y)
mapping = []
for i in Y:
#    angles = np.linspace(40,180-40,10)
    angles = np.array([43, 90, 137])
    
    D = i  # Distance acceptor - Capturing strands
    Q = 5  # Distance fully elongated Quencher imager
    A = np.deg2rad(angles)  # Angle between D and the quencher
    
    h = Q* np.sin(A) 
    r = Q* np.cos(A)  # if the angle is bigger than 90 the quencher is farder away so r is negative
    
    #r = np.sqrt(Q**2 + h**2)
    #l = D - r
    #X = np.sqrt(h**2 + l**2)
    X = np.sqrt(h**2 + (D-r)**2)
    mapping.append(X)
    
    
    
    
#    plt.plot(np.rad2deg(A),X,'-*')
    #plt.plot(np.rad2deg(A),r,'-*')
#    plt.show()
#    print("\n", X)
    
    plt.plot(np.rad2deg(A),X,'-*',label="Y_distance = {:.1f}".format(i))
    
plt.legend()
plt.show()
print(np.array(mapping))


#%% Ploting Fret Energy

import numpy as np
import matplotlib.pyplot as plt

R0 = 6.46
r = np.linspace(0, 3*R0, 100)
E = 1-(1 / (1+(r/R0)**6))
E2 = 1-(1 / (1+(r/R0)**4))
E3 = 1-(1 / (1+(r**6/((2*R0)**6))))


fix_r0 = min(r, key=lambda x:abs(x-R0))
E_r0 = E[np.where(r==fix_r0)][0]
E3_r0 =E3[np.where(r==fix_r0)][0]


E3_R2 = min(E3, key=lambda x:abs(x-E_r0))
R2 = r[np.where(E3==E3_R2)][0]
print( "\n", R0, R2,"\n", E_r0,E3_r0)

plt.plot(r,E,".-", label="1/6 (FRET)")
plt.plot(r,E3,".-", label="FRET2")
#plt.plot(r,E2,".-", label="1/4 (MIET)")
plt.xlabel("Distance [nm] (R0 = {})".format(R0))
plt.ylabel(" 1 - FRET / MIET Eff")
x_aux = [r[np.where(r==fix_r0)[0]],r[np.where(r==fix_r0)[0]],r[np.where(E3==E3_R2)[0]]]
plt.plot(x_aux,[E_r0,E3_r0,E3[np.where(E3==E3_R2)]], 'o')
#plt.xlim([4,8])
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.show()

#%% 2 dyes Fret

Eff = 0.6
r0 = R0
#respuesta = np.sqrt(np.sqrt(np.sqrt(((1/Eff)-1))))*r0
respuesta = (((1/Eff)-1)**(1/6))*r0

e=0.5 # r/R0 =
((1-e)/e)**(-1/6)

e=0.5
(2*((1-e)/e))**(1/6)

# E / 1-E = (R0/r1 )^6 + (R0/r2 )^6
e/(1-e)

same_distance = ((e/((1-e)*2))**(1/6))*R0
print(same_distance)

import numpy as np
from matplotlib import pyplot as plt

p = np.linspace(R0-2 , R0+2 ,200)
p = np.linspace(0.5 , 2 ,200)
q = np.linspace(0.5 , 2 ,200)

p,q = np.meshgrid(p,q)
#P = (R0 * (p**-6)) 
#Q = (R0 * (q**-6))
F = (R0/p)**6 + (R0/q)**6 - e/(1-e)
F = (1/p)**-1 + (1/q)**-1 - e/(1-e)
plt.contour(p, q, F, [0], cmap='coolwarm', vmin=-5, vmax=5)
plt.colorbar()
plt.xlabel("Dye 1 position (nm)")
plt.ylabel("Dye 2 position (nm)")
#plt.plot(same_distance,same_distance,'o')

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0 , 2 ,200)
x = np.linspace(0 , 2 ,200)

x,z = np.meshgrid(x,z)

Z = z**2 - 1
X = x

plt.contour(x,z,(X+Z), [0])
#plt.xlim([-1.5,1.5])
#plt.ylim([-1.5,1.5])
plt.show()
#%%
from sympy import plot_implicit, symbols, Eq, And
r0 = R0
e = 0.6
lowlim = 0.9*R0
maxlim = 1.2*R0
x, y = symbols('x y')
p1 = plot_implicit(Eq((r0/x)**6 + (r0/y)**6, e/(1-e)), (x, lowlim, maxlim), (y, lowlim, maxlim))
# plt.xlabel("Dye 1 position (r/R0)")
# plt.ylabel("Dye 2 position (r/R0)")
# plt.title("Distances 2 dyes with R0={}, to have E={}".format(R0,e))
#%%
import numpy as np
import matplotlib.pyplot as plt
# R0**6 (1/x**6)+(1/y**6) = e/(1-e)
#==> 1/y**6 =( e/(1-e) )*(R0**-6) - (1/x**6)
R0 = 6.46  # 6.46
e = 0.6
r1 = np.linspace(R0-0.121 , R0+1.14 ,201)
r2 =(( e/(1-e) )*(R0**-6) - (1/r1**6))**-(1/6)
tolerance = 0.1
plt.grid()
plt.plot(r1,r2)
# plt.fill_between(r1, r2-tolerance, r2+tolerance, alpha = 0.2)

plt.xlabel("Acceptor 1 position (nm)")
plt.ylabel("Acceptor 2 position (nm)")
# plt.title("Distances 2 dyes with R0={}, to have E={}".format(R0,e))
plt.vlines(R0,min(r2),max(r2), linestyles="dashed", color="orange")
plt.hlines(R0,min(r1),max(r1), linestyles="dashed", color="orange")
plt.text(min(r1),min(r2),"R0", color="orange")
plt.show()

#%% if r1 = r2
# HERRRREEEEEEEE  29.07.2022 TODO:
import numpy as np
import matplotlib.pyplot as plt
# R0**6 2* (1/same**6) = e/(1-e)
#==> 1/same**6 =( e/(1-e) )*(R0**-6)*(1/2)
R0 = 6.46  # 6.46
e = np.array([0.5,0.6,0.7])
# e=0.6
error_e = 0.25
one = (((1-e)/e)**(1/6))*(R0)
two_same = (((2*(1-e)/e))**(1/6)) * (R0)
# error_propagation = (-(1/((6*((1/e)-1)**(5/6) )* e**2)))*error_e
error_propagation = -(1/(3*((2**(5/6))*((1/e)-1)**(5/6))*e**2))*error_e
print(one,"\n", two_same, "±", abs(error_propagation))

#%%


import cv2

video_file = "C:/Origami testing Widefield/Test_morgane.mp4"

import cv2

vidcap = cv2.VideoCapture(video_file)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2

video_file = "C:/Origami testing Widefield/Test_morgane.mp4"

cap = cv2.VideoCapture(video_file)

N = 100
frames = np.zeros((N,1080,1920,3))
i=0



#cap = cv2.VideoCapture(0)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


n=0
while(cap.isOpened()):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n+=1
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
#        out.write(frame)
        
#        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

#out.release()

cv2.destroyAllWindows()
print(n, total)

#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io



#video_file = "C:/Origami testing Widefield/Test_morgane.mp4"
video_file = 'C:/Origami testing Widefield/MOV_2022_06_20_17_39_04_monochrome.mp4'

cap = cv2.VideoCapture(video_file)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frames", total_frames)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float `width`
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) # float `height`

#ret0, frame0 = cap.read()
#cap.release()
#mp4_channels = len(frame0[0][0])
#print("amount of channels in mp4 = ", mp4_channels)

#frames = np.zeros((total_frames,height,width,mp4_channels))
frames = np.zeros((total_frames,1080,1920,3))
i=0
for i in range(total_frames):
#    cap.isOpened()
    ret,frame = cap.read()
    if ret == True:
        frames[i] = frame




# =============================================================================
# while(cap.isOpened()):
#     i+=1
#     ret,frame = cap.read()
#     if ret == True:
#         frames[i] = frame
#     else:
#         break
# =============================================================================
#    frame2 = cv2.resize(frame,(1200,700))
   
    
#    cv2.imshow("video", frame2)
    
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
cap.release()

cv2.destroyAllWindows()

print("finish with",i, "or", total_frames )


#print(np.sum(frames[238]))
#%%

frame[0,0,:]
frame.shape
frames[0].shape
frames[0,0,0].shape
newframe = np.sum(frame,axis=2)
newframe.shape


plt.plot(frame)
plt.hist2d(frame)
plt.contourf(frame)
plt.imshow(newframe)
plt.plot(newframe[100:500,200:600])
plt.imshow(newframe[100:500,200:600])

plt.imshow(frame)
plt.imshow(np.sum(frames[100],axis=2))

plt.figure()
plt.imshow(frames[100,:,:,0])
plt.figure()
plt.imshow(frames[10,:,:,1])
plt.figure()
plt.imshow(frames[10,:,:,2])


data = np.sum(frames,axis=3)
plt.imshow(data)
data.shape


#%%
def plot_with_colorbar(imv,data):
    
    # Display the data and assign each frame a number
    x = np.linspace(1., data.shape[0], data.shape[0])

    # Load array as an image
    imv.setImage(data, xvals=x)

    # Set a custom color map
    colors = [
            (0, 0, 0),
            (45, 5, 61),
            (84, 42, 55),
            (150, 87, 60),
            (208, 171, 141),
            (255, 255, 255)
            ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    imv.setColorMap(cmap)





#%% Some test for Tau on Atto488
#from a file data

import numpy as np
import matplotlib.pyplot as plt

plt.hist(data, bins=25)



