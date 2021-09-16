# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:41:42 2021

@author: AdamczyA
"""

import numpy as np
import matplotlib.pyplot as plt
#
#plt.plot(data)
#plt.plot(data[:,10])
#data.shape
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 9_traces-35.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1 7_traces-35.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1 6_traces-42.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1 5_traces-44.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1 4_traces-46.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1 3_traces-39.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1 2_traces-44.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.1_traces-50.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 1_traces-42.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 2_traces-38.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 3_traces-32.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 4_traces-33.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 5_traces-37.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 6_traces-26.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 7_traces-31.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 8_traces-29.txt'           ) 
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp sds 0.3 9_traces-35.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 600mM 6_traces-43.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 600mM 5_traces-56.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 600mM 4_traces-50.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 600mM 3_traces-37.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 600mM 2_traces-46.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 600mM 1_traces-48.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 200mM 6_traces-51.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 200mM 5_traces-53.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 200mM 4_traces-61.txt' )
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 200mM 3_traces-46.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 200mM 2_traces-41.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 100mM 1_traces-52.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 100mM 2_traces-45.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 100mM 3_traces-51.txt')
#np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/0bp NaCl 100mM 4_traces-47.txt' )

np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/8bp-overhanging f2_traces-54.txt')

name = "8bp-overhanging f2_traces-54.txt"

#name = "0bp sds 0.1 7_traces-35.txt"
#name = "0bp sds 0.1 6_traces-42.txt"
#name = "0bp sds 0.1 5_traces-44.txt"
#name = "0bp sds 0.1 4_traces-46.txt"
#name = "0bp sds 0.1 3_traces-39.txt"
#name = "0bp sds 0.1 2_traces-44.txt"
#name = "0bp sds 0.1_traces-50.txt"
#name = "0bp sds 0.3 1_traces-42.txt"
#name = "0bp sds 0.3 2_traces-38.txt"
#name = "0bp sds 0.3 3_traces-32.txt"
#name = "0bp sds 0.3 4_traces-33.txt"
#name = "0bp sds 0.3 5_traces-37.txt"
#name = "0bp sds 0.3 6_traces-26.txt"
#name = "0bp sds 0.3 7_traces-31.txt"
#name = "0bp sds 0.3 8_traces-29.txt"
#name = "0bp sds 0.3 9_traces-35.txt"
#name = "0bp NaCl 600mM 6_traces-43.txt"
#name = "0bp NaCl 600mM 5_traces-56.txt"
#name = "0bp NaCl 600mM 4_traces-50.txt"
#name = "0bp NaCl 600mM 3_traces-37.txt"
#name = "0bp NaCl 600mM 2_traces-46.txt"
#name = "0bp NaCl 600mM 1_traces-48.txt"
#name = "0bp NaCl 200mM 6_traces-51.txt"
#name = "0bp NaCl 200mM 5_traces-53.txt"
#name = "0bp NaCl 200mM 4_traces-61.txt"
#name = "0bp NaCl 200mM 3_traces-46.txt"
#name = "0bp NaCl 200mM 2_traces-41.txt"
#name = "0bp NaCl 200mM 1_traces-57.txt"
#name = "0bp NaCl 100mM 1_traces-52.txt"
#name = "0bp NaCl 100mM 2_traces-45.txt"
#name = "0bp NaCl 100mM 3_traces-51.txt"
#name = "0bp NaCl 100mM 4_traces-47.txt"


data = np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/'+ name)
data.shape
np.mean(data[0:25, 0])

length=data.shape[0]
columns=data.shape[1]

avgdata = np.zeros(((int(length/25)),data.shape[1]))
avgdata.shape

for l in range(columns):
    for i in range(int(length/25)):
        j=i*25
        #print(i,j)
        avgdata[i,l] = (np.mean(data[j:j+25,l]))
      
theta=(np.linspace(0,170,18))
print 
print(avgdata)
plt.plot(theta,avgdata[:,0],"o")

tosavedata = np.zeros(((int(length/25)),data.shape[1] + 1))
tosavedata[:,0] = theta
tosavedata[:,1:] = avgdata

np.savetxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/'+"average curves_"+name, tosavedata, fmt='%.3e')

plt.plot(theta,avgdata[:,0],"o")
plt.plot(tosavedata[:,0], tosavedata[:,1], '*')

for i in range(columns):
    plt.plot(theta, avgdata[:,i])
    print(np.mean(avgdata[:,i]))

l=0
print(l)
plt.plot(theta, avgdata[:,l])
l=l+1

avgdata[:,3]



