# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:41:42 2021

@author: chiarelg
"""


a = [1,2,3,4,5,6,7]

print(a)

type(a)



import numpy as np
import matplotlib.pyplot as plt

#%%
b= np.arange(0,10,0.5)

b*2
len(b)
c=np.linspace(0,100,len(b))

j = np.sin(b)
#plt.plot()

#plt.plot(j,'m*-')
#

#data1.shape
#data = data1

#plt.plot(data1[:,30])

#data = np.loadtxt('C:/Analizando Imagenes/Si proyect/1_Cy5 alone bad TROLOX traces/FILTERED_1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_1_traces-63.txt')
data = np.zeros((450, 35))
vector = np.linspace(0,2*np.pi,450)
for i in range(35):
    data[:,i] = np.sin(vector+i)

data.shape

plt.plot(data[:,2])

plt.plot(data[0:250,5],'.-')

data[5000:200025,0].shape

length = data.shape[0]
length
colums = data.shape[1]
colums


avgdata = np.zeros((int(length/25), colums))
avgdata.shape

for l in range(colums):
#    print(l)
    for i in range(int(length/25)):
        j = i*25
#        print(i,j)
        avgdata[i,l] = np.mean(data[j:j+25,l])

plt.plot(avgdata[:,0],"o")

#np.savetxt()
#%%
#data = np.loadtxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/'+ name)
data = np.zeros((450, 35))
vector = np.linspace(0,2*np.pi,450)
for i in range(35):
    data[:,i] = np.sin(vector+i)

data[:,i//2] = np.tan(vector+i)

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

#np.savetxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/'+"average curves_"+name, tosavedata, fmt='%.3e')

#plt.plot(theta,avgdata[:,0],"o")
#plt.plot(tosavedata[:,0], tosavedata[:,1], '*')

for i in range(columns):
    plt.plot(theta, avgdata[:,i])
    print(np.mean(avgdata[:,i]))

l=0
print(l)
plt.plot(theta, avgdata[:,l])
l=l+1

avgdata[:,3]

#%%
columns = 29
plot_columns =  5  #  int(np.sqrt(columns))
plot_files =  4  #   int(np.ceil(columns/plot_columns))
graphs = int(np.ceil(columns / (plot_columns*plot_files)))


print(plot_files,plot_columns, "=", plot_files*plot_columns)

try:
    t=0
    for n in range(graphs):
        print("n",n)
        fig, axs = plt.subplots(plot_files, plot_columns)
        for i in range(plot_files):
#            print("i,j", i,j)
            for j in range(plot_columns):
#                print("i,j", i,j)
#                print("t",t)
                axs[i,j].plot(theta, avgdata[:,t],
                               label="{}".format((t)))
        #        axs[j,i].set_title("main{}".format(i+plot_columns*j))
                axs[i,j].legend(handlelength=0, handletextpad=0, fancybox=True)
                t+=1
except: pass
#except IOError as e:
#    print("I/O error({0}): {1}".format(e.errno, e.strerror))
    
    
#%%
    
    
    


