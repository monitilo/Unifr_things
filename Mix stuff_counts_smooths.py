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


b= np.arange(0,10,0.5)

b*2
len(b)
c=np.linspace(0,100,len(b))

j = np.sin(b)
plt.plot()

plt.plot(j,'m*-')
#

data1.shape
data = data1

plt.plot(data1[:,30])

data = np.loadtxt('C:/Analizando Imagenes/Si proyect/1_Cy5 alone bad TROLOX traces/FILTERED_1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_1_traces-63.txt')
data = np.zeros((450, 35))
vector = np.linspace(0,2*np.pi,450)
for i in range(35):
    data[:,i] = np.sin(vector+i)

data.shape

plt.plot(data[:,2])

plt.plot(data[0:25,5],'.-')

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

#%% ImageJ counts to Photons
""" Equation from the Camera sheet: P = CF x (out-Off) / (Q(lambda) / 100) 
Where CF : conversion factor (electron / counts) {given by the camera} = 0.2
Out : counts 
Off : Dark offset = 100/pixel = 400 for 2x2 binning
Q(lambda) : Quantum Efficiency [%] = 80 %
"""
import numpy as np
import matplotlib.pyplot as plt
CF = 0.2
off = 400
QE= 0.8

x = np.linspace(0,800, 400)
x = np.array([540, 581, 603, 617])

P = CF * (x-off) / QE
#P = (CF/QE) * x - (CF/QE)*off

plt.plot(x, P, '-o')


#%%   Smooths for Nicole
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:  # , msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
N=10
x = np.linspace(0,2*np.pi,N)
y = np.sin(x) + np.random.random(N) * 0.5
yhat = savitzky_golay(y, 5, 3) # window size 51, polynomial order 3

plt.plot(x,y,'.-')
plt.plot(x,yhat, '*-',  color='red')
plt.show()


#%% Now the same with real data


x = np.linspace(0, len(data[:,0])//10, len(data[:,0]))
y = data[:, 15]
yhat = savitzky_golay(y, 5, 3) # window size 51, polynomial order 3

plt.plot(x,y,'.-')
plt.plot(x,yhat, '*-',  color='red')
plt.xlabel("time [s]")
plt.show()

#%%
ini = 800
end = 900
plt.plot(trace[ini:end,0],'.')
plt.plot(smooth_trace[ini:end,0])
#plt.plot(smooth_bg[ini:end,0], 'g')
plt.show()

plt.plot(nicole[ini:end,0], 'r')
plt.plot(smooth_trace[ini:end,0]-smooth_bg[ini:end,0], '.b')



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

"""

xc = 77
yc = 77
pixsize = 130
sigmax_laser = 5300
sigmay_laser = 4200
a = 0.5*(int(sigmax_laser/pixsize))
b = 0.5*(int(sigmay_laser/pixsize))

s=0

x = np.array(finaldata[samples[s]]["x"])
y = np.array(finaldata[samples[s]]["y"])

tic = time.time()

x1circle = []
y1circle = []
phot1circle = []

for i in range(len(x)):

    if (((x[i]-xc)/a)**2 + ((y[i]-yc)/b)**2) <= 1:
        x1circle.append(x[i])
        y1circle.append(y[i])
        phot1circle.append(finaldata[samples[s]]["photons"][i])

print( time.time()-tic)

hist2d = plt.hist2d(x1circle, y1circle, bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
plt.colorbar(hist2d[3])
plt.show()


tic = time.time()

x2circle = []
y2circle = []
phot2circle = []
for i in range(len(x)):

    if 1 < (((x[i]-xc)/a)**2 + ((y[i]-yc)/b)**2) <= 4:
        x2circle.append(x[i])
        y2circle.append(y[i])
        phot2circle.append(finaldata[samples[s]]["photons"][i])

print( time.time()-tic)

hist2d = plt.hist2d(x2circle, y2circle, bins=bines, range=([0,154], [0,154]), cmin=0, cmax=4000)
plt.colorbar(hist2d[3])
plt.show()

h1 = plt.hist(phot1circle, bins=60, alpha=0.5, range=(0,3500))
h2 = plt.hist(phot2circle, bins=180, alpha=0.5, range=(0,3500))


#tic = time.time()
#
#xnew = []
#for i in range(len(x)):
#    if (x[i]-xc/a)**2 <=1:
#        xnew[i].append(x[i])
#
#ynew = []
#for i in range(len(y)):
#    if (y[i]-yc/b)**2 <=1:
#        ynew[i].append(y[i])
#
#print( time.time()-tic)

#%%







