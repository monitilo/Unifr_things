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




