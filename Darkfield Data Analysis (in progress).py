# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:32:22 2019

@author: chiarelg

PARA la data del Nanodrop
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
import re
#%%

root = Tk()
file = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.txt")))
root.withdraw()
file_folder = os.path.dirname(file)
file_name = os.path.basename(file)

print(file)

data = np.loadtxt(file)
print(data.shape)
#N = 9  # Cantidad de espectros esperados. CAMBIADO MAS ABAJO
#c = ['g','r','b','k','g','r','b']

#labels = ["60", "40", "Si", "Blank", "Si"]

# %% 
numbers_for_x = re.findall(r'\d+', file)


x = np.arange(int(numbers_for_x[0]), int(numbers_for_x[2])+1,
                int(numbers_for_x[1]))


plt.plot(x,data[:,570])


#%%


#%% 
""" The data "NICOLE" is already smoothed (using poly3 and window 5).
savitzky_golay(self.raw_data[p], 5, 3)
No need for this in principle.
"""

from scipy.signal import savgol_filter

smothing = 5
poly = 3


data_savgol = dict()
data_savgol_norm = dict()

for i in range(data.shape[0]) :
    data_savgol[i] = savgol_filter(data[:,i],smothing,poly)
    data_savgol_norm[i] = data_savgol[i]
    # plt.plot(x, data_savgol[i])
    plt.plot(x, data_savgol_norm[i])
    # plt.legend()
    # plt.xlim(660,790)
    # plt.ylim(0.8,1.02)
    # plt.title("Norm")

# plt.plot(x, data_savgol[i])

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

range_peaks = 3
guess_peaks = np.zeros((data.shape[0]))
fitted_data = dict()
data_dict = dict()

for i in range(data.shape[0]):
    data_dict[i] = data[:,i]  # /(np.max(data[:,i]))
    xdata = x
    ydata = data_dict[i]
    guess_peak = int(np.argwhere(data_dict[i]== max(data_dict[i]))[-1][-1])
    guess_peaks[i] = x[guess_peak]

    p0 = (max(data_dict[i]),
          x[guess_peak], 50)
    popt, pcov = curve_fit(func, xdata[guess_peak-range_peaks:guess_peak+range_peaks], ydata[guess_peak-range_peaks:guess_peak+range_peaks],p0=p0)
    # print (popt)
    plt.figure("Maxs")
    ym = func(xdata, popt[0], popt[1], popt[2])
    # plt.plot(xdata[guess_peak-range_peaks:guess_peak+range_peaks], ym[guess_peak-range_peaks:guess_peak+range_peaks], c='r', linestyle="--", alpha=0.5)
    # plt.legend()
    peak_position = x[np.argwhere(ym==max(ym))[-1][-1]]
    # print("fit Peaks position", peak_position,"Guess peak pos", x[guess_peak])

    plt.plot(x, data_dict[i], label=(str([i])+" Max {:.2}".format(max(ym))+" peak {:2}".format(peak_position)))
    # plt.legend()
    # plt.xlim(500,851)
    # plt.ylim(0,0.4)
    plt.title("All spectras {}".format(len(data_dict)))
    fitted_data[i] = popt

    
    # plt.figure("positions of peaks")

    # data_savgol[labels[i]] = savgol_filter(alldata[i],smothing,poly)
    # data_savgol[labels[i]] = data_savgol[labels[i]]/max(data_savgol[labels[i]][500:])
    # xdata = waveleng[xcut:xmax]
    # ydata = data_savgol[labels[i]][xcut:xmax]
    # p0 = (1, waveleng[np.argwhere(data_savgol[labels[i]]== max(data_savgol[labels[i]][500:]))[-1][-1]], 7)
    # popt, pcov = curve_fit(func, xdata, ydata,p0=p0)
    # # print ("savgol", popt)
    # ym = func(xdata, popt[0], popt[1], popt[2])
    # plt.plot(xdata, ym, c='r', linestyle="--", alpha=1)
    # plt.legend()
    # plt.plot(waveleng, data_savgol[labels[i]],
    #          label=(labels[i]+" peak {:2}".format(waveleng[np.argwhere(ym== max(ym))[-1][-1]+xcut])))
    # plt.legend()
    # plt.xlim(660,790)
    # plt.ylim(0.8,1.02)
    # plt.title("Norm")
    
plt.show()

#%% Cheching if it make sense to do the gaussian fit
fitted_peaks = np.zeros((len(fitted_data)))
for i in range(len(fitted_data)):
    fitted_peaks[i] = fitted_data[i][1]


plt.plot(guess_peaks, fitted_peaks,'o')
plt.plot([620, 780], [620, 780], ls="--", c=".3")
plt.xlabel("Guess Peaks (max of smooth signal)")
plt.ylabel("Fitted peak (max of gaussian fit to smooth signal)")
plt.title("comparing max vs fit")
#%%

# %% Bad way to compare max vs fitt

bines = None

alfa = 0.9
blue = (79/255,134/255,198/255)
green = (104/255,153/255,93/255) 
yellow = (222/255,157/255,38/255)

color = [yellow, blue,green]
edgecolor = ["#996633", "#336699","#336633"]

i = 0
h1 = plt.hist(guess_peaks, bins=bines)
h2 = plt.hist(fitted_peaks, bins=bines)
plt.close()
# , bins=bines, range=(0,photon_treshold), density=density,alpha = alfa, label=(subgroups[i]+"; "+str(len(photons_to_plot))+" locs"),color=color[subgroups[i]],width=10,edgecolor=edgecolor[subgroups[i]],linewidth=1)  #, color = color[subgroups[i]])
step = abs(h1[1][0]-h1[1][1] )/2
plt.bar(h1[1][:-1]+step,h1[0]/max(h1[0]),width=step*2, alpha=alfa,
        color=color[i],edgecolor=edgecolor[i],label="{}".format(i))    

i += 1

step2 = abs(h2[1][0]-h2[1][1] )/2
plt.bar(h2[1][:-1]+step2,h2[0]/max(h2[0]),width=step2*2, alpha=alfa,
        color=color[i],edgecolor=edgecolor[i],label="fitted {}".format(i))    

plt.legend()
plt.show()




#%%
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
    except:
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
#%%
