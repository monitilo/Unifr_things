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
#%%
#file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 NP y Antennas/UV-Vis 1_15_2020 6_46_40 AM german antennas.tsv'
#file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252 Gold NPs/UV-Vis 1_15_2020 8_29_14 AM gold antennas german.tsv'

#file = '//common/Physics/nicoleS/buffer stability/UV-Vis 2_3_2021 4_30_24 PM.tsv'

#file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252/UV-Vis 2_18_2022 3_47_58 PM nr antenna mojo &dimer.tsv'

# file = 'C:/Origami testing Widefield/NanodropOne_AZY1812252/UV-Vis 2_10_2022 5_08_13 PM.tsv'

#file = 'C:/Origami testing Widefield/Nanodrop/Dimers and more 31.05/UV-Vis 5_19_2022 4_52_18 PM dimer good and bad. and same after frozen.tsv'


root = Tk()
file = filedialog.askopenfilename(filetypes=(("", "*.tsv"), ("", "*.")))
root.withdraw()
file_folder = os.path.dirname(file)
file_name = os.path.basename(file)

print(file)

a = 1333  # cantidad de puntos. # 1333 en UV, 273 en ssDNA
skip = 10  # header y texto entre espectros.
#N = 9  # Cantidad de espectros esperados. CAMBIADO MAS ABAJO
#c = ['g','r','b','k','g','r','b']

labels = ["Blank", "300 ramp", "300 cycle","600 ramp","600 Cycle","NR","2nd NR",
         "1st 300 ramp","1st 300 cycle","1st 600 ramp","1st 600 Cycle"]
#labels = ["BLank", "NP recovered/10", "Dimer", "Monomer", "NP diluted 1:10"]
#labels = ["60", "40", "Si", "Blank", "Si"]
# %% 

dos = 1

i=0
length = False
while length == False:
    try:
        print(i)
        dos = 2*dos
        spectrum = np.loadtxt(file, delimiter='\t', skiprows = skip+i*a, max_rows = a-skip) # 1334
        i=i+1
    except:
        length = True
        N = i
        print("number of spectras = ", N)

alldata=np.zeros((N,a-2-skip))
allxaxis=np.copy(alldata)

datatosave = []
#try:
for i in range(N):
    # print(i)
    dos = 2*dos
    spectrum = np.loadtxt(file, delimiter='\t', skiprows = skip+i*a, max_rows = a-skip) # 1334
    # print(spectrum)
    datatosave.append(np.array(spectrum[:,1]))
    alldata[i,:] = spectrum[:,1]
    waveleng = spectrum[:,0]
    plt.figure(file)
#    if i < len(labels):
#        plt.plot(spectrum[:,0], spectrum[:,1], label=labels[i]) #, c[i])
#    else:
    plt.plot(spectrum[:,0], spectrum[:,1]) #, c[i])
#    plt.xlim((300,700))
#    plt.ylim((-0.1, 2))

#    left = np.where(waveleng==340)[0][0]
#    right = np.where(waveleng==600)[0][0]
    
#    maxpeak = np.where(spectrum[left:right,1]==np.max(spectrum[left:right,1]))[0][0]
#    maxvalue = spectrum[maxpeak+left,1]
#        plt.ylim((0,maxvalue*1.2))
#    plt.legend()

#    if maxvalue < 0.1:
#        print("\n BLANK!!!! \n")
#        dos = 1

#    print( "maxpeak = ", waveleng[maxpeak+left], "\n maxvalue y=", maxvalue, dos)
#    print("\n concentration 40 (nM)=", maxvalue /3.36 )
#    print("\n concentration 50 (nM)=", maxvalue /1.935 )
#except:
#     print("el i maximo es", i)

#plt.figure()
#plt.grid()
#plt.plot(spectrum[:,0], spectrum[:,1])
#plt.plot(waveleng, alldata[6,:],'g')
#plt.plot(waveleng, alldata[10,:],'r')
#plt.xlim((340,500))
#for i in range(2):
#    print(i+9)
#    plt.plot(waveleng, alldata[i+16,:])

print("number of spectras = ", N)
finalpart = file.split("/")[-1]

datatosave_name = file[:-len(finalpart)] + "PYTHON_" + finalpart
#datatosave_name = "//common/Physics/nicoleS/buffer stability/PYTHON_UV-Vis 2_3_2021 4_30_24 PM.txt"

datatosave_name = file[:-len(finalpart)] + "Wabeleng_" + finalpart[:6]
#datatosave_waveleng = "//common/Physics/nicoleS/buffer stability/Wavelengs_UV-Vis.txt"


save_folder = "//onecopy/Science/Physics/Acuna_group/German_onecopy/NanoDrop/Analysis/"
save_name = save_folder + "PYTHON_" + file_name[:-3] + "txt"
save_name_wave = save_folder + "Wabeleng_" + file_name[:-3] + "txt"

# np.savetxt(save_name_wave, np.array(waveleng).T, delimiter="    ", newline='\r\n')

np.savetxt(save_name, np.array(datatosave).T, delimiter="    ", newline='\r\n')
plt.show()

#%%
# =============================================================================
# 
# plt.plot(waveleng, alldata[3],label="3x")
# plt.plot(waveleng, alldata[4], label = "Normal (600)")
# plt.plot(waveleng, alldata[5], label = "300")
# plt.plot(waveleng, alldata[6], label = "900")
# plt.xlim(500,851)
# plt.ylim(0,0.6)
# plt.title("Final antennas: 40 --> 20; 30min each °C.")
# plt.legend()
# 
# print(max(alldata[3][500:]), max(alldata[4][500:]))
# plt.show()
# =============================================================================


# =============================================================================
# plt.plot(waveleng, alldata[1],label="Peak={}".format(waveleng[np.argwhere(alldata[1] == max(alldata[1][500:]))[0][0]]))
# # plt.plot(waveleng, alldata[2], label = "NR B")
# 
# plt.xlim(480,851)
# plt.ylim(0,0.6)
# plt.title("Dimer 27.05.22")
# plt.legend()
# 
# plt.vlines((waveleng[np.argwhere(alldata[1] == max(alldata[1][500:]))[0]]), 0, 9, alpha=0.5, color='grey',linestyles="dashed")
# 
# print(max(alldata[1][500:]), max(alldata[1][500:]))
# print(waveleng[np.argwhere(alldata[1] == max(alldata[1][500:]))[0]])
# =============================================================================

# plt.plot(waveleng, alldata[1], label="Peak={}".format(waveleng[np.argwhere(alldata[1] == max(alldata[1][500:]))[0][0]]))
# plt.plot(waveleng, alldata[2], label = "NR B")

plt.xlim(480,851)
plt.ylim(0,0.6)
plt.title("Dimer 30.05 after N2 Freezing")
plt.legend()

plt.vlines((waveleng[np.argwhere(alldata[1] == max(alldata[1][500:]))[0]]), 0, 9, alpha=0.5, color='grey',linestyles="dashed")

print(max(alldata[1][500:]), max(alldata[1][500:]))
print(waveleng[np.argwhere(alldata[1] == max(alldata[1][500:]))[0]])

plt.show()
#%%

names = ["blank", "24nmMg", "Normal", " 50min step ", "Cyclex6 ", "300", "900"]
maximos = [0,0,0]
# for i in range(1,N):
#     print(i)
#     plt.plot(waveleng, alldata[i],label="{}_Peak={}".format(names[i],waveleng[np.argwhere(alldata[i] == max(alldata[i][500:]))[-1][-1]]))
#     maximos.append(np.max(alldata[i][500:]))
#     # plt.hlines(maximos[i], 0, 850, linestyles="dashed", alpha=0.5, color='grey')

# plt.plot(x, afterN2, label="N2_Freeze_Peak={}".format(waveleng[np.argwhere(afterN2 == max(afterN2[500:]))[-1][-1]]))
plt.xlim(500,851)
plt.ylim(-0,0.38)
plt.title(file_name)
plt.legend()
print(maximos)

#%%
# x = waveleng
# afterN2 = alldata[1]
# plt.plot(x, afterN2)
#%%

i=1
plt.plot(waveleng, alldata[i]/max(alldata[i][500:]),label="24mmMg")
i+=1
plt.plot(waveleng, alldata[i]/max(alldata[i][500:]), label = "Normal")
i+=1
plt.plot(waveleng, alldata[i]/max(alldata[i][500:]),label="50minStep")
i+=1
plt.plot(waveleng, alldata[i]/max(alldata[i][500:]),label="Cyclex6")
# plt.plot(x, afterN2/max(afterN2[500:]), label = "N2 Freeze")
plt.xlim(500,851)
plt.ylim(0,1.5)
plt.title("Norm")
plt.legend()

print(max(alldata[2][500:]), max(alldata[3][500:]))
print(waveleng[np.argwhere(alldata[2]==max(alldata[2][500:]))])
print(waveleng[np.argwhere(alldata[3]==max(alldata[3][500:]))])
plt.show()

#%%

lista = range(len(labels))
lista = [1,2,3,4,7,8,9,10]
# lista = [1]
data_dict = dict()
for i in lista :
    data_dict[labels[i]] = alldata[i]
    plt.plot(waveleng, data_dict[labels[i]], label=(labels[i]+" Max {}".format(max(data_dict[labels[i]][500:]))))
    plt.legend()
    plt.xlim(500,851)
    plt.ylim(0,0.4)
    plt.title("All spectras")
plt.show()


#%%
from scipy.signal import savgol_filter

smothing = 99
poly = 3

lista = range(len(labels))
lista = [1,2,3,4,7,8,9,10]
# lista = [1]
data_savgol = dict()
for i in lista :
    data_savgol[labels[i]] = savgol_filter(alldata[i],smothing,poly)
    data_savgol[labels[i]] = data_savgol[labels[i]]/max(data_savgol[labels[i]][500:])
    plt.plot(waveleng, data_savgol[labels[i]],
             label=(labels[i]+" peak {}".format(waveleng[np.argwhere(data_savgol[labels[i]]== max(data_savgol[labels[i]][500:]))[-1][-1]])))
    plt.legend()
    plt.xlim(660,790)
    plt.ylim(0.8,1.02)
    plt.title("Norm")


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


smothing = 99
poly = 3

def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


xrange = 75 # 75 seems credible. More start to be out of the peak

lista = range(len(labels))
lista = [6,7,8,9,10]
lista = [1,2,3,4]
lista = [1,2,3,4,6,7,8,9,10]
# lista = [5]
# data_dict = dict()
fitted_data = dict()
for i in lista :
    data_dict[labels[i]] = alldata[i]
    xcut = np.argwhere(data_dict[labels[i]] == max(data_dict[labels[i]][500:]))[-1][-1] - xrange
    xmax = xcut + xrange*2
    xdata = waveleng[xcut:xmax]
    ydata = alldata[i][xcut:xmax]
    p0 = (max(data_dict[labels[i]][xcut:]),
          waveleng[np.argwhere(data_dict[labels[i]]== max(data_dict[labels[i]][500:]))[-1][-1]], 7)
    popt, pcov = curve_fit(func, xdata, ydata,p0=p0)
    # print (popt)
    plt.figure("Maxs")
    ym = func(xdata, popt[0], popt[1], popt[2])
    plt.plot(xdata, ym, c='r', linestyle="--", alpha=0.5)
    plt.legend()
    peak_position = waveleng[np.argwhere(ym==max(ym))[-1][-1]+xcut]
    # print("Peaks position", peak_position)

    plt.plot(waveleng, data_dict[labels[i]], label=(labels[i]+" Max {:.2}".format(max(ym))+" peak {:2}".format(peak_position)))
    plt.legend()
    plt.xlim(500,851)
    plt.ylim(0,0.4)
    plt.title("All spectras")
    fitted_data[labels[i]] = popt
    
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


#%%


from scipy.signal import savgol_filter

smothing = 99
x = waveleng
Dimer = alldata[2]/max(alldata[2][500:])
Sdimer = savgol_filter(Dimer,smothing,3)
Monomer = alldata[3]/max(alldata[3][500:])
Smonomer = savgol_filter(Monomer,smothing,3)

plt.plot(x, Sdimer, label="dimer")
plt.plot(x, Smonomer, label="monomer")
plt.vlines(waveleng[np.argwhere(Sdimer==max(Sdimer[500:]))[-1]], 0,1.5)
plt.vlines(waveleng[np.argwhere(Smonomer==max(Smonomer[500:]))[-1]], 0,1.5, color="orange")

plt.xlim(500,851)
plt.ylim(0,1.0)
plt.title("Smothed")
plt.legend()
plt.show()

print(waveleng[np.argwhere(Smonomer==max(Smonomer[500:]))])
print(waveleng[np.argwhere(Sdimer==max(Sdimer[500:]))])

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
