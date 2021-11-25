# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:41:42 2021

@author: chiarelg
"""

#%% Here I take the data and bin it by 25

import numpy as np
import matplotlib.pyplot as plt
import numpy, scipy.optimize
import os
from tkinter import Tk, filedialog

root = Tk()
filename = filedialog.askopenfilename(title="Choose your long Traces", filetypes=(("", "*.txt"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(filename)
only_name = os.path.basename(filename)
print("Load traces from: ", filename, "\n")
data = np.loadtxt(filename)


root = Tk()
nametoload_ori = filedialog.askopenfilename(title="Choose Origami orientation", filetypes=(("", "*.txt"), ("", "*.")))
root.withdraw()
folder_ori = os.path.dirname(nametoload_ori)
only_name_ori = os.path.basename(nametoload_ori)
print("Load origami from:", nametoload_ori, "\n")
file_origami = np.loadtxt(nametoload_ori)


length=data.shape[0]
columns=data.shape[1]

avgdata = np.zeros(((int(length/25)),data.shape[1]))
avgdata.shape

for l in range(columns):
    for i in range(int(length/25)):
        j=i*25
        #print(i,j)
        avgdata[i,l] = (np.mean(data[j:j+25,l]))
      
theta=(np.linspace(0,170,len(avgdata)))  # angle vector

#plt.plot(theta,avgdata[:,0],"o")  # to show that works.
#plt.show()

for i in range(columns):
    plt.plot(theta, avgdata[:,i]) #Plot all the curves together

    
# =============================================================================
# # Save intermediate data already bin by 25
# tosavedata = np.zeros(((int(length/25)),data.shape[1] + 1))
# tosavedata[:,0] = theta
# tosavedata[:,1:] = avgdata
# 
# np.savetxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/'+"average curves_"+name, tosavedata, fmt='%.3e')
# 
# load the data saved before if needed
# avgdata = np.loadtxt('C:/Analizando Imagenes/code/Aleksandra/odp_stretchprojectdataanalysispythoncode/average curves_e 2bp_traces-39.txt')
# 
# =============================================================================


# =============================================================================
# Find the max of each trace (old method. Now we use the fit)
# cy5_angle = np.zeros(columns)
# for i in range(columns):
#     if i not in todelete:
# #        print(i)
#         cy5_angle[i] = theta[np.where(avgdata[:,i] == np.max(avgdata[:,i]))][0]
# 
# #plt.plot(cy5_angle,'-o')
# plt.hist(cy5_angle)
# plt.show()
# =============================================================================

"""
Fitting and ploting

"""

import scipy as sp
from scipy.optimize import curve_fit


plot_columns =  5  #  int(np.sqrt(columns))
plot_files =  4  #   int(np.ceil(columns/plot_columns))
graphs = int(np.ceil(columns / (plot_columns*plot_files)))


origami_number = np.zeros(len(file_origami[:,0]),dtype=int)
for i in range(len(origami_number)):
    origami_number[i] = int(file_origami[i,0])

origami_angle_m = file_origami[:,1]

#all_posibles_oris = np.arange(columns)
#
#not_in_superres = np.delete(all_posibles_oris, origami_number)


fits = dict()
x=theta
def my_sin2(t,peroid,amplitude,phase,offset):
           return amplitude*((sp.sin((t-phase)*sp.pi/peroid)))**2 + offset

x1 = sp.linspace(0,180,100000)

failed_fits = []

t=0
for n in range(graphs):
    fig, axs = plt.subplots(plot_files, plot_columns)
    for i in range(plot_files):

        for j in range(plot_columns):
            if t in origami_number:
                try:
                    V = avgdata[:,t]
                    axs[i,j].plot(x, V, linewidth=2,
                                   label="{}".format((t)))
                    
                    guess_peroid= 180
                    minimo = np.array(x[np.where(V==np.min(V))[0]])[0]
                    guess_phase = minimo
                    guess_offset = np.min(V)
                    guess_amplitude = np.max(V) - guess_offset
                    guess_bounds = ([100,0,0,0], [260, numpy.max(V), 180, numpy.max(V)])
                    p0 =[guess_peroid, guess_amplitude, guess_phase, guess_offset]
                    fit = curve_fit(my_sin2,x, V, p0=p0, bounds=guess_bounds)
                    data_fit = my_sin2(x1,*fit[0])
    
                    fits[t] = fit[0]  # fit[0] = Period, amplitud, phase, offset
    

                    axs[i,j].plot(x1, data_fit, "r--", linewidth=1)  # ,
                    axs[i,j].legend(handlelength=0, handletextpad=0, fancybox=True)
                except: 
                    print("failed fit trace N = ", t)
                    failed_fits.append(t)
            t+=1

#except IOError as e:
#    print("I/O error({0}): {1}".format(e.errno, e.strerror))



plt.show()


print("Failed fits: ",failed_fits)
#%% Delete the ones That look bad


add_to_delete = [25] + failed_fits

print("Traces to NOT analyse: ", add_to_delete)

#todelete = np.sort(np.concatenate((not_in_superres, add_to_delete)))

good_origamis = np.copy(origami_number)
origami_angle_minus = np.copy(origami_angle_m)

for i in range(len(add_to_delete)):
#    print(i,add_to_delete[i])
    origami_angle_minus = np.delete(origami_angle_minus, np.where(good_origamis == add_to_delete[i])[0])
    good_origamis = np.delete(good_origamis, np.where(good_origamis == add_to_delete[i])[0])

cy5_angle = np.zeros(columns)

try:
    t=0
    for n in range(graphs):
        fig, axs = plt.subplots(plot_files, plot_columns)
        for i in range(plot_files):
            for j in range(plot_columns):
                if t  in good_origamis:
                    V = avgdata[:,t]
                    
                    data_fit = my_sin2(x1,*fits[t])
                    cy5_angle[t] = fits[t][2]+90
    
                    axs[i,j].plot(x, V, linewidth=2,
                                   label="{}".format((t)))
                    axs[i,j].plot(x1, data_fit, "r--", linewidth=1)  # , label="angle_{:0.3f}".format(cy5_angle))  # ,
                    axs[i,j].legend(handlelength=0, handletextpad=0, fancybox=True)
                t+=1

except: pass
#except IOError as e:
#    print("I/O error({0}): {1}".format(e.errno, e.strerror))

plt.show()





#%% Determine the interested parameters:

"""
calculate modulation
fiting sin2(data). {A*sin(pi*(x-xc)/w1)**2 + y0 } ==> get amplitud and mean.
w1 should be 180 ± 10
modulation = (Amp/2) / (y0 + Amp/2)

"""

periods = np.zeros(len(good_origamis))
modulation = np.zeros(len(good_origamis))

t=0
for i in range(columns):
    if i in good_origamis:
        amp = fits[i][1]  # fits[i] = Period, amplitud, phase, offset
        y0 = fits[i][3]
        periods[t] = fits[i][0]
        modulation[t] = 0.5*amp/(y0 + (0.5*amp))
        t +=1

print("modulation=", modulation)
print("periods =", periods)
#%%
"""
Determine the Relative_angle:
first: change the angle range from -180 _ 180 to 0_360
"""


""" Data in "origamis_angles_m": 
Colum 1: #traces  || column2: origamis_angles_m
"""




origami_angle_ok = np.copy(origami_angle_minus)

for i in range(len(origami_angle_ok)):
    if origami_angle_ok[i] < 0:
        origami_angle_ok[i] = 360 + origami_angle_minus[i]
    else:
        origami_angle_ok[i] = origami_angle_minus[i]  # dah

"""
Then, calculate the difference between origami_angle_ok and cy5_angle
"""

cy5_angle_pro = np.copy(cy5_angle)
difference = np.zeros(len(good_origamis))
for p in range(len(good_origamis)):
    if cy5_angle[int(good_origamis[p])] > 180:
        cy5_angle_pro[int(good_origamis[p])] = cy5_angle[int(good_origamis[p])] - 180
    difference[p] = cy5_angle_pro[int(good_origamis[p])] - origami_angle_ok[p]

#
#"""
#difference goes between -180_180 les change it to 0_180
#"""
#
#relative_angle = np.copy(difference)
#for i in range(len(difference)):
#    if difference[i] > (-180):
#        relative_angle[i] = abs(difference[i])
#    else:
#        relative_angle[i] = abs(180+difference[i])
#
#bines = len(diff_angles//2)
#plt.hist(relative_angle, bins=bines, alpha=0.8)
#plt.hist(diff_angles, bins=bines, alpha=0.2)
##print(diff_angles, "\n")


"""
difference goes between -180_180 les change it to 0_180, NEW version 01.10
"""


relative_angle = np.copy(difference)
for i in range(len(difference)):
    if difference[i] > (0):
        relative_angle[i] = difference[i]
    elif difference[i] < (-180):
        relative_angle[i] = abs( 360 + difference[i] )
    else:
        relative_angle[i] = 180 + difference[i]


""" Data out: 
Colum1: #traces  || column2: relative_angle || Column3: modulation || Col4: period
    
"""

tosavefinaldata = np.zeros(( len(good_origamis), 4))
tosavefinaldata[:,0] = good_origamis
tosavefinaldata[:,1] = relative_angle
tosavefinaldata[:,2] = modulation
tosavefinaldata[:,3] = periods
plt.show()



np.savetxt(filename[:-4] + "_Angle_diff.txt", tosavefinaldata,
           fmt='%.d' +'\t'+ '%.1f'+'\t'+ '%.3f'+'\t'+ '%.1f',
           header="origami"+ "\t"+ "Angle_Diff"+ "\t"+ "Modulation"+ "\t"+ "Period",
           delimiter='\t')

print("final data saved as " + filename[:-4] + "_Angle_diff.txt")

"""
Origami orientation filtered from the imput
Colum 1: #traces  || column2: Angle origami from image 
"""

data_for_teun = np.zeros(( len(good_origamis), 2))
data_for_teun[:,0] = good_origamis
data_for_teun[:,1] = origami_angle_ok

np.savetxt(filename[:-4] + "_Ori_angle_Teun.txt", data_for_teun,
           fmt='%.d' +'\t'+ '%.1f',
           header="origami"+ "\t"+ "Angle from image",
           delimiter='\t')
#%%  PLOTS!!!

fig, axs = plt.subplots(3)

bines = None # len(relative_angle)// 2
axs[0].hist(relative_angle, bins=bines, color="blue", label="Relative angle")
axs[0].set_title("Relative angle")
#axs[0].set_xlabel("relative angle °")
axs[0].legend()
axs[1].hist(periods, bins=bines, color="Orange", label="Period")
axs[1].set_title("Period")
#axs[1].set_xlabel("Period 1/s")
axs[1].legend()
axs[2].hist(modulation, bins=bines, color="magenta", label="Modulation")
axs[2].set_title("Modulation")
axs[2].legend()
#axs[3].hist(origami_angle_ok, bins=bines, color="green", label="Origami orientation")
#axs[3].set_title("Origami orientation")
#axs[3].legend()
fig.tight_layout()
plt.show()


#%% Other plots

#plt.title("Origami angle Teun")
#plt.plot(good_origamis, origami_angle_ok, 'o-', color="green", label="Origami angle from imageJ")
#plt.hlines(np.mean(origami_angle_ok), 0, np.max(good_origamis)+1, color="red",
#           label="Mean")
#plt.xlabel("Index origami")
#plt.xticks(np.arange(0, max(good_origamis)+2, 5.0))
#plt.grid()
#plt.legend()
#plt.show()

plt.title("relative_angle")
plt.plot(good_origamis, relative_angle, 'o-',color="blue", label="relative_angle")
plt.hlines(np.mean(relative_angle), 0, np.max(good_origamis)+1, color="red",
           label="Mean")
plt.xlabel("Index origami")
plt.xticks(np.arange(0, max(good_origamis)+2, 5.0))
plt.grid()
plt.legend()
plt.show()

plt.title("modulation")
plt.plot(good_origamis, modulation, 'o-', color="orange", label="modulation")
plt.hlines(np.mean(modulation), 0, np.max(good_origamis)+1, color="red",
           label="Mean")
plt.xlabel("Index origami")
plt.xticks(np.arange(0, max(good_origamis)+2, 5.0))
plt.grid()
plt.legend()
plt.show()

plt.title("periods")
plt.plot(good_origamis, periods, 'o-', color="magenta", label="periods")
plt.hlines(np.mean(periods), 0, np.max(good_origamis)+1, color="red",
           label="Mean")
plt.xlabel("Index origami")
plt.xticks(np.arange(0, max(good_origamis)+2, 5.0))
plt.grid()
plt.legend()
plt.show()



#%% bad automatic way

# =============================================================================
# names = ["Origami angle Theun", "relative_angle", "Modulation", "Periods"]
# y_for_plot = dict()
# y_for_plot[names[0]] = origami_angle_ok
# y_for_plot[names[1]] = relative_angle
# y_for_plot[names[2]] = modulation
# y_for_plot[names[3]] = periods
# 
# colors = ["green", "blue", "orange", "magenta"]
# 
# for idx, val in enumerate(names):
#     y = y_for_plot[val]
#     plt.title(val)
#     plt.plot(good_origamis, y, 'o-', color=colors[idx], label=val)
#     plt.hlines(np.mean(y), 0, np.max(good_origamis)+1, color="red",
#                label="Mean")
#     plt.xlabel("Index origami")
#     plt.xticks(np.arange(0, max(good_origamis)+2, 5.0))
#     plt.grid()
#     plt.legend()
#     plt.show()
# =============================================================================

