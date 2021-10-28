# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:41:42 2021

@author: chiarelg
"""

#%% Here I take the data and bin it by 25

import numpy as np
import matplotlib.pyplot as plt
import numpy, scipy.optimize


filename = 'C:/Analizando Imagenes/code/Aleksandra/odp_stretchprojectdataanalysispythoncode/b 2bp _traces-45.txt'
data = np.loadtxt(filename)

file_origami = np.loadtxt('C:/Analizando Imagenes/code/Aleksandra/odp_stretchprojectdataanalysispythoncode/ori_m table.txt')

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

try:
    t=0
    for n in range(graphs):
        fig, axs = plt.subplots(plot_files, plot_columns)
        for i in range(plot_files):
            for j in range(plot_columns):
                if t in origami_number:
                    V = avgdata[:,t]
                    
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
    
                    axs[i,j].plot(x, V, linewidth=2,
                                   label="{}".format((t)))
                    axs[i,j].plot(x1, data_fit, "r--", linewidth=1)  # ,
                    axs[i,j].legend(handlelength=0, handletextpad=0, fancybox=True)
                t+=1
except: pass
#except IOError as e:
#    print("I/O error({0}): {1}".format(e.errno, e.strerror))

plt.show()

#%% Delete the ones That look bad


add_to_delete = [11, 16]

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


difference = np.zeros(len(good_origamis))
for p in range(len(good_origamis)):
    difference[p] = cy5_angle[int(good_origamis[p])] - origami_angle_ok[p]

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
        relative_angle[i] = abs(360+difference[i])
    else:
        relative_angle[i] = 180+difference[i]

bines = len(relative_angle)//2
plt.hist(relative_angle, bins=bines, alpha=0.8, label="Relative")
plt.hist(origami_angle_ok, bins=bines, alpha=0.3, label="SuperRes_Angles")
plt.hist(cy5_angle[good_origamis], bins=bines, alpha=0.3, label="Cy5 angle")
plt.legend()
#print(diff_angles, "\n")



""" Data out: 
Colum 1: #traces  || column2: relative_angle || Column 3 modulation
    
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

#%% 

