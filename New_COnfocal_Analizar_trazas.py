# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:16:53 2022

@author: chiarelg
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
#%%

root = Tk()
file = filedialog.askopenfilename(filetypes=(("", "*.dat"), ("", "*.")))
root.withdraw()
file_folder = os.path.dirname(file)
file_name = os.path.basename(file)

print(file)


#%%
labels = ["Calculated IRF", "Overall Decay", "Roi Decay","Fitted Curve","Residuals"]
import numpy as np
data= np.loadtxt(file, dtype=str, delimiter='\t', skiprows=2)
data = np.char.replace(data, ',', '.')
# data = np.char.replace(data, '\'', '')
# data = np.char.replace(data, '---', '0')
# data = np.char.replace(data, 'b', '').astype(np.float64)
print("shape =", data.shape)
data_float = np.zeros(data.shape)
for j in range(len(data[0,:])):
    for i in range(len(data[:,j])):

        if data[i,j] == ' ' or data[i,j] == '---':
            data_float[i,j] = np.nan
        else:
            data_float[i,j] = float(data[i,j])

#%%
colors = ["black", "blue", "green", "red"]
lss = ["--", "-", "-", "-."]
for i in range(4):
    j=i*2
    plt.plot(data_float[:,j], data_float[:,j+1], label=labels[i], color = colors[i], ls=lss[i])
plt.legend()
plt.xlim([0,12])
plt.title("ATTO647N lifetime")
plt.yscale("log")
plt.show()
#%%

#%%Paper plot
colors = ["black", "blue", "green", "red"]
lss = ["--", "-", "-", "-."]
labeled = "Decay ATTO 647N" # "Decay Cy3B"# "Decay ATTO 647N"
colores = "darkgoldenrod" #  "forestgreen"
plt.plot(data_float[:,0], data_float[:,1]/max(data_float[:,1]), label= "IRF", color = "black", ls="-", linewidth = 1)
plt.plot(data_float[:,4], data_float[:,5]/max(data_float[:,5]), marker = "o" ,label= labeled, color = colores, linewidth = 0, markersize=5)
plt.plot(data_float[:,6], data_float[:,7]/np.nanmax(data_float[:,7]), label= "Fitted Curve", color = "Red", ls="-", linewidth = 1.5)
plt.legend()
plt.xlim([0,12])
# plt.title("{} lifetime".format(file[-35:-4]))
plt.xlabel("Photons arrival time (ns)")
plt.ylabel("#photons Norm")
plt.yscale("log")
plt.show()



#%% For the tables with the fits
#%%
# =============================================================================
# 
# root = Tk()
# file = filedialog.askopenfilename(filetypes=(("", "*.dat"), ("", "*.")))
# root.withdraw()
# file_folder = os.path.dirname(file)
# file_name = os.path.basename(file)
# 
# print(file)
# #%%
# # labels = ["Calculated IRF", "Overall Decay", "Roi Decay","Fitted Curve","Residuals"]
# import numpy as np
# data_fit= np.loadtxt(file, dtype=str, delimiter='\t', skiprows=0)
# data_fit = np.char.replace(data_fit, ',', '.')
# # data = np.char.replace(data, '\'', '')
# # data = np.char.replace(data, '---', '0')
# # data = np.char.replace(data, 'b', '').astype(np.float64)
# print("shape =", data_fit.shape)
# columnas = data_fit[0,1:]
# filas = []
# for i in range(1,len(data_fit)):
#     filas.append(data_fit[i,0])
#     
# data_fit_float = np.zeros(data_fit.shape)
# 
# for j in range(1,len(data_fit[0,:])):
#     for i in range(1,len(data_fit[:,j])):
# 
#         if data_fit[i,j] == ' ' or data_fit[i,j] == '---':
#             data_fit_float[i,j] = np.nan
#         else:
#             data_fit_float[i,j] = float(data_fit[i,j])
# #%%
# print("lifetimes = ", data_fit_float[1:,3])
# #%%
# for i in range(3):
#     plt.hist(data_fit_float[i], label=str(i))
# plt.legend()
# # plt.xlim([0,12])
# # plt.title("ATTO647N lifetime")
# =============================================================================
#%%

# =============================================================================
# import numpy as np
# a = np.random.random(10)
# songs = np.random.randint(1,888,100)
# plt.hist(songs, bins=888)
# 
# =============================================================================


