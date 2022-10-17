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
file = filedialog.askopenfilename(filetypes=(("", "*.txt"), ("", "*.")))
root.withdraw()
file_folder = os.path.dirname(file)
file_name = os.path.basename(file)

print(file)

#%%


labels = ["Calculated IRF", "Overall Decay", "Roi Decay","Fitted Curve","Residuals"]

#Copied the data from Origin....
data= np.loadtxt(file, delimiter='\t', skiprows = 0)
# data = np.loadtxt(file)
# data = np.load(file)
# x = np.fromfile(file, dtype=dt)
plt.plot(data[:,0], data[:,1])
#%%

plt.plot(data[:,0], data[:,1], '.', label="Roi decay")
plt.plot(data[:,2], data[:,3], 'r', label="fit")
plt.xlim([0,12.5])
plt.title("Norm Decay")


