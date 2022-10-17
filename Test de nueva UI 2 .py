# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:18:25 2022

@author: chiarelg

import pyqtgraph.examples
pyqtgraph.examples.run()
"""


# -*- coding: utf-8 -*-
"""
Demonstration of ScatterPlotWidget for exploring structure in tabular data.

The widget consists of four components:

1) A list of column names from which the user may select 1 or 2 columns
    to plot. If one column is selected, the data for that column will be
    plotted in a histogram-like manner by using pg.pseudoScatter(). 
    If two columns are selected, then the
    scatter plot will be generated with x determined by the first column
    that was selected and y by the second.
2) A DataFilter that allows the user to select a subset of the data by 
    specifying multiple selection criteria.
3) A ColorMap that allows the user to determine how points are colored by
    specifying multiple criteria.
4) A PlotWidget for displaying the data.

"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


import os
from tkinter import Tk, filedialog

root = Tk()
nametoload = filedialog.askopenfilename(filetypes=(("", "*.npz"), ("", "*.")))
root.withdraw()
folder = os.path.dirname(nametoload)
only_name = os.path.basename(nametoload)
print("file loaded: ", nametoload)
data = np.load(nametoload)    


parameters = ["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group"]


#pg.mkQApp()

## Make up some tabular data with structure
#data = np.empty(1000, dtype=[('x_pos', float), ('y_pos', float), 
#                             ('count', int), ('amplitude', float), 
#                             ('decay', float), ('type', 'U10')])
#strings = ['Type-A', 'Type-B', 'Type-C', 'Type-D', 'Type-E']
#print(strings)
#typeInds = np.random.randint(5, size=1000)
#data['type'] = np.array(strings)[typeInds]
#data['x_pos'] = np.random.normal(size=1000)
#data['x_pos'][data['type'] == 'Type-A'] -= 1
#data['x_pos'][data['type'] == 'Type-B'] -= 1
#data['x_pos'][data['type'] == 'Type-C'] += 2
#data['x_pos'][data['type'] == 'Type-D'] += 2
#data['x_pos'][data['type'] == 'Type-E'] += 20
#data['y_pos'] = np.random.normal(size=1000) + data['x_pos']*0.1
#data['y_pos'][data['type'] == 'Type-A'] += 3
#data['y_pos'][data['type'] == 'Type-B'] += 3
#data['amplitude'] = data['x_pos'] * 1.4 + data['y_pos'] + np.random.normal(size=1000, scale=0.4)
#data['count'] = (np.random.exponential(size=1000, scale=100) * data['x_pos']).astype(int)
#data['decay'] = np.random.normal(size=1000, scale=1e-3) + data['amplitude'] * 1e-4
#data['decay'][data['type'] == 'Type-A'] /= 2
#data['decay'][data['type'] == 'Type-E'] *= 3


# Create ScatterPlotWidget and configure its fields
#spw = pg.ScatterPlotWidget()
#spw.setFields([
#    ('x_pos', {'units': 'm'}),
#    ('y_pos', {'units': 'm'}),
#    ('count', {}),
#    ('amplitude', {'units': 'V'}),
#    ('decay', {'units': 's'}),    
#    ('type', {'mode': 'enum', 'values': strings}),
#    ])
#    
#spw.setData(data)
#spw.show()

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
          
    return list

spw = pg.ScatterPlotWidget()
print("SSSSS", data["frame"][5])
spw.setFields([
    ('frame', {'units': 'm'}),
    ('x', {'units': 'm'}),
    ('y', {}),
    ('photons', {'units': 'V'}),
    ('sx', {'units': 's'}),
    ('sy', {'units': 's'}),
    ('bg', {'units': 's'}),
    ('lpx', {'units': 's'}),
    ('lpy', {'units': 's'}),
    ('ellipticity', {'units': 's'}),
    ('net_gradient', {'units': 's'}),
    ('group', {'units': 's'}),
    ])
print("TTTTTTTTTTT", getList(data))    
spw.setData(data)
print("ppppppppppppppppp")
spw.show()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
