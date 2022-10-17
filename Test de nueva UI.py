# -*- coding: utf-8 -*-
"""
Created on Thu Feb 03 16:21:25 2022

@author: chiarelg

import pyqtgraph.examples
pyqtgraph.examples.run()
"""

"""
This example demonstrates the use of pyqtgraph's dock widget system.

The dockarea system allows the design of user interfaces which can be rearranged by
the user at runtime. Docks can be moved, resized, stacked, and torn out of the main
window. This is similar in principle to the docking system built into Qt, but 
offers a more deterministic dock placement API (in Qt it is very difficult to 
programatically generate complex dock arrangements). Additionally, Qt's docks are 
designed to be used as small panels around the outer edge of a window. Pyqtgraph's 
docks were created with the notion that the entire window (or any portion of it) 
would consist of dockable components.

"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.console
import numpy as np

from pyqtgraph.dockarea import DockArea, Dock

import os
from tkinter import Tk, filedialog

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('pyqtgraph example: dockarea')

## Create docks, place them into the window one at a time.
## Note that size arguments are only a suggestion; docks will still have to
## fill the entire dock area and obey the limits of their internal widgets.
d1 = Dock("Dock1", size=(1, 1))     ## give this dock the minimum possible size
#d2 = Dock("Dock2 - Console", size=(500,300), closable=True)
d3 = Dock("Dock3", size=(500,400))
d4 = Dock("Dock4 (tabbed) - Plot", size=(500,200))
d5 = Dock("Dock5 - Image", size=(500,200))
d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
#area.addDock(d2, 'right')     ## place d2 at right edge of dock area
area.addDock(d3, 'right', d1)## place d3 at bottom edge of d1
area.addDock(d4, 'right')     ## place d4 at right edge of dock area
area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4

## Test ability to move docks programatically after they have been placed
area.moveDock(d4, 'top', d3)     ## move d4 to top edge of d2
area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
area.moveDock(d5, 'bottom', d4)     ## move d5 to top edge of d2


## Add widgets into each dock

## first dock gets save/restore buttons
w1 = pg.LayoutWidget()
label = QtGui.QLabel(""" -- DockArea Example -- 
This window has 6 Dock widgets in it. Each dock can be dragged
by its title bar to occupy a different space within the window 
but note that one dock has its title bar hidden). Additionally,
the borders between docks may be dragged to resize. Docks that are dragged on top
of one another are stacked in a tabbed layout. Double-click a dock title
bar to place it in its own window.
""")
saveBtn = QtGui.QPushButton('Save dock state')
restoreBtn = QtGui.QPushButton('Restore dock state')
w1.addWidget(label, row=0, col=0) 
w1.addWidget(saveBtn, row=1, col=0)
w1.addWidget(restoreBtn, row=2, col=0)

loadfileBtn = QtGui.QPushButton('load file')
w1.addWidget(loadfileBtn, 5,0,1,2) # row, col, height, width

nextpickBtn = QtGui.QPushButton('Next pick ')
w1.addWidget(nextpickBtn, row=7, col=0)
previouspickBtn = QtGui.QPushButton('Previous pick')
w1.addWidget(previouspickBtn, row=7, col=1)

d1.addWidget(w1)

state = None
def save():
    global state
    state = area.saveState()
    restoreBtn.setEnabled(True)
def load():
    global state
    area.restoreState(state)
def load_file(self):
    root = Tk()
    nametoload = filedialog.askopenfilename(filetypes=(("", "*.npz"), ("", "*.")))
    root.withdraw()
    folder = os.path.dirname(nametoload)
    only_name = os.path.basename(nametoload)
    print("file loaded: ", nametoload)

    open_data(nametoload)
#    make_hist(data, w3)
    return folder, only_name

def open_data(nametoload):
    data = np.load(nametoload)    
    pick_list = data["group"]
    frame = data["frame"]
    photons = data["photons"]
    pick_number = np.unique(pick_list)
    locs_of_picked = np.zeros(len(pick_number))
    photons_of_groups = dict()
    frame_of_groups = dict()
    
    for i in range(len(pick_number)):
        pick_id = pick_number[i]
        index_picked = np.where(pick_list == pick_id)
        frame_of_picked = frame[index_picked]
        locs_of_picked[i] = len(frame_of_picked)
        photons_of_groups[i] = photons[index_picked]
        frame_of_groups[i] = frame[index_picked]
    make_hist(photons_of_groups, w3)
    return photons_of_groups

def make_hist(data, widget):
    print("making hist")
    y,x = np.histogram(data[0], bins=np.linspace(0, np.max(data[0]), 50))
    widget.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150),edgecolor='red', linewidth=10 )
    
def next_pick():
    print("next pick")
    
def previous_pick():
    print("previous_pick")

saveBtn.clicked.connect(save)
restoreBtn.clicked.connect(load)
loadfileBtn.clicked.connect(load_file)
nextpickBtn.clicked.connect(next_pick)
previouspickBtn.clicked.connect(previous_pick)


#w2 = pg.console.ConsoleWidget()
#d2.addWidget(w2)

## Dock 3
#d3.hideTitleBar()
w3 = pg.PlotWidget(title="Plot inside dock with no title bar (no me gusto)")
#w3.plot(np.random.normal(size=100))

# make interesting distribution of values
vals = np.hstack([np.random.normal(size=500), np.random.normal(size=260, loc=4)])
# compute standard histogram
y,x = np.histogram(vals, bins=np.linspace(-3, 8, 40))
# Using stepMode=True causes the plot to draw two lines for each sample.
# notice that len(x) == len(y)+1
w3.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150),edgecolor='red', linewidth=10 )
# Now draw all points as a nicely-spaced scatter plot
y = pg.pseudoScatter(vals, spacing=0.15)
#plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5)
w3.plot(vals, y, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(255,0,0,150))


d3.addWidget(w3)



## Dock 4
w4 = pg.PlotWidget(title="Dock 4 plot")
w4.plot(np.random.normal(size=100))
d4.addWidget(w4)

w5 = pg.ImageView()
w5.setImage(np.random.normal(size=(100,100)))
d5.addWidget(w5)

w6 = pg.PlotWidget(title="Dock 6 plot")
w6.plot(np.random.normal(size=100))
d6.addWidget(w6)




state = area.saveState()
win.show()



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
