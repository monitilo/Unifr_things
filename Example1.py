# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:23:45 2019

@author: chiarelg
"""


#import initExample ## Add path to library (just for examples; you do not need this)
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
#from pyqtgraph.Point import Point

#generate layout
app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: crosshair')
label = pg.LabelItem(justify='right')
win.addItem(label, row=0,col=0)

label2 = pg.LabelItem(justify='right')
win.addItem(label2, row=0,col=1)

label3 = pg.LabelItem(justify='right')
win.addItem(label3, row=0,col=2)
#p1 = win.addPlot(row=1, col=0)
p2 = win.addPlot(row=1, col=0, rowspan=1, colspan=3)

region = pg.LinearRegionItem(pen='g')
region.setZValue(10)
# Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
# item when doing auto-range calculations.
p2.addItem(region, ignoreBounds=True)

region2 = pg.LinearRegionItem(pen='r')
region2.setZValue(10)
p2.addItem(region2, ignoreBounds=True)

#pg.dbg()
#p1.setAutoVisible(y=True)


#create numpy arrays
#make the numbers large to show that the xrange shows data from 10000 to all the way 0
data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)

#p1.plot(data1, pen="r")
#p1.plot(data2, pen="g")

p2.plot(data1, pen="w")

def update():
    region.setZValue(10)
    minX, maxX = region.getRegion()
    region2.setZValue(10)
    minX2, maxX2 = region2.getRegion()
#    p1.setXRange(minX, maxX, padding=0)
#    print(minX,maxX)
    medio = np.nanmean(data1[int(minX):int(maxX)])
    medio2 = np.nanmean(data1[int(minX2):int(maxX2)])

    label2.setText("<span style='font-size: 12pt'> <span style='color: green'>Mean=%0.1f</span>" % (medio))
    label3.setText("<span style='font-size: 12pt'> <span style='color: red'>Mean=%0.1f</span>" % (medio2))

region.sigRegionChanged.connect(update)
region2.sigRegionChanged.connect(update)

def updateRegion(window, viewRange):
    rgn = viewRange[0]
    region.setRegion(rgn)
    region2.setRegion(rgn)

#p1.sigRangeChanged.connect(updateRegion)

region.setRegion([1000, 2000])
region2.setRegion([5000, 6000])

#cross hair
vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
p2.addItem(vLine, ignoreBounds=True)
p2.addItem(hLine, ignoreBounds=True)


vb = p2.vb

def mouseMoved(evt):

    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    if p2.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        index = int(mousePoint.x())
        if index > 0 and index < len(data1):
            label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())



proxy = pg.SignalProxy(p2.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
#p1.scene().sigMouseMoved.connect(mouseMoved)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

#%%

# -*- coding: utf-8 -*-
"""
In this example we draw two different kinds of histogram.
"""
#import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.GraphicsWindow()
win.resize(800,350)
win.setWindowTitle('pyqtgraph example: Histogram')
plt1 = win.addPlot()
plt2 = win.addPlot()

## make interesting distribution of values
vals = np.hstack([np.random.normal(size=500), np.random.normal(size=260, loc=4)])

## compute standard histogram
y,x = np.histogram(vals, bins=np.linspace(-3, 8, 40))

## Using stepMode=True causes the plot to draw two lines for each sample.
## notice that len(x) == len(y)+1
plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))

## Now draw all points as a nicely-spaced scatter plot
y = pg.pseudoScatter(vals, spacing=0.15)
#plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5)
plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(0,0,255,150))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


