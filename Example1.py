# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:23:45 2019

@author: chiarelg
"""
#%% Connecting the z axis of the Imageview item
import pyqtgraph as pg
import numpy as np

pg.mkQApp()
view = pg.ImageView()
view.setImage(np.random.normal(size=(100,100,100)))
view.show()

def indexChanged():
    print (view.currentIndex)

view.sigTimeChanged.connect(indexChanged)
# %% Ejample for context menus
"""
Demonstrates adding a custom context menu to a GraphicsItem
and extending the context menu of a ViewBox.

PyQtGraph implements a system that allows each item in a scene to implement its 
own context menu, and for the menus of its parent items to be automatically 
displayed as well. 

"""
#import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.GraphicsWindow()
win.setWindowTitle('pyqtgraph example: context menu')


view = win.addViewBox()

# add two new actions to the ViewBox context menu:
zoom1 = view.menu.addAction('Zoom to box 1')
zoom2 = view.menu.addAction('Zoom to box 2')

# define callbacks for these actions
def zoomTo1():
    # note that box1 is defined below
    view.autoRange(items=[box1])
zoom1.triggered.connect(zoomTo1)

def zoomTo2():
    # note that box1 is defined below
    view.autoRange(items=[box2])
zoom2.triggered.connect(zoomTo2)



class MenuBox(pg.GraphicsObject):
    """
    This class draws a rectangular area. Right-clicking inside the area will
    raise a custom context menu which also includes the context menus of
    its parents.    
    """
    def __init__(self, name):
        self.name = name
        self.pen = pg.mkPen('r')
        
        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        
        # note that the use of super() is often avoided because Qt does not 
        # allow to inherit from multiple QObject subclasses.
        pg.GraphicsObject.__init__(self) 

    
    # All graphics items must have paint() and boundingRect() defined.
    def boundingRect(self):
        return QtCore.QRectF(0, 0, 10, 10)
    
    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawRect(self.boundingRect())
    
    
    # On right-click, raise the context menu
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()

    def raiseContextMenu(self, ev):
        menu = self.getContextMenus()
        
        # Let the scene add on to the end of our context menu
        # (this is optional)
        menu = self.scene().addParentContextMenus(self, menu, ev)
        
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        return True

    # This method will be called when this item's _children_ want to raise
    # a context menu that includes their parents' menus.
    def getContextMenus(self, event=None):
        if self.menu is None:
            self.menu = QtGui.QMenu()
            self.menu.setTitle(self.name+ " options..")
            
            green = QtGui.QAction("Turn green", self.menu)
            green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green
            
            blue = QtGui.QAction("Turn blue", self.menu)
            blue.triggered.connect(self.setBlue)
            self.menu.addAction(blue)
            self.menu.green = blue
            
            alpha = QtGui.QWidgetAction(self.menu)
            alphaSlider = QtGui.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider
        return self.menu

    # Define context menu callbacks
    def setGreen(self):
        self.pen = pg.mkPen('g')
        # inform Qt that this item must be redrawn.
        self.update()

    def setBlue(self):
        self.pen = pg.mkPen('b')
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a/255.)


# This box's context menu will include the ViewBox's menu
box1 = MenuBox("Menu Box #1")
view.addItem(box1)

# This box's context menu will include both the ViewBox's menu and box1's menu
box2 = MenuBox("Menu Box #2")
box2.setParentItem(box1)
box2.setPos(5, 5)
box2.scale(0.2, 0.2)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


# %% Example to Linear Region an Crosshair
        
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

#%% Example for histograms in pyqtgraph

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


