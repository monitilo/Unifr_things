import numpy as np
from lmfit import Model 
import matplotlib.pyplot as plt
import scipy.signal
import os
import pandas as pd

# read data from file
shape = 'aaa'
grid = 'AN11'
directory = "C:/Users/ZhuF/Desktop/20210503 new up left and right monomer 1uW 640nm/{}/{}/trace".format(shape, grid)
extension = ".dat"
start = "b.dat"
#files = [file for file in os.listdir(directory) if file.lower().endswith(extension) and not file.startswith(start)]
#files = [file for file in os.listdir(directory) if file.lower().endswith(extension) and not file.endswith(start)]

filename = 'C:/Data Confocal/CONFOCAL/Nrs_green/Decay green.dat'
irffilename = 'C:/Data Confocal/CONFOCAL/Nrs_green/Decay green.dat'
#irffilename = 'C:/Data Confocal/CONFOCAL/Nrs_green/Decay red.dat'

# define the single exponential model
def jumpexpmodel(x, tau1, ampl1, y0, irf):
    ymodel = ampl1*np.exp(-(x)/tau1)
    z = scipy.ndimage.filters.convolve(ymodel,irf,mode='wrap',origin=-int(len(ymodel)/2))
    z+=y0
    return z

data = pd.DataFrame(columns=["spot", "tau1", "tau1error", "ampl1", "ampl1error", "y0", "y0error", "Rsquared"])

#for file in files:
#try:
#    df=np.loadtxt("C:/Users/ZhuF/Desktop/20210503 new up left and right monomer 1uW 640nm/{}/{}/trace/{}".format(shape, grid, file), usecols=range(4))
df=np.loadtxt(filename, usecols=range(4))

x = df[4:101,0]

decay1 = df[4:101,2]
#    a = file.split('.')

#    df=np.loadtxt("C:/Users/ZhuF/Desktop/20210503 new up left and right monomer 1uW 640nm/{}/{}/trace/{}b.dat".format(shape, grid, a[0]), usecols=range(4))
df=np.loadtxt(irffilename,  usecols=range(4))


irf = df[4:101,2]
       
mod = Model(jumpexpmodel, independent_vars=('x', 'irf'))

#initialize the parameters - showing error
pars = mod.make_params(tau1=2, ampl1=3.669, y0=-1374)
pars['y0'].vary =True
mod.set_param_hint('tau1', value=2, min=0.1, max=10)
pars['ampl1'].vary =True

result = mod.fit(decay1, params=pars, method='leastsq', x=x, irf=irf)
tau1 = result.values.get('tau1')
tau1error = result.params['tau1'].stderr
ampl1 = result.values.get('ampl1')
ampl1error = result.values.get('ampl1')
y0 = result.values.get('y0')
y0error = result.values.get('y0')
Rsquared = 1 - result.residual.var() / np.var(decay1)
name = "a" # int(file.split(".")[0])

# plot results
plt.figure(5)
ax = plt.subplot(2,1,1)
plt.plot(x,decay1,'r-',label='I')
plt.plot(x,irf,'orange',label='IRF')
plt.plot(x,result.best_fit,'b--',label='Fit')
plt.ylabel("Intensity (counts)")
text = "t="+"%.3f" % tau1+"±{:.3f}ns\n".format(tau1error)+"$R^2$={:.3f}".format(Rsquared)
plt.text(0.65, 0.75, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.legend()
plt.subplot(2,1,2)
plt.scatter(x,result.residual)
plt.xlabel("Lifetime (ns)")
plt.ylabel("Residual")
plt.savefig("C:/Users/ZhuF/Desktop/20210503 new up left and right monomer 1uW 640nm/{}/{}/trace/{}.png".format(shape, grid, name), dpi=600)
plt.show()
data = data.append({'spot':name, 'tau1':tau1, 'tau1error':tau1error, 'ampl1':ampl1, 'ampl1error':ampl1error, 'y0':y0, 'y0error': y0error, 'Rsquared':Rsquared}, ignore_index=True)
#except:
#    pass
data = data.sort_values(['spot'], ascending = [True])
data.to_csv("C:/Data Confocal/CONFOCAL/Nrs_green/{}/lll.txt".format(shape), sep='\t', index=False, float_format='%.3f')
