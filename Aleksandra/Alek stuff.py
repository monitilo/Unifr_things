# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:41:42 2021

@author: chiarelg
"""


a = [1,2,3,4,5,6,7]

print(a)

type(a)



import numpy as np
import matplotlib.pyplot as plt

#%%
b= np.arange(0,10,0.5)

b*2
len(b)
c=np.linspace(0,100,len(b))

j = np.sin(b)
#plt.plot()

#plt.plot(j,'m*-')
#

#data1.shape
#data = data1

#plt.plot(data1[:,30])

#data = np.loadtxt('C:/Analizando Imagenes/Si proyect/1_Cy5 alone bad TROLOX traces/FILTERED_1_TROLOX_30minUV_30minWait_cy5alone_2.1mW_640nm_1_traces-63.txt')
data = np.zeros((450, 35))
vector = np.linspace(0,2*np.pi,450)
for i in range(35):
    data[:,i] = np.sin(vector+i)

data.shape

plt.plot(data[:,2])

plt.plot(data[0:250,5],'.-')

data[5000:200025,0].shape

length = data.shape[0]
length
colums = data.shape[1]
colums


avgdata = np.zeros((int(length/25), colums))
avgdata.shape

for l in range(colums):
#    print(l)
    for i in range(int(length/25)):
        j = i*25
#        print(i,j)
        avgdata[i,l] = np.mean(data[j:j+25,l])

plt.plot(avgdata[:,0],"o")
plt.show()
#np.savetxt()
#%% Here I create Fake data and bin it by 25


data = np.loadtxt('C:/Analizando Imagenes/code/Aleksandra/odp_stretchprojectdataanalysispythoncode/b 2bp _traces-45.txt')

#data = np.zeros((450, 35))
vector = np.linspace(0,2*np.pi,450)
#for i in range(35):
#    data[:,i] = np.sin(vector+i)
#
#data[:,i//2] = np.tan(vector+i)

#data.shape
#np.mean(data[0:25, 0])

length=data.shape[0]
columns=data.shape[1]

avgdata = np.zeros(((int(length/25)),data.shape[1]))
avgdata.shape

for l in range(columns):
    for i in range(int(length/25)):
        j=i*25
        #print(i,j)
        avgdata[i,l] = (np.mean(data[j:j+25,l]))
      
theta=(np.linspace(0,170,len(avgdata)))
print 
print(avgdata)

plt.plot(theta,avgdata[:,0],"o")
plt.show()

tosavedata = np.zeros(((int(length/25)),data.shape[1] + 1))
tosavedata[:,0] = theta
tosavedata[:,1:] = avgdata

#np.savetxt('C:/Users/AdamczyA/Desktop/Pythons/Single_molecule/'+"average curves_"+name, tosavedata, fmt='%.3e')

#plt.plot(theta,avgdata[:,0],"o")
#plt.plot(tosavedata[:,0], tosavedata[:,1], '*')
#%%
#avgdata = np.loadtxt('C:/Analizando Imagenes/code/Aleksandra/odp_stretchprojectdataanalysispythoncode/average curves_e 2bp_traces-39.txt')
for i in range(columns):
    plt.plot(theta, avgdata[:,i])
    print(np.mean(avgdata[:,i]))

l=0
print(l)
plt.plot(theta, avgdata[:,l])
l=l+1



#%% Plot all of them in a no so crazy way

#columns = 29
plot_columns =  5  #  int(np.sqrt(columns))
plot_files =  4  #   int(np.ceil(columns/plot_columns))
graphs = int(np.ceil(columns / (plot_columns*plot_files)))


print(plot_files,plot_columns, "=", plot_files*plot_columns)

try:
    t=0
    for n in range(graphs):
        print("n",n)
        fig, axs = plt.subplots(plot_files, plot_columns)
        for i in range(plot_files):
#            print("i,j", i,j)
            for j in range(plot_columns):
#                print("i,j", i,j)
#                print("t",t)
                axs[i,j].plot(theta, avgdata[:,t],
                               label="{}".format((t)))
        #        axs[j,i].set_title("main{}".format(i+plot_columns*j))
                axs[i,j].legend(handlelength=0, handletextpad=0, fancybox=True)
                t+=1
except: pass
#except IOError as e:
#    print("I/O error({0}): {1}".format(e.errno, e.strerror))
plt.show()
    
#%% Delete the ones That look bad
todelete = [37,9]

avgdata[:,todelete] = np.nan
plt.plot(avgdata[:,1:5])
plt.legend([1,2,3,4,5])
plt.show()
#%% Find the max of each trace because modulation is harder



#plt.plot(theta,avgdata[:,2])
#theta[np.where(avgdata[:,2] == np.max(avgdata[:,2]))]

cy5_angle = np.zeros(columns)
for i in range(columns):
    if i not in todelete:
#        print(i)
        cy5_angle[i] = theta[np.where(avgdata[:,i] == np.max(avgdata[:,i]))][0]

#plt.plot(cy5_angle,'-o')
plt.hist(cy5_angle)
plt.show()


#%%
"""
calculate modulation
fiting sin2(data). {A*sin(pi*(x-xc)/w1)**2 + y0 } ==> get amplitud and mean.
w1 should be 180 ± 10
modulation = (Amp/2) / (y0 + Amp/2)

"""


#plt.show()

import numpy, scipy.optimize

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])/2  # excluding the zero frequency "peak", which is related to offset. Is /2 because is sin^2
    guess_offset = numpy.min(yy)
    guess_amp = (numpy.std(yy)) * 2.5  # it works sometimes
#    guess_phase = np.array(tt[np.where(yy==np.max(yy))[0]])[0]
    minimo = np.array(tt[np.where(yy==np.min(yy))[0]])[0]
    guess_phase = minimo
    print(guess_phase)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, guess_phase, guess_offset])
#    guess_bounds = ([0,0,0,0], [numpy.max(yy), 1, 360, numpy.max(yy)])

    def sin2func(t, A, w, p, y0):  return A * numpy.sin((np.pi/w)*(t-p))**2 + y0
    popt, pcov = scipy.optimize.curve_fit(sin2func, tt, yy, p0=guess)  #, bounds=guess_bounds)
    A, w, p, y0 = popt
    print("guess A, w, p, y0", guess,"\n vs fit " ,popt)
    f =  w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin((np.pi/w)*(t-p))**2 + y0
    return {"amp": A, "period": w, "phase": p, "offset": y0, "freq": f, "aaa": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}

import pylab as plt2

#N, amp, omega, phase, offset, noise = len(avgdata[:,i]), np.max(avgdata[:,i]), 180, 3., 5., 2
#N, amp, omega, phase, offset, noise = len(avgdata[:,i]), 1., .4, .5, 4., .21
#N, amp, omega, phase, offset, noise = 200, 1., 20, .5, 4., 1
#tt = numpy.linspace(0, 10, N)
#tt2 = numpy.linspace(0, 10, 10*N)
#yy = amp*numpy.sin(omega*tt + phase)**2 + offset

i=7

noise= 5*+np.random.random(len(avgdata[:,i]))
plt.plot(theta, avgdata[:,i]+noise)
res = fit_sin(theta, avgdata[:,i]+ noise)
print( "Amplitude=%(amp)s, Angular freq.=%(period)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

print(np.pi/res['period'])


plt2.plot(theta, avgdata[:,i], "ok", label="data")
plt2.plot(theta, res["fitfunc"](theta), "r-", label="fit curve", linewidth=2)
plt2.legend(loc="best")
plt2.show()


print(res["phase"], res["phase"])



#%%

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x=theta
def my_sin2(t,peroid,amplitude,phase,offset):
           return (amplitude*(sp.sin((t-phase)*sp.pi/peroid)))**2 + offset

def fit_sin2(x_data, y_data):
    guess_peroid = 180  # could be better
    guess_amplitude = np.max(avgdata[:,i])/2.
    minimo = np.array(x[np.where(V==np.min(V))[0]])[0]
    guess_phase = minimo
    guess_offset = np.min(avgdata[:,i])
    guess_bounds = ([100,0,0,0], [260, numpy.max(V), 180, numpy.max(V)])
    
    p0 =[guess_peroid, guess_amplitude, guess_phase, guess_offset]
    fit = curve_fit(my_sin2,x, V, p0=p0, bounds=guess_bounds)
    


for i in range(len(avgdata)-17):
    V=avgdata[:,i]

    
    guess_peroid= 180
    guess_amplitude = np.max(avgdata[:,i])/2.
    minimo = np.array(x[np.where(V==np.min(V))[0]])[0]
    guess_phase = minimo
    guess_offset = 2
    guess_bounds = ([100,0,0,0], [260, numpy.max(V), 180, numpy.max(V)])
    
    p0 =[guess_peroid, guess_amplitude, guess_phase, guess_offset]
    fit = curve_fit(my_sin2,x, V, p0=p0, bounds=guess_bounds)
    print ('Guess paramters are:', p0)
    print ('The fit paramters are:', fit[0])
    x1 = sp.linspace(0,170,1000)
    data_fit = my_sin2(x1,*fit[0])
    
    plt.figure(i)
    plt.plot(x1,data_fit)
    plt.errorbar(x,V,fmt='x')
    plt.show()
    
    print("the guess max is in {}".format(minimo+90))
    print("the fitted max is in {}".format(fit[0][2]+90))

#%%


#cy5_angle
#%%
t = np.linspace(0,180, 100)
A, w, t0, y0 = 1, 180, 45, 1


testing = sin2func(t, A, w, t0, y0)  

t[np.where(testing==np.max(testing))]

plt.plot(t,testing)
#plt.plot(t, fit_sin(t, testing)["fitfunc"](t))
np.pi/fit_sin(t, testing)["period"]

#%%
    
#columns = 29
plot_columns =  5  #  int(np.sqrt(columns))
plot_files =  4  #   int(np.ceil(columns/plot_columns))
graphs = int(np.ceil(columns / (plot_columns*plot_files)))


print(plot_files,plot_columns, "=", plot_files*plot_columns)

i=0
try:
    t=0
    for n in range(graphs):
        print("n",n)
        fig, axs = plt.subplots(plot_files, plot_columns)
        for i in range(plot_files):
#            print("i,j", i,j)
            for j in range(plot_columns):
                res = fit_sin(theta, avgdata[:,t])
#                print("i,j", i,j)
#                print("t",t)
                axs[i,j].plot(theta, avgdata[:,t], linewidth=2,
                               label="{}".format((t)))
                axs[i,j].plot(theta, res["fitfunc"](theta), "r--", linewidth=1)  # ,
                   #  label= "{:0.0f}".format(np.pi/res['period']))
        #        axs[j,i].set_title("main{}".format(i+plot_columns*j))
                axs[i,j].legend(handlelength=0, handletextpad=0, fancybox=True)
                t+=1
except: pass
#except IOError as e:
#    print("I/O error({0}): {1}".format(e.errno, e.strerror))
plt.show()

#%%

# =============================================================================
# import numpy, scipy.optimize
# 
# def fit_sin(tt, yy):
#     '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
#     tt = numpy.array(tt)
#     yy = numpy.array(yy)
#     ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
#     Fyy = abs(numpy.fft.fft(yy))
#     guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_amp = numpy.std(yy) * 2.**0.5
#     guess_offset = numpy.mean(yy)
#     guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])
# 
#     def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p)**2 + c
#     popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, w, p, c = popt
#     f = w/(2.*numpy.pi)
#     fitfunc = lambda t: A * numpy.sin(w*t + p)**2 + c
#     return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}
# 
# import pylab as plt2
# 
# N, amp, omega, phase, offset, noise = len(avgdata[:,i]), np.max(avgdata[:,i]), (180), 3., 5., 0
# #N, amp, omega, phase, offset, noise = len(avgdata[:,i]), 1., .4, .5, 4., .21
# #N, amp, omega, phase, offset, noise = 200, 1., 20, .5, 4., 1
# 
# res = fit_sin(theta, avgdata[:,i])
# print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )
# 
# plt2.plot(theta, avgdata[:,i], "ok", label="data")
# plt2.plot(theta, res["fitfunc"](theta), "r-", label="fit curve", linewidth=2)
# plt2.legend(loc="best")
# plt2.show()
# 
# 45-res["phase"]*np.pi*2
# =============================================================================
#%% Determine the Relative_angle:
"""
first: change the angle range from -180 _ 180 to 0_360
"""


""" Data in "origamis_angles_m": 
Colum 1: #traces  || column2: origamis_angles_m
    """
file_origami = np.loadtxt('C:/Analizando Imagenes/code/Aleksandra/odp_stretchprojectdataanalysispythoncode/ori_m table.txt',skiprows=1) #  fake data np.copy(cy5_angle)
origami_angle_m = file_origami[:,1]
#diff_angles = np.zeros(len(cy5_angle))
#for i in range(len(origami_angle_m)):
#    diff_angles[i] =  (np.random.rand()*180)*(-1)**i
#    origami_angle_m[i] = cy5_angle[i] + diff_angles[i]
    
origami_angle_ok = np.copy(origami_angle_m)

for i in range(len(origami_angle_m)):
    if origami_angle_m[i] < 0:
        origami_angle_ok[i] = 360 + origami_angle_m[i]
    else:
        origami_angle_ok[i] = origami_angle_m [i]  # dah

"""
Then, calculate the difference between origami_angle_ok and cy5_angle
"""
difference = cy5_angle - origami_angle_ok
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

bines = len(cy5_angle)//4
plt.hist(relative_angle, bins=bines, alpha=0.8, label="relative")
#plt.hist(diff_angles, bins=bines, alpha=0.3, label="dif from sr")
plt.hist(cy5_angle, bins=bines, alpha=0.3, label="cy5 angle")
plt.legend()
#print(diff_angles, "\n")






""" Data out: 
Colum 1: #traces  || column2: relative_angle || Column 3 modulation
    
    """


