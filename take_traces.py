# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:24:32 2019

@author: chiarelg
"""
# %%
"""
Now I will try with a Gaussian fit and some wird arguments.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import  io #,img_as_float, data

#FILE = 'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif'
FILE = 'C:/Origami testing Widefield/2019-10-16/Green_Reference-11mW(5,4)_532nm-100ms_1/Green_Reference-11mW(5,4)_532nm-100ms_1_MMStack_Pos0.ome.tif'
plot = True
tiff = io.imread(FILE)

Nmeanstart = 50
Nmeanfinish = 70
im = np.sum(tiff[Nmeanstart:Nmeanfinish], axis=0)

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
#image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=5)

# display results
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original (Sum of '+str(Nmeanfinish-Nmeanstart)+' frames)')

#ax[1].imshow(image_max, cmap=plt.cm.gray)
#ax[1].axis('off')
#ax[1].set_title('Maximum filter')

ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()

plt.show()

maxvalues = []
for i in range(len(coordinates[:,0])):
    maxvalues.append(im[coordinates[i,0],coordinates[i,1]])

nomax = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]



aux = np.arange(len(maxvalues))
goodmax = np.delete(aux,nomax)

muymax = np.where(np.array(maxvalues) > 1.5*np.mean(np.array(maxvalues)[goodmax]))
othermax = np.delete(aux,muymax)

toerase = np.sort(np.append(nomax, muymax))
finalmax = np.delete(aux,toerase)

print( "\n \n", "len maxvalues",len(maxvalues),"\n len nomax", len(nomax))
print(" Number of peaks", len(maxvalues)-len(nomax), len(goodmax))

plt.figure("que onda")
plt.title(len(maxvalues))
plt.imshow(im, cmap=plt.cm.gray)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')

#lll=[]
#for j in range(len(coordinates[:,1])):
#    if j not in nomax:
#        lll.append(j)
#        plt.plot(coordinates[j, 1], coordinates[j, 0], 'b.')
#print(lll)

newcoordinateX = coordinates[finalmax,0]
newcoordinateY = coordinates[finalmax,1]

for j in finalmax:
#    plt.plot(coordinates[j, 1], coordinates[j, 0], 'm*')
    plt.plot(newcoordinateY, newcoordinateX, 'm*')
for j in goodmax:
    plt.plot(coordinates[j, 1], coordinates[j, 0], 'b.')

#%%
roisize = 5  # es el doble de esto
roi = dict()
roi2 = dict()

deletear = []
p=0
for i in finalmax:
    aaa=0
    suming = 0
    roi[p] = []
    try:
        for x in range(-roisize, roisize+1): # goes from (-4,-4) to (4,4)
            for y in range(-roisize,roisize+1):
#                print(aaa, x, y, "i=",i)
                suming += im[coordinates[i,0]+x,coordinates[i,1]+y]
                aaa += 1
        roi[p].append(suming)
    except:
        deletear.append(p)
        print("i",i,"p", p, "roi[i]", roi[p],deletear)
#        deletear.append(i)
    p += 1

for d in deletear:
    print("dddddeletearrrrr", d)
    del roi[d]


for i in range(len(newcoordinateX)):
#    roi2[i] = np.sum(im[newcoordinateX[i]:newcoordinateX[i]+roisize+1, newcoordinateY[i]:newcoordinateY[i]+roisize+1])
#    roi2[i] += np.sum(im[newcoordinateX[i]-roisize:newcoordinateX[i], newcoordinateY[i]-roisize:newcoordinateY[i]])
#    roi2[i] += np.sum(im[newcoordinateX[i]:newcoordinateX[i]+roisize+1, newcoordinateY[i]-roisize:newcoordinateY[i]])
#    roi2[i] += np.sum(im[newcoordinateX[i]-roisize:newcoordinateX[i], newcoordinateY[i]:newcoordinateY[i]+roisize+1])
    smallimup = np.concatenate((im[newcoordinateX[i]-roisize:newcoordinateX[i], newcoordinateY[i]-roisize:newcoordinateY[i]],
                              im[newcoordinateX[i]-roisize:newcoordinateX[i], newcoordinateY[i]:newcoordinateY[i]+roisize]),
                              axis=1)
    smallimdown = np.concatenate((im[newcoordinateX[i]:newcoordinateX[i]+roisize, newcoordinateY[i]-roisize:newcoordinateY[i]],
                                  im[newcoordinateX[i]:newcoordinateX[i]+roisize, newcoordinateY[i]:newcoordinateY[i]+roisize]),
                                  axis=1)
    smallim = np.concatenate((smallimup,smallimdown),axis=0)
    roi2[i] = smallim

#print(roi[0],np.sum(roi2[0]))
plt.imshow(roi2[2])

# %%
j=1
N=10  # len(roi2)  # 132
all_params = np.zeros((N,5))
Rsquared = np.zeros((N))
for j in range(N):
    data = np.transpose(roi2[j])
    params = fitgaussian(data)
    fit = gaussian(*params)
    new_params = fitgaussian(roi2[j])
    all_params[j] = new_params
    (height, x, y, width_x, width_y) = new_params
    print("j=",j," \n new_params", new_params)
    
    xv = np.linspace(0, roisize*2, roisize*2)
    yv = np.linspace(0, roisize*2, roisize*2)
    xy_mesh = np.meshgrid(xv, yv)
    X, Y = np.meshgrid(xv, yv)
    #fig, ax = plt.subplots()
    #p = ax.pcolor(X, Y, np.transpose(roi2[j]), cmap=plt.cm.jet)
    #fig.colorbar(p)
    #ax.set_xlabel('x [um]')
    #ax.set_ylabel('y [um]')
    
    
    #xg = int(np.floor(x))
    #yg = int(np.floor(y))
    #
    #resol = 2
    #xsum, ysum = 0, 0
    #for i in range(resol):
    #    for j in range(resol):
    #        ax.text(X[xg+i, yg+j], Y[xg+i, yg+j], "Ga", color='m')
    ##                    ax.text(X2[xc+i, yc+j], Y2[xc+i, yc+j], "Ga", color='m')
    #        xsum = X[xg+i, yg+j] + xsum
    #        ysum = Y[xg+i, yg+j] + ysum
    #xmean = xsum / (resol**2)
    #ymean = ysum / (resol**2)
    #ax.text(xmean, ymean, "✔", color='r')
    #ax.set_title("Centro en x={:.3f}, y={:.3f}".format(xmean, ymean))
    #plt.text(0.95, 0.05, """
    #        x : %.1f
    #        y : %.1f """ % (X[xg, yg], Y[xg, yg]),
    #         fontsize=16, horizontalalignment='right',
    #         verticalalignment='bottom', transform=ax.transAxes)
    
    
    
    
    plt.matshow(data, cmap=plt.cm.gist_earth_r, origin='lower',
                            interpolation='none',
                            extent=[xv[0], xv[-1], yv[0], yv[-1]])
    plt.colorbar()
    plt.grid(True)
    plt.contour(fit(*np.indices(data.shape)),
                cmap=plt.cm.copper, interpolation='none',
                extent=[xv[0], xv[-1], yv[0], yv[-1]])
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params
    
    xc = int(np.floor(x))
    yc = int(np.floor(y))
    resol = 2
    xsum, ysum = 0, 0
    for i in range(resol):
        for j in range(resol):
    #        ax.text(X[xc+i, yc+j], Y[xc+i, yc+j], "Ga", color='m')
            try:
                xsum = X[xc+i, yc+j] + xsum
                ysum = Y[xc+i, yc+j] + ysum
            except:
                pass
                
    xmean = xsum / (resol**2)
    ymean = ysum / (resol**2)
    ax.text(xmean, ymean, "✔", color='r')
    #            Normal = self.scanRange / self.numberofPixels  # Normalizo
    #            ax.set_title((self.xcm*Normal + float(initPos[0]),
    #                          self.ycm*Normal + float(initPos[1])))
    #plt.text(0.95, 0.05, """x : %.2f y : %.2f """
    #         % (xmean, ymean),  # X[xc, yc], Y[xc, yc]
    #         fontsize=16, horizontalalignment='right',
    #         verticalalignment='bottom', transform=ax.transAxes)
    #print("x", xv[int(x)], X[xc, yc], xmean)
    #            Normal = self.scanRange / self.numberofPixels  # Normalizo
    ax.set_title("Centro en x={:.3f}, y={:.3f}".format(xmean, ymean))
    plt.show()

    # manually calculate R-squared goodness of fit
#    fit_residual = roi2[j] - gaussian_2d(xy_mesh, *new_params).reshape(np.outer(xv,yv).shape)
#    print(fit_residual)
#    Rsquared[j] = 1 - np.var(fit_residual)/np.var(roi2[j])

print(all_params[0,:])  # (height, x, y, width_x, width_y)
plt.figure("aaaaa")
c = ['k', 'b', 'r', 'c', 'm']
labeled = ['height', 'x', 'y', 'width_x', 'width_y']
for l in range(3,len(all_params[0,:])):
    plt.plot(all_params[:,l], '*-', color=c[l], label=labeled[l])
plt.grid()
plt.legend()
plt.show()

#plt.figure("R2")
#plt.plot(Rsquared, '.-r')

#%% Another way to 2d gauss fit. Do not like it

fit_params = np.zeros((N,5))
fit_Rsquared = np.zeros((N))
fit_errors = np.zeros((N,5))
l=2
for l in range(N):
    xv = np.linspace(0, roisize*2, roisize*2)
    yv = np.linspace(0, roisize*2, roisize*2)
    xy_mesh = np.meshgrid(xv, yv)
    (X,Y) = np.meshgrid(xv, yv)
    
    amp = all_params[l,0]
    xc, yc = all_params[l,1], all_params[l,2]
    sigma_x, sigma_y = all_params[l,3], all_params[l,4]
    
    # define some initial guess values for the fit routine
    #guess_vals = [all_params[l,0], all_params[l,1], all_params[l,2], all_params[l,3], all_params[l,4]]
    guess_vals = [amp, xc, yc, sigma_x, sigma_y]
     
    # perform the fit, making sure to flatten the noisy data for the fit routine 
    fit_params[l], cov_mat = curve_fit(gaussian_2d, xy_mesh, np.ravel(roi2[l]), p0=guess_vals)

    # calculate fit parameter errors from covariance matrix
    fit_errors = np.sqrt(np.diag(cov_mat)) 
     
    # manually calculate R-squared goodness of fit
    fit_residual = roi2[l] - gaussian_2d(xy_mesh, *fit_params[l]).reshape(np.outer(xv,yv).shape)
    fit_Rsquared[l] = 1 - np.var(fit_residual)/np.var(roi2[l])

    print('Fit R-squared:', fit_Rsquared[l], '\n')
    print('Fit Amplitude:', fit_params[l][0], '\u00b1', fit_errors[0])
    print('Fit X-Center: ', fit_params[l][1], '\u00b1', fit_errors[1])
    print('Fit Y-Center: ', fit_params[l][2], '\u00b1', fit_errors[2])
    print('Fit X-Sigma:  ', fit_params[l][3], '\u00b1', fit_errors[3])
    print('Fit Y-Sigma:  ', fit_params[l][4], '\u00b1', fit_errors[4])
    
    
    # set contour levels out to 3 sigma
    sigma_x_pts = xc + [sigma_x, 2*sigma_x, 3*sigma_x]
    sigma_y_pts = yc + [sigma_y, 2*sigma_y, 3*sigma_y]
    sigma_xy_mesh = np.meshgrid(sigma_x_pts, sigma_y_pts)
    
    contour_levels = gaussian_2d(sigma_xy_mesh, amp, xc, yc, 
                                 sigma_x, sigma_y).reshape(sigma_xy_mesh[0].shape)
    contour_levels = list(np.diag(contour_levels)[::-1])
     

    # make labels for each contour
    labels = {}
    label_txt = [r'$3\sigma$', r'$2\sigma$', r'$1\sigma$']
    for level, label in zip(contour_levels, label_txt):
        labels[level] = label
     
    # plot the function with noise added
    plt.figure(figsize=(6,6))
    plt.title('probability coverage')
    plt.imshow(np.transpose(roi2[l]), origin='lower')
    CS = plt.contour(roi2[l], levels=contour_levels, colors=['red', 'orange', 'white'])
    plt.clabel(CS, fontsize=16, inline=1, fmt=labels)
    plt.grid(visible=False)

    ax = plt.gca()
    xc = int(np.floor(fit_params[l][1]))
    yc = int(np.floor(fit_params[l][2]))
    resol = 2
    xsum, ysum = 0, 0
    for i in range(resol):
        for j in range(resol):
    #        ax.text(X[xc+i, yc+j], Y[xc+i, yc+j], "Ga", color='m')
            xsum = X[xc+i, yc+j] + xsum
            ysum = Y[xc+i, yc+j] + ysum
    xmean = xsum / (resol**2)
    ymean = ysum / (resol**2)
    ax.text(xmean, ymean, "✔", color='r')
    
    plt.show()

plt.figure("aaaa")
c = ['k', 'b', 'r', 'c', 'm']
labeled = ['Amplitude', 'x-center', 'y-center', 'x-Sigma', 'Y-sigma']
for l in range(1,len(fit_params[0,:])):
    plt.plot(fit_params[:,l], '*-', color=c[l], label=labeled[l])
plt.grid()
plt.legend()
plt.show()

plt.figure("R2")
plt.plot(fit_Rsquared, 'o-g',label="R2")

#%%
import math as mth
import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model

def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):
    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh
    
    # make the 2D Gaussian matrix
    gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)
    
    # flatten the 2D Gaussian down to 1D
    return np.ravel(gauss)

#%%
from scipy.signal import convolve2d

im2 = convolve2d(im,np.ones((3,3),dtype=int),'same')

plt.imshow(im2, cmap=plt.cm.gray)
# %%
    
    
#trace = {}
#
#p = 0
#deletear = []
#for i in range(len(coordinates[:,0])):
#    if i not in nomax:
#
#        trace[p] = []

#        for f in range(tiff.shape[0]):
#            suming = 0

#            try:
#                for x in range(-4,5): # goes from (-4,-4) to (4,4)
#                    for y in range(-4,5):  # its 81 numbers. 9x9 box
#
#                        suming += tiff[f][coordinates[i+x,0],coordinates[i+y,1]]
#                trace[p].append(suming)
#            except:
##                print("i",i, "f", f, "p",p,"trace[p]", trace[p])
#                deletear.append(p)
##                trace[p].append(np.nan)
#
#        p += 1
#
#for d in deletear:
#    print("dddddeletearrrrr", d)
#    del trace[d]
#
#print("len trace",len(trace))
#print("len maxvalues-nomax",len(maxvalues)-len(nomax))


# %%
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    
    from scipy import optimize
    
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

# %%

# =============================================================================
# # %%
# """
# My old program to find spots and extract the trace from them.
# the second part try to analize this trace.
# 
# 
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage as ndi
# from skimage.feature import peak_local_max
# from skimage import  io #,img_as_float, data
# 
# 
# #def take_traces(FILE, plot = False):
# #    """ input: .tif file with spots
# #    Find all the interesting peaks in the second frame
# #    and all the traces in these points"""
# 
# 
# #im = img_as_float(data.coins())
# FILE = 'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif'
# plot = True
# tiff = io.imread(FILE)
# 
# im = tiff[1]
# 
# # image_max is the dilation of im with a 20*20 structuring element
# # It is used within peak_local_max function
# image_max = ndi.maximum_filter(im, size=20, mode='constant')
# 
# # Comparison between image_max and im to find the coordinates of local maxima
# coordinates = peak_local_max(im, min_distance=5)
# 
# # display results
# fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
# ax = axes.ravel()
# ax[0].imshow(im, cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('Original')
# 
# ax[1].imshow(image_max, cmap=plt.cm.gray)
# ax[1].axis('off')
# ax[1].set_title('Maximum filter')
# 
# ax[2].imshow(im, cmap=plt.cm.gray)
# ax[2].autoscale(False)
# ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
# ax[2].axis('off')
# ax[2].set_title('Peak local max')
# 
# fig.tight_layout()
# 
# plt.show()
# 
# 
# 
# 
# maxvalues = []
# for i in range(len(coordinates[:,0])):
#     maxvalues.append(im[coordinates[i,0],coordinates[i,1]])
# 
# #print(maxvalues)
# 
# nomax = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]
# 
# print( "\n \n", "len maxvalues",len(maxvalues),"\n len nomax", len(nomax))
# print(" Number of peaks",len(maxvalues)-len(nomax))
# 
# plt.figure("que onda")
# plt.title(len(maxvalues))
# plt.imshow(im, cmap=plt.cm.gray)
# plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
# 
# for j in range(len(coordinates[:,1])):
#     if j not in nomax:
#         plt.plot(coordinates[j, 1], coordinates[j, 0], 'b.')
# 
# trace = {}
# 
# p = 0
# for i in range(len(coordinates[:,0])):
#     if i not in nomax:
# 
#         trace[p] = []
#         for f in range(tiff.shape[0]):
#             suming = 0
#             deletear = []
#             try:
#                 for x in range(-4,5): # goes from (-4,-4) to (4,4)
#                     for y in range(-4,5):  # its 81 numbers. 9x9 box
# 
#                         suming += tiff[f][coordinates[i+x,0],coordinates[i+y,1]]
#                 trace[p].append(suming)
#             except:
# #                print("i",i, "f", f, "p",p,"trace[p]", trace[p])
#                 deletear.append(p)
# #                trace[p].append(np.nan)
# 
#         p += 1
# 
# for d in deletear:
#     print("dddddeletearrrrr", d)
#     del trace[d]
# 
# print("len trace",len(trace))
# print("len maxvalues-nomax",len(maxvalues)-len(nomax))
# 
# 
# Nframes = len(trace[0])
# Ntraces = len(trace)
# 
# 
# realmaxs = []
# realmins = []
# 
# hipart = {}
# lowpart = {}
# 
# lessmax = 0.97
# moremin = 1.03
# 
# N = Ntraces
# 
# PLOT = plot
# graph = np.linspace(1, Ntraces-5, num=10, endpoint=False, dtype=int)
# for i in range(N):
# #    print(i)
#     if PLOT and i in graph:
#         plt.figure(i)
#         plt.plot(trace[i], '*b')
# 
#     H = True
#     L = True
#     mm = []
#     ml = []
#     hipart[i] = [trace[i][0]]
#     hipart[i].append(trace[i][1])
#     
#     lowpart[i] = [trace[i][-1]]
#     for j in range(1, Nframes-3):
# 
#         if H:
#             mm.append(trace[i][j])
# 
#             if (np.mean(np.array(mm)))*lessmax < trace[i][j+1] < (np.mean(np.array(mm)))*moremin:
#                 hipart[i].append(trace[i][j+1])
# 
#             else:
#                 if (np.mean(np.array(mm)))*lessmax < trace[i][j+2] < (np.mean(np.array(mm)))*moremin:
#                     hipart[i].append(trace[i][j+1])
#                 else:
#                     if (np.mean(np.array(mm)))*lessmax < trace[i][j+3] < (np.mean(np.array(mm)))*moremin:
#                         hipart[i].append(trace[i][j+1])
#                     else:
#                         # Finished the high part
#                         H = False
# 
#         if L:
#             ml.append(trace[i][-j])
# 
#             if (np.mean(np.array(ml)))*lessmax < trace[i][-j-1] < (np.mean(np.array(ml)))*moremin:
#                 lowpart[i].append(trace[i][-j-1])
#                 
#             else:
#                 if (np.mean(np.array(ml)))*lessmax < trace[i][-j-2] < (np.mean(np.array(ml)))*moremin:
#                     lowpart[i].append(trace[i][-j-2])
#                 else:
#                     if (np.mean(np.array(ml)))*lessmax < trace[i][-j-3] < (np.mean(np.array(ml)))*moremin:
#                         lowpart[i].append(trace[i][-j-3])
#                     else:
#                         L = False
# 
# 
#     fixedhighpartaux = np.array(hipart[i])
#     fixedhighpart = np.concatenate((fixedhighpartaux, np.nan*np.zeros(Nframes-len(fixedhighpartaux))),axis=0)
# 
#     fixedlowpartaux = np.array(np.flip(lowpart[i]))
#     fixedlowpart = np.concatenate((np.nan*np.zeros(Nframes-len(fixedlowpartaux)), fixedlowpartaux,),axis=0)
# 
#     realmaxs.append(np.mean(hipart[i]))
#     realmins.append(np.mean(lowpart[i]))
#     
#     if PLOT and i in graph:
#         plt.plot(fixedhighpart, '--r')
#         plt.plot(fixedlowpart, '--g')    
# 
# 
# finaldata = (np.array(realmaxs)-np.array(realmins))
#     
#     
# plt.figure("histogram")
# for i in [20, 16, 12, 8, 4]:
#     print(i)
#     plt.hist(finaldata, int(len(finaldata)/i))
# plt.title((len(finaldata)))
# plt.grid()
# print(len(finaldata))
# 
# 
# 
# mu = np.mean(np.array(finaldata))
# sigma = np.sqrt(((len(finaldata)-1)**(-1))*np.sum((finaldata-mu)**2))
# #se = sigma/np.sqrt(Nframes)
# 
# print(mu,"+-", sigma)
# 
# plt.figure("histo/16")
# plt.title((len(finaldata)))
# plt.hist(finaldata, int(len(finaldata)/16), color='m')
# plt.axvline(mu, linestyle=':', color='k')
# plt.axvline(mu+sigma, linestyle='-.', color='r')
# plt.axvline(mu-sigma, linestyle='-.', color='r')
# plt.axvline(mu+2*sigma, linestyle='--', color='orange')
# plt.axvline(mu-2*sigma, linestyle='--', color='orange')
# 
# 
# fixeddata = np.copy(finaldata)
# fixeddata[fixeddata>mu + (1.6*sigma)] = np.nan 
# fixeddata[fixeddata<mu - (1.6*sigma)] = np.nan 
# mu = np.nanmean(np.array(fixeddata))
# sigma = np.sqrt(((np.count_nonzero(~np.isnan(fixeddata))-1)**(-1))*np.nansum((fixeddata-mu)**2))
# se = sigma/np.sqrt(Nframes)
# 
# print("valor final = ", mu,"+-", sigma)
# 
# plt.figure("histo centrado")
# plt.hist(finaldata, int(len(finaldata)/16), color='b')
# plt.title((np.count_nonzero(~np.isnan(finaldata))))
# plt.axvline(mu, linestyle=':', color='k')
# plt.axvline(mu+sigma, linestyle='-.', color='r')
# plt.axvline(mu-sigma, linestyle='-.', color='r')
# plt.axvline(mu+2*sigma, linestyle='--', color='orange')
# plt.axvline(mu-2*sigma, linestyle='--', color='orange')
#     
# #    return finaldata
# =============================================================================
