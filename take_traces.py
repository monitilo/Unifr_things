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
roisize = 1
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
        print("i",i,"p", p, "roi[i]", roi[i],deletear)
#        deletear.append(i)
    p += 1

for d in deletear:
    print("dddddeletearrrrr", d)
    del roi[d]


for i in range(len(newcoordinateX)):
    roi2[i] = np.sum(im[newcoordinateX[i]:newcoordinateX[i]+roisize+1, newcoordinateY[i]:newcoordinateY[i]+roisize+1])
    roi2[i] += np.sum(im[newcoordinateX[i]-roisize:newcoordinateX[i]+1, newcoordinateY[i]-roisize:newcoordinateY[i]+1])
    roi2[i] -= im[newcoordinateX[i],newcoordinateY[i]]

print(roi[0],roi2[0])

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
