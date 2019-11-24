# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:24:32 2019

@author: chiarelg
"""
# %% Now I will try with a Gaussian fit and some wird arguments.


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import  io #,img_as_float, data
from scipy.ndimage.measurements import center_of_mass, label
# %% Functions to make the Gauss fit 1
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
#FILE = 'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif'
#FILE = 'C:/Origami testing Widefield/2019-10-16/Green_Reference-11mW(5,4)_532nm-100ms_1/Green_Reference-11mW(5,4)_532nm-100ms_1_MMStack_Pos0.ome.tif'
FILE = 'C:/Origami testing Widefield/2019-10-16/Green_Reference-11mW(5,4)_532nm-100ms_2/IMG_20191107_143820.jpg'
plot = True
tiff = io.imread(FILE)

Nmeanstart = 50
Nmeanfinish = 70
im = np.mean(tiff, axis=2)  #np.sum(tiff[Nmeanstart:Nmeanfinish], axis=0)
print("shapes tiff, im", tiff.shape, im.shape)
# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
#image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates2 = peak_local_max(im, min_distance=50)

is_peak = peak_local_max(im, min_distance=50, indices=False) # outputs bool image
s = ndi.generate_binary_structure(2,50)
labels = label(is_peak,structure=s)[0]
merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
coordinates = np.array(merged_peaks, dtype = int)

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
# %%

# Generate test data with two peaks, one of which consists of two pixels of equal value
image = np.zeros((15,15),dtype=np.uint8)
image[5,3] = 128
image[5,2] = 255
image[5,7:9] = 255
image[3,8] = 255
image[6,8] = 128
image[7,8] = 255
image[9,8] = 255
image[9,3] = 255
image[4,9] = 255
d=2
# Finding peaks normally; results in three peaks
peaks = peak_local_max(image,min_distance=d)

# Find peaks and merge equal regions; results in two peaks
is_peak = peak_local_max(image,min_distance=d, indices=False) # outputs bool image
s = ndi.generate_binary_structure(2,2)
labels = label(is_peak,structure=s)[0]
merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
merged_peaks = np.array(merged_peaks, dtype=int)

# Visualize the results
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.imshow(image.T,cmap='gray')
ax1.plot(peaks[:,0],peaks[:,1],'ro')

ax2.imshow(image.T,cmap='gray')
ax2.plot(merged_peaks[:,0],merged_peaks[:,1],'ro')




#%%  takin the spots
roisize = 6  # es el doble de esto
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

nocenterx = 0
nocentery = 0
p=0
for i in range(len(newcoordinateX)):
#    roi2[i] = np.sum(im[newcoordinateX[i]:newcoordinateX[i]+roisize+1, newcoordinateY[i]:newcoordinateY[i]+roisize+1])
#    roi2[i] += np.sum(im[newcoordinateX[i]-roisize:newcoordinateX[i], newcoordinateY[i]-roisize:newcoordinateY[i]])
#    roi2[i] += np.sum(im[newcoordinateX[i]:newcoordinateX[i]+roisize+1, newcoordinateY[i]-roisize:newcoordinateY[i]])
#    roi2[i] += np.sum(im[newcoordinateX[i]-roisize:newcoordinateX[i], newcoordinateY[i]:newcoordinateY[i]+roisize+1])
    try:
        smallimup = np.concatenate((im[newcoordinateX[i]+nocenterx-roisize:newcoordinateX[i]+nocenterx,
                                       newcoordinateY[i]-roisize+nocentery:newcoordinateY[i]+nocentery],
                                  im[newcoordinateX[i]+nocenterx-roisize:newcoordinateX[i]+nocenterx,
                                     newcoordinateY[i]+nocentery:newcoordinateY[i]+roisize+nocentery]),
                                  axis=1)
        smallimdown = np.concatenate((im[newcoordinateX[i]+nocenterx:newcoordinateX[i]+roisize+nocenterx,
                                         newcoordinateY[i]-roisize+nocentery:newcoordinateY[i]+nocentery],
                                      im[newcoordinateX[i]+nocenterx:newcoordinateX[i]+roisize+nocenterx,
                                         newcoordinateY[i]+nocentery:newcoordinateY[i]+roisize+nocentery]),
                                      axis=1)
#        print(smallimup.shape, smallimdown.shape)
        smallim = np.concatenate((smallimup,smallimdown),axis=0)
        roi2[p] = smallim
        p+=1
    except:
#        pass
        print("noup")

##print(roi[0],np.sum(roi2[0]))
#plt.imshow(roi2[0])
#n=2
#roinew = roi2[n]
#for i in range(len(roi2[n][:,0])):
#    for j in range(len(roi2[n][0,:])):
#        if roi2[n][i,j] < 4000:
#            roinew[i,j] = 0
#
#
#plt.imshow(roinew)



# =============================================================================
# # %% I give up.....
# from Example2_Hoshen_Kopelman import percolation
# m=9
# 
# plt.imshow(roi2[m])
# 
# plt.figure()
# threshold = np.mean(roi2[m])
# L = percolation(roi2[m], threshold);
# print(L)
# plt.imshow(L)
# plt.colorbar()
# plt.show()
# 
# # %% circularity last atempt
# 
# from skimage import data
# from skimage import measure
# 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # Construct some test data
# x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# #r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
# r2 = roi2[n]
# 
# # Find contours at a constant value of 0.8
# contours = measure.find_contours(r2, 0.8)
# 
# # Display the image and plot all contours found
# plt.imshow(r2, interpolation='nearest')
# 
# for n, contour in enumerate(contours):
#     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
# 
# plt.axis('image')
# plt.xticks([])
# plt.yticks([])
# plt.show()
# 
# # %% Circularity Part 3    ..... noup
# 
# import cv2 
# import numpy as np 
#   
# # Load image 
# image = cv2.imread('C:/Users/chiarelG/Downloads/Box3.jpg',0)
# image.dtype
# 
# image = np.array(roi2[1])
# plt.imshow(image)
# plt.imshow(roi2[1])
# 
#  
#   
# # Set our filtering parameters 
# # Initialize parameter settiing using cv2.SimpleBlobDetector 
# params = cv2.SimpleBlobDetector_Params() 
#   
# # Set Area filtering parameters 
# params.filterByArea = True
# params.minArea = 100
#   
# # Set Circularity filtering parameters 
# params.filterByCircularity = True 
# params.minCircularity = 0.9
#   
# # Set Convexity filtering parameters 
# params.filterByConvexity = True
# params.minConvexity = 0.2
#       
# # Set inertia filtering parameters 
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
#   
# # Create a detector with the parameters 
# detector = cv2.SimpleBlobDetector_create(params) 
#       
# # Detect blobs 
# keypoints = detector.detect(image)
#   
# # Draw blobs on our image as red circles 
# blank = np.zeros((1, 1))  
# blobs = cv2.drawKeypoints(image, keypoints, blank, (200, 200, 200), 
#                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
#   
# number_of_blobs = len(keypoints) 
# text = "Number of Circular Blobs: " + str(len(keypoints)) 
# cv2.putText(blobs, text, (20, 550), 
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
#   
# # Show blobs 
# cv2.imshow("Filtering Circular Blobs Only", blobs) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
# 
# # %% circularity attempt number 2 fail again. Need 1 more dimension. 
# 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# from skimage.draw import ellipse
# from skimage.measure import find_contours, approximate_polygon, \
#     subdivide_polygon
# 
# hand = contours[0]
# hand2 = np.array([[1.64516129, 1.16145833],  
#                  [1.64516129, 1.59375],
#                  [1.35080645, 1.921875],
#                  [1.375, 2.18229167],
#                  [1.68548387, 1.9375],
#                  [1.60887097, 2.55208333],
#                  [1.68548387, 2.69791667],
#                  [1.76209677, 2.56770833],
#                  [1.83064516, 1.97395833],
#                  [1.89516129, 2.75],
#                  [1.9516129, 2.84895833],
#                  [2.01209677, 2.76041667],
#                  [1.99193548, 1.99479167],
#                  [2.11290323, 2.63020833],
#                  [2.2016129, 2.734375],
#                  [2.25403226, 2.60416667],
#                  [2.14919355, 1.953125],
#                  [2.30645161, 2.36979167],
#                  [2.39112903, 2.36979167],
#                  [2.41532258, 2.1875],
#                  [2.1733871, 1.703125],
#                  [2.07782258, 1.16666667]])
# 
# # subdivide polygon using 2nd degree B-Splines
# new_hand = hand.copy()
# for _ in range(5):
#     new_hand = subdivide_polygon(new_hand, degree=2, preserve_ends=True)
# 
# # approximate subdivided polygon with Douglas-Peucker algorithm
# appr_hand = approximate_polygon(new_hand, tolerance=0.02)
# 
# print("Number of coordinates:", len(hand), len(new_hand), len(appr_hand))
# 
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
# 
# ax1.plot(hand[:, 0], hand[:, 1])
# ax1.plot(new_hand[:, 0], new_hand[:, 1])
# ax1.plot(appr_hand[:, 0], appr_hand[:, 1])
# 
# 
# # create two ellipses in image
# img = np.zeros((800, 800), 'int32')
# rr, cc = ellipse(250, 250, 180, 230, img.shape)
# img[rr, cc] = 1
# rr, cc = ellipse(600, 600, 150, 90, img.shape)
# img[rr, cc] = 1
# 
# plt.gray()
# ax2.imshow(img)
# 
# # approximate / simplify coordinates of the two ellipses
# for contour in find_contours(img, 0):
#     coords = approximate_polygon(contour, tolerance=2.5)
#     ax2.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
#     coords2 = approximate_polygon(contour, tolerance=39.5)
#     ax2.plot(coords2[:, 1], coords2[:, 0], '-g', linewidth=2)
#     print("Number of coordinates:", len(contour), len(coords), len(coords2))
# 
# ax2.axis((0, 800, 0, 800))
# 
# plt.show()
# 
# # %% Circularity attempt Fail
# 
# import numpy as np
# import math
# from skimage.measure import regionprops
# 
# circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)
# 
# ex1 = np.zeros((100, 100), dtype=np.uint)
# for i in range(100):
#         ex1[i, i] = 1
# for i in range(100):
#         ex1[-i, i] = 1
# [circ(region) for region in regionprops(ex1)]
# 
# 
# 
# 
# ex2 = np.ones((100, 100), dtype=np.uint)
# [circ(region) for region in regionprops(ex2)]
# 
# ex3 = np.zeros((10, 10), dtype=np.uint)
# for i in range(2, 4):
#     ex3[i, i] = 1
# 
# for region in regionprops(ex3):
#     print('Perimeter: {}\nCircularity: {}'
#           .format(region.perimeter, circ(region)))
# plt.imshow(ex3)
# #plt.imshow(ex1[:10,:10])
# 
# #plt.imshow()
# =============================================================================

# %% GAUSSIAN FIT 1
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
            try:
                ax.text(X[xc+i, yc+j], Y[xc+i, yc+j], "Ga", color='m')

                xsum = X[xc+i, yc+j] + xsum
                ysum = Y[xc+i, yc+j] + ysum
            except:
                pass

    xmean = xsum / (resol**2)
    ymean = ysum / (resol**2)
    ax.text(xmean, ymean, "✔", color='r')
#    ax.text(x, y, "◙", color='m')
    #            Normal = self.scanRange / self.numberofPixels  # Normalizo
    #            ax.set_title((self.xcm*Normal + float(initPos[0]),
    #                          self.ycm*Normal + float(initPos[1])))
    #plt.text(0.95, 0.05, """x : %.2f y : %.2f """
    #         % (xmean, ymean),  # X[xc, yc], Y[xc, yc]
    #         fontsize=16, horizontalalignment='right',
    #         verticalalignment='bottom', transform=ax.transAxes)
    #print("x", xv[int(x)], X[xc, yc], xmean)
    #            Normal = self.scanRange / self.numberofPixels  # Normalizo
    ax.set_title("Centro en x={:.3f}, y={:.3f}".format(x,y))
    plt.show()

print(all_params[0,:])  # (height, x, y, width_x, width_y)
plt.figure("Search for a good fit")
c = ['k', 'b', 'r', 'c', 'm']
labeled = ['height', 'x', 'y', 'width_x', 'width_y']
for l in range(1,len(all_params[0,:])):
    plt.plot(all_params[:,l], '*-', color=c[l], label=labeled[l])
plt.grid()
plt.legend()
plt.show()



#%% Another way to 2d gauss fit. Do not like it. 

fit_params = np.zeros((N,5))
fit_Rsquared = np.zeros((N))
fit_errors = np.zeros((N,5))
l=3
for l in range(1,N):
    print("l",l)
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
     
    q = 5
    w = 1/5
    boundariesin = (amp*w, xc*w, yc*w, sigma_x*w, sigma_y*w)
    boundariesout = (amp*q, xc*q, yc*q, sigma_x*q, sigma_y*q) 
    boundaries = (boundariesin, boundariesout)
    # perform the fit, making sure to flatten the noisy data for the fit routine 
    fit_params[l], cov_mat = curve_fit(gaussian_2d, xy_mesh, np.ravel(np.transpose(roi2[l])), bounds=boundaries)# p0=guess_vals)
    
    # calculate fit parameter errors from covariance matrix
    fit_errors = np.sqrt(np.diag(cov_mat)) 
     
    # manually calculate R-squared goodness of fit
    fit_residual = roi2[l] - gaussian_2d(xy_mesh, *fit_params[l]).reshape(np.outer(xv,yv).shape)
    fit_Rsquared[l] = 1 - np.var(fit_residual)/np.var(roi2[l])
    
#    print('Fit R-squared:', fit_Rsquared[l], '\n')
#    print('Fit Amplitude:', fit_params[l][0], '\u00b1', fit_errors[0])
#    print('Fit X-Center: ', fit_params[l][1], '\u00b1', fit_errors[1])
#    print('Fit Y-Center: ', fit_params[l][2], '\u00b1', fit_errors[2])
#    print('Fit X-Sigma:  ', fit_params[l][3], '\u00b1', fit_errors[3])
#    print('Fit Y-Sigma:  ', fit_params[l][4], '\u00b1', fit_errors[4])
    print("new fit", fit_params[l][1:5])
    print("old xy", all_params[l,1:5])
    
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
            try:
        #        ax.text(X[xc+i, yc+j], Y[xc+i, yc+j], "Ga", color='m')
                xsum = X[xc+i, yc+j] + xsum
                ysum = Y[xc+i, yc+j] + ysum
            except:
                pass
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

#%% function to create a 2d gaussian
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

#%% maybe usuful. convolve2d
from scipy.signal import convolve2d

im2 = convolve2d(im,np.ones((3,3),dtype=int),'same')

plt.imshow(im2, cmap=plt.cm.gray)



# %% My old program to find spots and extract the trace from them.

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
