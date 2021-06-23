# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:15:38 2021

@author: chiarelg
"""

### .fits reading


import numpy as np

# Set up matplotlib and use a nicer set of plot parameters
#%config InlineBackend.rc = {}
import matplotlib
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt

from astropy.io import fits

image_file = 'C:/Users/chiarelG/Downloads/fits for nicole/Si_Bare_acq2s_norm_scalebar.fits'


hdu_list = fits.open(image_file)
hdu_list.info()

image_data = hdu_list[0].data



print(type(image_data))
print(image_data.shape)



hdu_list.close()

#%%


image_data.shape

import matplotlib.pyplot as plt

from astropy.io import fits

image_file = 'C:/Users/chiarelG/Downloads/fits for nicole/Si_Bare_acq2s_norm_scalebar.fits'
image_data = fits.getdata(image_file)
print(type(image_data))
print(image_data.shape)
plt.imshow(np.sum(image_data[:,:,:],axis=0), cmap='gray')
plt.colorbar()

for i in range(100):
    plt.plot(image_data[:,10*i,1000])

# To see more color maps
# http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
print('Min:', np.min(image_data))
print('Max:', np.max(image_data))
print('Mean:', np.mean(image_data))
print('Stdev:', np.std(image_data))





print(type(image_data.flatten()))



NBINS = 1000
histogram = plt.hist(image_data.flatten(), NBINS)


from matplotlib.colors import LogNorm



plt.imshow(image_data, cmap='gray', norm=LogNorm())

# I chose the tick marks based on the histogram above
cbar = plt.colorbar(ticks=[5.e3,1.e4,2.e4])
cbar.ax.set_yticklabels(['5,000','10,000','20,000'])



#%%
from skimage import io
data = io.imread(image_file)
