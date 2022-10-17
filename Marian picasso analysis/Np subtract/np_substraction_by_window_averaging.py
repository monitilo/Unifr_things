# -*- coding: utf-8 -*-
"""
NPs subtraction by using fixed window average

Created on Wed Apr 28 14:46:29 2021
Fribourg, Switzerland

@author: Mariano Barella
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tifffile import imread, imwrite

plt.close('all')

def calculate_average_image(averaging_window, baseline, threshold, working_folder, filename):

    filepath = os.path.join(working_folder, filename)

    # load tiff file
    video16 = imread(filepath)
    # convert to 32-bit signed integer to avoid overflow while subtracting
    video32 = np.array(video16, dtype = np.int32)
    # find number of frames
    number_of_frames, rows, cols = video32.shape
    
    # watch-dog: check if remainder is integer
    if number_of_frames % averaging_window != 0:
        print('Error: the number of frames must be divisible by the selected averaging window.')
        print('Current number of frames: %d' % number_of_frames)
        print('Current averaging window: %d' % averaging_window)
        print('Ending program.')
        sys.exit()
    
    # define number of chuncks
    number_of_chunks = int(number_of_frames/averaging_window)
    
    # allocate new video and baseline frame
    print("AAAA", video32.shape[1:2], video32.shape[1], video32.shape[2])
    new_video32 = np.zeros(video32.shape, dtype = np.int32)
#    baseline_frame = np.ones(video32.shape[1:2], dtype = np.int32)*baseline
    baseline_frame = np.ones((video32.shape[1],video32.shape[2]), dtype = np.int32)*baseline
    
    
    avg_sequence32 = np.zeros([number_of_chunks, video32.shape[1], video32.shape[2]], dtype = np.int32)
    
    # average the chuncks
    for i in range(number_of_chunks):
        a = int(i*averaging_window)
        b = int((i + 1)*averaging_window)
        avg_sequence32[i,:,:] = np.mean(video32[a:b, :, :], dtype = np.int32, axis = 0)
    avg_sequence32 = np.where(avg_sequence32 < threshold, baseline, avg_sequence32)

    # substract NPs
    for i in range(number_of_chunks):    
        a = i*averaging_window
        b = (i + 1)*averaging_window
        for j in range(a,b):
            new_video32[j,:,:] = np.subtract(video32[j,:,:], avg_sequence32[i,:,:], dtype = np.int32)
    
    # mask or add basline to simulate camera counts
    new_video32 += baseline_frame
    new_video32 = np.where(new_video32 < baseline, baseline, new_video32)
    # mask extreme values
    new_video32 = np.where(new_video32 > 2**16, 2**16, new_video32)
    # convert to 16-bit
    new_video16 = np.array(new_video32, dtype = np.uint16)
    avg_sequence16 = np.array(avg_sequence32, dtype = np.uint16)

    # save to tiff file    
    new_filepath = filepath.replace('.tif', '_NPs_subtracted.tif')
    imwrite(new_filepath, new_video16)
    
    new_filepath = filepath.replace('.tif', '_averaged_sequence.tif')
    imwrite(new_filepath, avg_sequence16)

    print("Finished")
    return

if __name__ == "__main__":
    
    averaging_window = 100 # in number of frames
    baseline = 100 # in counts
    threshold = 0 # in counts

    from tkinter import Tk, filedialog
    
    root = Tk()
    nametoload = filedialog.askopenfilename(filetypes=(("", "*.tif"), ("", "*.")))
    root.withdraw()
    working_folder = os.path.dirname(nametoload)
    filename = os.path.basename(nametoload)
    

#    working_folder = 'C:/Origami testing Widefield/2022-04-20 Dimers 3 and 1 spot/well1_640nm_14mW_zone1_Before paint_1'
#    filename = 'Test marian code.tif'

    ### well 2 (12 mM MgCl2)
#    working_folder = 'C:/datos_mariano/posdoc/DNA-PAINT/data_fribourg/20210422_NPs_overnight_incubation/well2_532_3979uW_tirf2470_2'
#    filename = 'well2_532_3979uW_tirf2470_2_MMStack_Pos0.ome.tif'
    # working_folder = 'C:/datos_mariano/posdoc/DNA-PAINT/data_fribourg/20210422_NPs_overnight_incubation/well2_642_4709uW_tirf2470_1'
    # filename = 'well2_642_4709uW_tirf2470_1_MMStack_Pos0.ome.tif'
    
    ### well 4 (12 mM MgCl2 + 600 mM NaCl)
    # working_folder = 'C:/datos_mariano/posdoc/DNA-PAINT/data_fribourg/20210422_NPs_overnight_incubation/well4_532_3979uW_tirf2470_1'
    # filename = 'well4_532_3979uW_tirf2470_1_MMStack_Pos0.ome.tif'
    # working_folder = 'C:/datos_mariano/posdoc/DNA-PAINT/data_fribourg/20210422_NPs_overnight_incubation/well4_642_4709uW_tirf2470_2'
    # filename = 'well4_642_4709uW_tirf2470_2_MMStack_Pos0.ome.tif'
    
    calculate_average_image(averaging_window, baseline, threshold, working_folder, filename)