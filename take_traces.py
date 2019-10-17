# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:24:32 2019

@author: chiarelg
"""


def take_traces(FILE):
    """ input: .tif file with spots
    Find all the interesting peaks in the second frame
    and all the traces in these points"""

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    import matplotlib.pyplot as plt
    from skimage.feature import peak_local_max
    from skimage import  io #,img_as_float, data
    
    #im = img_as_float(data.coins())
#    FILE = 'C:/Origami testing Widefield/2019-10-11/Morgane/1Atto542_1640um_50mW_1/1Atto542_1640um_50mW_1_MMStack_Pos0.ome.tif'
    
    tiff = io.imread(FILE)
    
    im = tiff[1]
    
    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant')
    
    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=5)
    
    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')
    
    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')
    
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
    
    #print(maxvalues)
    
    nomax = np.where(np.array(maxvalues) < np.mean(maxvalues))[0]
    
    print( "\n \n", len(maxvalues), len(nomax))
    
    plt.figure("que onda")
    plt.title(len(maxvalues))
    plt.imshow(im, cmap=plt.cm.gray)
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    
    for j in range(len(coordinates[:,1])):
        if j in nomax:
    #        print("no")
            2+2
        else:
            plt.plot(coordinates[j, 1], coordinates[j, 0], 'b.')
    
    trace = {}
    p = 0
    for i in range(len(coordinates[:,0])):
        if i not in nomax:
            trace[p] = []
    
            for f in range(tiff.shape[0]):
                trace[p].append(tiff[f][coordinates[i,0],coordinates[i,1]])
            p += 1
    
    print(len(trace))
    print(len(maxvalues)-len(nomax))
    
    
    #for j in range(5):
    #    plt.plot(trace[j])
      
    
    Nframes = len(trace[0])
    Ntraces = len(trace)
    
    
    realmaxs = []
    realmins = []
    
    hipart = {}
    lowpart = {}
    
    lessmax = 0.95
    moremin = 1.05
    
    N = Ntraces
    
    PLOT = False
    graph = np.linspace(1, Ntraces-5, num=10, endpoint=False, dtype=int)
    for i in range(N):
    #    print(i)
        if PLOT and i in graph:
            plt.figure(i)
            plt.plot(trace[i], '*b')
    
        H = True
        L = True
        mm = []
        ml = []
        hipart[i] = [trace[i][0]]
        hipart[i].append(trace[i][1])
        
        lowpart[i] = [trace[i][-1]]
        for j in range(1, Nframes-3):
    
            if H:
                mm.append(trace[i][j])
    
                if (np.mean(np.array(mm)))*lessmax < trace[i][j+1] < (np.mean(np.array(mm)))*moremin:
                    hipart[i].append(trace[i][j+1])
    
                else:
                    if (np.mean(np.array(mm)))*lessmax < trace[i][j+2] < (np.mean(np.array(mm)))*moremin:
                        hipart[i].append(trace[i][j+1])
                    else:
                        if (np.mean(np.array(mm)))*lessmax < trace[i][j+3] < (np.mean(np.array(mm)))*moremin:
                            hipart[i].append(trace[i][j+1])
                        else:
                            # Finished the high part
                            H = False
    
            if L:
                ml.append(trace[i][-j])
    
                if (np.mean(np.array(ml)))*lessmax < trace[i][-j-1] < (np.mean(np.array(ml)))*moremin:
                    lowpart[i].append(trace[i][-j-1])
                    
                else:
                    if (np.mean(np.array(ml)))*lessmax < trace[i][-j-2] < (np.mean(np.array(ml)))*moremin:
                        lowpart[i].append(trace[i][-j-2])
                    else:
                        if (np.mean(np.array(ml)))*lessmax < trace[i][-j-3] < (np.mean(np.array(ml)))*moremin:
                            lowpart[i].append(trace[i][-j-3])
                        else:
                            L = False
    
    
        fixedhighpartaux = np.array(hipart[i])
        fixedhighpart = np.concatenate((fixedhighpartaux, np.nan*np.zeros(Nframes-len(fixedhighpartaux))),axis=0)
    
        fixedlowpartaux = np.array(np.flip(lowpart[i]))
        fixedlowpart = np.concatenate((np.nan*np.zeros(Nframes-len(fixedlowpartaux)), fixedlowpartaux,),axis=0)
    
        realmaxs.append(np.mean(hipart[i]))
        realmins.append(np.mean(lowpart[i]))
        
        if PLOT and i in graph:
            plt.plot(fixedhighpart, '--r')
            plt.plot(fixedlowpart, '--g')    
    
    
    finaldata = (np.array(realmaxs)-np.array(realmins))
    
    
    plt.figure("histogram")
    for i in [20, 16, 12, 8, 4]:
        print(i)
        plt.hist(finaldata, int(len(finaldata)/i))
    plt.title((len(finaldata)))
    plt.grid()
    print(len(finaldata))
#    
#    plt.figure("histo/16")
#    plt.hist(finaldata, int(len(finaldata)/12))
    
#    mu = np.mean(np.array(finaldata))
#    sigma = np.sqrt(((len(finaldata)-1)**(-1))*np.sum((finaldata-mu)**2))
#    #se = sigma/np.sqrt(Nframes)
#    
#    print(mu,"+-", sigma)
    return finaldata