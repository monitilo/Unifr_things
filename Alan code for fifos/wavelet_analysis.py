import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from tqdm import tqdm
import sys
import xmltodict
from datetime import date
# plt.ion()


def load_image(directory):
    '''Open a image file from a directory. This will search for a .hdr file in the given directory and then read as many lines as specified in the header file.
    Currently assumes no ALEX excitation and returns the first two channels'''
    files = listdir(directory)
    headerfile = [x for x in files if ".hdr" in x]
    if len(headerfile) > 1:
        raise ValueError("More than one header file in directory.")
    else:
        headerfile = headerfile[0]
    lines = [x for x in files if ".fifo" in x]
    with open(directory + "/" + headerfile) as fd:
        doc = xmltodict.parse(fd.read())
        x_res, y_res, z_res = float(doc['LVData']['Cluster']['Cluster']['Cluster'][2]['DBL'][0]['Val']), float(
            doc['LVData']['Cluster']['Cluster']['Cluster'][2]['DBL'][1]['Val']), float(
            doc['LVData']['Cluster']['Cluster']['Cluster'][2]['DBL'][2]['Val'])  # in nm/pixel
        time_res = float(doc['LVData']['Cluster']['Cluster']['DBL']['Val'])  # in ms/pixel
        x_size, y_size = float(doc['LVData']['Cluster']['Cluster']['Cluster'][3]['DBL'][0]['Val']), float(
            doc['LVData']['Cluster']['Cluster']['Cluster'][3]['DBL'][1]['Val'])  # in um
        num_channels = float(doc['LVData']['Cluster']['Array']['Cluster']['Cluster'][0]['U16'][3]['Val'])
        tac_range = float(doc['LVData']['Cluster']['Array']['Cluster']['Cluster'][3]['SGL'][0]['Val'])  # in ns
        pixel_per_line = round((x_size * 1e-6) / (x_res * 1e-9))
        lines_per_image = round((y_size * 1e-6) / (y_res * 1e-9))
        time_per_line = float(((x_size * 1e-6) / (x_res * 1e-9)) * time_res * 1e-3)
        bins = np.linspace(0, time_per_line, pixel_per_line+1)
        repetition_rate = 40e6
    meta={}
    meta['misc'] = doc
    meta['lines_per_image'] = lines_per_image
    meta['time_per_line'] = time_per_line
    meta['pixel_per_line'] = pixel_per_line
    print("TAC", tac_range)
    meta['tac_range'] = tac_range
    meta['repition_rate'] = repetition_rate  # HARDCODED

    if len(lines) < lines_per_image:
        raise Exception("Warning!! The image you are loading is missing %d line(s)!\n This program currently only works with quadratic images."% int(lines_per_image-len(lines)))
    ch1_image = np.zeros((pixel_per_line, lines_per_image))
    ch2_image = np.zeros((pixel_per_line, lines_per_image))
    results_dict = {}
    ch1_micro = {}
    ch2_micro = {}
    ch1_macro = {}
    ch2_macro = {}

    for i in tqdm(range(0, lines_per_image)):
        s = parse_file(directory+lines[i])
        dtime_ch1, micro_ch1 = process_entry(s, 0)
        dtime_ch2, micro_ch2 = process_entry(s, 1)
        ch1_image[i] = np.histogram(dtime_ch1, bins)[0]
        ch2_image[i] = np.histogram(dtime_ch2, bins)[0]
        ch1_micro[i] = micro_ch1
        ch2_micro[i] = micro_ch2
        ch1_macro[i] = dtime_ch1
        ch2_macro[i] = dtime_ch2
    unique, counts = np.unique(ch1_image, return_counts=True, axis=0)
    if np.max(counts) > 1:
        print("Warning!! There might be a duplicate in line", np.where(counts > 1))
    results_dict['ch1_image'] = ch1_image
    results_dict['ch2_image'] = ch2_image
    results_dict['ch1_micro'] = ch1_micro
    results_dict['ch2_micro'] = ch2_micro
    results_dict['ch1_macro'] = ch1_macro
    results_dict['ch2_macro'] = ch2_macro
    results_dict['meta'] = meta
    return results_dict


def parse_file(filename):
        '''Read in a single line file from filename, using the data standard from KK confocal. Returns an array with special bit, channel, nsync, dtime'''
        dt = np.dtype(np.uint8)  # 32 bits are 8 bytes
        dt = dt.newbyteorder('>')  # change the byteorder
        with open(filename,"rb") as s:  # read the file to a numpy array
            buffer = np.frombuffer(s.read(), dtype=dt)
        bits = np.unpackbits(buffer)  # unpack every bit
        bits = bits.reshape(int(len(bits)/32), 32)  # reshape so that every 32 bits there is a new row
        arr = np.zeros((int(len(bits)), 4))  # initiate result array
        arr[:, 0] = bits[:, 0] #special
        arr[:, 1] = bits[:, 1:7].dot(1 << np.arange(bits[:, 1:7].shape[-1] - 1, -1, -1)) #channel
        arr[:, 2] = bits[:, 7:22].dot(1 << np.arange(bits[:, 7:22].shape[-1] - 1, -1, -1))  # nsync
        arr[:, 3] = bits[:, 22:32].dot(1 << np.arange(bits[:, 22:32].shape[-1] - 1, -1, -1))  # dtime
        return arr

def rgb_representation(data_dict,gauss_smoothing=False):
            if gauss_smoothing:
                from skimage.filters import gaussian
                ch1_img,ch2_img = gaussian(data_dict["ch1_image"]), gaussian(data_dict['ch2_image'])
            else:
                ch1_img,ch2_img = data_dict['ch1_image'], data_dict['ch2_image']
            ch1_img[ch1_img>20] = 20
            ch1_img[ch1_img<2] = 0
            ch1_img = ch1_img/20.0
            ch2_img[ch2_img>20] = 20
            ch2_img[ch2_img<2] = 0
            ch2_img = ch2_img/20.0
            rgb = np.dstack((ch2_img,ch1_img,np.zeros((len(ch1_img),len(ch1_img)))))
            return rgb


def process_entry(arr, channel):
    '''Overflow correction, takes the array produced by parse_file and a channel ID and returns overflow corrected dtime and microtime'''
    ovf=np.zeros(len(arr[:, 0]))
    ovf_index = np.where((arr[:, 0] == True)&(arr[:, 1] == 63))  # this is an overflow event
    ovf[ovf_index] = 1024
    ovf_mult = np.cumsum(ovf*12.5e-9)  # time in seconds
    arr[:, 3] = (arr[:, 3]*12.5e-9)+ovf_mult  # multiply the nsync with the overflow
    channel_index = np.where((arr[:, 0] == 0)&(arr[:, 1] == channel))  # find the photons corresponding to the channel
    dtime_channel = arr[channel_index, 3]
    micro_channel = arr[channel_index, 2]*1e-12   # should still find out microtime binning (assuming 1 picosecs)

    return dtime_channel.flatten(),micro_channel.flatten()


def fret_hist(results_dict):
    '''Spotfinding algorithm. Looks for spots in both channels. Attempts to do a segmentation based on the random_walk algorithm by scikit-image.
    Takes in ch1, ch2 as produced by load_image and returns a numpy array with all spotwise proximity ratios GR/(GR+GG) (intensities are summed up per spot) as well as an image overlay with the data and the spots that were detected. '''
    fret_h = []
    green_ints =[]
    red_ints=[]
    ch1_image = results_dict['ch1_image']
    ch2_image = results_dict['ch2_image']
    from skimage.color import label2rgb
    from scipy import ndimage
    spots = wavelet_spotfinder(ch1_image+ch2_image)
    max_spotindex=np.max(spots)
    for i in range(1, max_spotindex):  # iterate over all spots and get summed intensities
        if np.sum(ch1_image[spots == i]) != 0:  # only use spots that are there
            gg_sum = np.sum(ch1_image[spots == i])
            gr_sum = np.sum(ch2_image[spots == i])
            if gg_sum + gr_sum > 10.0:  # use only spots with at least 10 photons
                fret_h.append(gr_sum/(gr_sum+gg_sum))
                green_ints.append(np.sum(ch1_image[spots == i])/len(ch1_image[spots==i]))
                red_ints.append(np.sum(ch2_image[spots == i])/len(ch1_image[spots==i]))
        else:
            pass
    fret_h = np.array(fret_h)
    output_image = np.zeros((len(ch1_image),len(ch1_image),3)) #build rgb image for label2rgb
    output_image[:, :, 1] = ch1_image/np.max(ch1_image)
    output_image[:, :, 0] = ch2_image/np.max(ch2_image)
    overlay = label2rgb(spots, image=output_image, bg_label=0,alpha=0.4)
    ints = np.array([green_ints,red_ints])
    return fret_h,overlay,ints

def remove_weird_spots(ar, min_size=64, max_size=65, connectivity=1, in_place=False): #This doesnt work for some reason
    '''Remove spots from a spots array that are smaller or bigger than the given integers in size, taken from remove_small_objects'''
    from skimage.measure import label
    out = ar
    ccs = out
    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes < min_size
    too_big = component_sizes > max_size
    too_small_mask = too_small[ccs]
    too_big_mask = too_big[ccs]
    out[too_small_mask] = 0
    out[too_big_mask] = 0
    return out


def calculate_wavelet(input,max_level,k):
    '''calculate a trous wavelet based on olivo martin 2002'''
    from scipy.ndimage import convolve1d
    from skimage.filters import gaussian,threshold_otsu
    #input = gaussian(input)
    import numpy as np
    data_size = len(input)
    out = np.zeros((data_size,data_size,max_level))
    out[:,:,0] = input
    for i in range(1,max_level):
        kernel = np.concatenate(([1/16],np.zeros(2**(i-1)-1),[1/4],np.zeros(2**(i-1)-1),[3/8],np.zeros(2**(i-1)-1),[1/4],np.zeros(2**(i-1)-1),[1/16]))
        wavelet = convolve1d(input,kernel,mode="reflect",axis=0)
        wavelet = convolve1d(wavelet,kernel,mode="reflect",axis=1)
        wavelet = wavelet-out[:,:,i-1]
        abs = np.abs(wavelet)
        wavelet[abs<k*np.median(wavelet)] = 0
        out[:,:,i] = wavelet
    final_wavelet = np.abs(np.prod(out[:,:,1:],axis=2))
    final_wavelet = np.ma.log10(final_wavelet).filled(0)
    final_wavelet[final_wavelet<threshold_otsu(final_wavelet[final_wavelet!=0])]=0
    return final_wavelet


def wavelet_spotfinder(input_array):
    import pywt
    import numpy as np
    from skimage.filters import threshold_otsu
    from skimage.feature import peak_local_max
    from skimage.segmentation import random_walker
    from skimage.measure import label, regionprops
    wavelets = calculate_wavelet(input_array,max_level=3,k=2)  #k level = 2 max level 3 seem to be good
    maxima = peak_local_max(wavelets, min_distance=3)  # find local maxima (spot centers) original: distnace
    max_array = np.zeros((len(input_array), len(input_array)))  # initiate spot array
    max_array[maxima[:, 0], maxima[:, 1]] = 1  # array with local maxima, same size as image array
    max_array = label(max_array, background=0)
    spots = random_walker(wavelets > 0, max_array, mode="bf")
    spots = remove_weird_spots(spots, min_size=5, max_size=5000)
    props = regionprops(spots)
    med = np.median([i['area'] for i in props])
    spots = remove_weird_spots(spots, min_size=0.25*med, max_size=2*med)
    #spots = remove_weird_spots(spots, min_size=20, max_size=5*med)
    return spots


def fret_hist_lt(results_dict):
    fret_h = []
    green_ints = []
    red_ints = []
    ch1_image = results_dict['ch1_image']
    ch2_image = results_dict['ch2_image']
    spots = wavelet_spotfinder(ch1_image+ch2_image)
    lifetime_dict = get_spot_lifetimes(results_dict, spots)
    from skimage.color import label2rgb
    from scipy import ndimage
    max_spotindex=np.max(spots)
    for spot in spots:  # iterate over all spots and get summed intensities
        if np.sum(ch1_image[spots == i]) != 0:  # only use spots that are there
            fret_h.append(1 - (lifetime_dict['ch1'][i] / 3.5e-9))
            green_ints.append(np.sum(ch1_image[spots == i]))
            red_ints.append(np.sum(ch2_image[spots == i]))
        else:
            pass
    fret_h = np.array(fret_h)
    output_image = np.zeros((len(ch1_image),len(ch1_image),3)) #build rgb image for label2rgb
    output_image[:, :, 1] = ch1_image/np.max(ch1_image)
    output_image[:, :, 0] = ch2_image/np.max(ch2_image)
    overlay = label2rgb(spots, image=output_image, bg_label=0,alpha=0.4)
    mean_values = [np.mean(), np.mean()]
    return fret_h, overlay, mean_values

def get_spot_lifetimes(results_dict,spots):
    from scipy.stats import expon
    lifetime_dict = get_spot_micro(results_dict,spots)
    ch1_lts = lifetime_dict['ch1']
    ch2_lts = lifetime_dict['ch2']
    results = {'ch1':{}, 'ch2':{}}
    for i in range(0,len(ch1_lts)):
        ch1_h = np.histogram(ch1_lts[i],20)
        ch2_h = np.histogram(ch1_lts[i],20)
        peak_ch1 = ch1_h[1][np.argmax(ch1_h[0])]
        peak_ch2 = ch2_h[1][np.argmax(ch2_h[0])]
        ch1_cropped = ch1_lts[i][ch1_lts[i] > peak_ch1]
        ch2_cropped = ch2_lts[i][ch2_lts[i] > peak_ch2]
        loc_ch1, scale_ch1 = expon.fit(ch1_cropped)
        loc_ch2, scale_ch2 = expon.fit(ch2_cropped)
        results['ch1'][i] = scale_ch1
        results['ch2'][i] = scale_ch2
    return results


def get_spot_micro(results_dict, spots):
    from skimage.measure import regionprops
    props = regionprops(spots)
    bins = np.linspace(0, results_dict['meta']['time_per_line'], results_dict['meta']['pixel_per_line']+1)
    coords = [i['coords'] for i in props]
    ch1_lts = {}
    ch2_lts = {}
    for i in range(0,len(coords)):
        ch1_lts[i] = np.empty(0)
        ch2_lts[i] = np.empty(0)
        for spot_coord in coords[i]:
            ch1_macro = results_dict['ch1_macro'][spot_coord[0]]
            ch2_macro = results_dict['ch2_macro'][spot_coord[0]]
            ch1_micro = results_dict['ch1_micro'][spot_coord[0]]
            ch2_micro = results_dict['ch2_micro'][spot_coord[0]]
            ch1_res = ch1_micro[(ch1_macro > bins[spot_coord[1]]) & (ch1_macro < bins[spot_coord[1]+1])]
            ch2_res = ch2_micro[(ch2_macro > bins[spot_coord[1]]) & (ch2_macro < bins[spot_coord[1]+1])]
            ch1_lts[i] = np.concatenate((ch1_lts[i],ch1_res))
            ch2_lts[i] = np.concatenate((ch2_lts[i],ch2_res))
    lifetime_dict = {}
    lifetime_dict['ch1'] = ch1_lts
    lifetime_dict['ch2'] = ch2_lts
    return lifetime_dict

if __name__ == "__main__":
    # execute only if run as a script
    # it gets adventurous down here
    if len(sys.argv)!=2:
        print("Usage: wavelet_analysis.py Directory_with_scans")
        sys.exit()
    else:
        from skimage.filters import gaussian
        directoriesr = listdir(sys.argv[1])
        directories = [i for i in directoriesr if "." not in i]
        path = sys.argv[1]
        path = path.replace("\\","/")
        counter=1
        final_hist=np.array([])
        hist_bins=np.arange(0,1,0.02)
        print(".........Analysing ",len(directories)," scans")
        for scan in directories:
            data_dict = load_image(path+"/"+scan+"/")
            print(".........Read file ",counter)
            fh,overlay,ints=fret_hist(data_dict)
            final_hist = np.append(final_hist,fh)
            print(".........Analysed file ",counter)
            print("Total: %.0f spots \n Mean proximity ratio: %.3f \n %% open hinges: %.1f"%(len(fh),np.mean(fh),100-(len(fh[fh>0.3])/len(fh))*100))
            np.savetxt(path+"/%s_%s.txt"%(scan,date.today()),fh)
            print(".........Saved fret hist ",counter)
            f, (ax1, ax3, ax4) = plt.subplots(1, 3)
            f.set_size_inches(7,3)
            ch1_img,ch2_img = gaussian(data_dict["ch1_image"]), gaussian(data_dict['ch2_image'])
            ch1_img[ch1_img>20] = 20
            ch1_img[ch1_img<2] = 0
            ch1_img = ch1_img/20.0
            ch2_img[ch2_img>20] = 20
            ch2_img[ch2_img<2] = 0
            ch2_img = ch2_img/20.0
            rgb = np.dstack((ch2_img,ch1_img,np.zeros((len(ch1_img),len(ch1_img)))))
            ax1.imshow(rgb)
            ax3.imshow(overlay)
            ax1.set_axis_off()
            ax3.set_axis_off()
            ax4.hist(fh,hist_bins)
            ax4.set_title("Proximity ratio")
            ax1.set_title("Channel 1 mean int = %.1f/px, std = %.1f\nChannel 2 mean int = %.1f/px, std = %.1f"%(np.mean(ints[0]),np.std(ints[0]),np.mean(ints[1]),np.std(ints[1])),fontsize=6)
            ax3.set_title("Overlay with spot image")
            plt.tight_layout()
            plt.savefig(path+"/%s_%s.pdf"%(scan,date.today()))
            print(".........Saved img overlay ",counter)
            counter+=1
        np.savetxt(path+"/final_hist_%s.txt"%date.today(),final_hist)
        fraction_closed = len(final_hist[final_hist>0.3])/len(final_hist)
        fraction_closed = fraction_closed * 100
        plt.clf()
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.axvspan(0,0.3,alpha=0.2,color="green")
        ax.axvspan(0.3,1,alpha=0.2,color="red")
        ax.hist(final_hist,np.arange(0,1,0.005))
        ax.set_xlabel("Proximity ratio")
        ax.set_ylabel("Counts")
        ax.text(0.4, 0.9, "Closed hinges: %.2f%%"%fraction_closed, ha='left', va='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(path+"/final_hist_%s_%s.pdf"%(scan,date.today()))
        plt.clf()
        print(".........Saved final hist")
        print(".........ALL DONE!")
        sys.exit()
