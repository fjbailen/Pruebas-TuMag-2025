"""
This program reads a series of images from a directory, processes them, and 
computes the mean value for each exposure times
"""
import os, sys, getopt, time
sys.path.append('./functions')
sys.path.append('./imread')
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
import matplotlib as mpl #dos
from mpl_toolkits.axes_grid1 import make_axes_locatable
from SPGCam_lib import *
from tqdm import tqdm
import plot_func as pf
import utils as ut

#Exposure times
min_t=32 #minimum exposure time in ms
dt=1 #step exposure time in ms

tic=time.time()

#Dir name
dir_d = 'images'
dir_d ='./' + dir_d +'/'

#Read images
width = 2016
height = 2016
filename = '*.img'
maxNumOfFiles_per_serie = 0; # 0 = no limit.

"""
files = rf.list_files(dir_d,filename)
files.sort()

for i in range(len(files)):
    im_dummy = np.fromfile(dir_d+files[i],dtype='<i2')
    im_dummy = im_dummy.reshape([width, height])
quit()
"""

#im_array = rf.read_raw_16(dir_d,filename, width, height, 
#                          maxNumOfFiles_per_serie)

files = rf.list_files(dir_d,filename)
files.sort()


im_array=np.zeros((len(files),2016,2016))
mean_value=np.zeros(len(files))
for i in tqdm(range(len(files))):
    #print('Reading file: ', files[i])

    im_array[i,:,:],hdr=ut.read_Tumag(dir_d+files[i], write_fits = False, 
                        fits_file_name = 'Image.fits', plot_flag = False,
                            vmin = 0, vmax = 4096, onlyheader = False)
    mean_value[i]= np.mean(im_array[i,900:1100,900:1100])
    rms=np.std(im_array[i,:,:])
    if i == 0:
        plt.imshow(im_array[i,:,:],cmap='gray',vmin=mean_value[i]-2*rms,
                   vmax=mean_value[i]+2*rms)
        plt.colorbar()
        plt.show()

exp_times=np.arange(min_t, min_t+len(files)*dt, dt)

plt.close()
plt.plot(exp_times, mean_value, 'o-')
plt.xlabel('Exposure time (ms)')
plt.ylabel('Mean value')
plt.title('Mean value vs Exposure time')
plt.show()


"""
j=-2
im_diff=np.zeros((2048,2048,32))
for i in range(31):
    j+=2
    im_diff[:,:,i]=im_array[:,:,j+1]-im_array[:,:,j]

    plt.imshow(im_diff[:,:,i],cmap='gray',vmax=4,vmin=-4)
    plt.colorbar()
    plt.show()
    plt.close()

filename='peli.mp4'
pf.movie(im_diff,filename,resol=720,axis=2,fps=10,cbar='no',clims='no',vmin=-4,vmax=4)
quit()
"""
