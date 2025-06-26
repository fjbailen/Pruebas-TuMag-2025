


gain = '1' #1,2,3,4 or 5
Tset='20' #20
date='02Jul'
conditions='vac' #amb or vac

dir = './Cam1/'
LED='530'+'nm'
name =LED+'_'+conditions+'_'+date
named ='darks_'+conditions+'_'+date
bl = '-704'
mode='1'
time_format='int' #Format of the exposure times in images: 'int' or 'float'
limit_frames=0#2
#Ojo. Cuidado con los archivos (nombres) VER *****

rxa=[500,1000]
rya=[500,1000]



# %%
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

tic=time.time()


#Dir name
dir_d = name+'_g'+gain+'_mod'+mode+'/'
if not os.path.isdir(dir+dir_d):
    dir_d =  name+'_g'+gain+'/'






#Read images 
inm,t_exp,image = read_data(dir,name,gain,bl,Tset,mode,limit_frames=limit_frames,\
time_format=time_format)
inm,t_exp,darks = read_data(dir,named,gain,bl,Tset,mode,limit_frames=limit_frames,\
time_format=time_format)

#Save dark-corrected images in a numpy array with format (Nx,Ny,frame,inm)
#save_as_npy(dir,dir_d,image,darks,t_exp,inm,dark_corr=True)
save_as_npy(dir,dir_d,darks,darks,t_exp,inm,dark_corr=False)

#Convert texp from Tlines to ms and save it into a npy file
t_exp=t_exp*0.02052
np.save(dir+dir_d+'t_exp.npy',t_exp)


quit()


#Rolling index to make index=0 to coincide with t=1 (if necessary)

if np.mean(image[:,:,0])>(2*np.mean(image[:,:,1])):
    image=np.roll(image,-1,axis=2)
    darks=np.roll(darks,-1,axis=2)
    print('WARNING: rolling of indices was performed!!!')
    quit()

"""
for i in range(62):
    print(np.mean(image[10:-10,10:-10,i]))
    plt.imshow(image[10:-10,10:-10,i])
    plt.colorbar()
    plt.show()
    plt.close()
quit()
"""

"""
Plots and statistics
"""
print('Images with light:',name)
print('Images in dark conditions:', named)
plots_and_statistics(dir,dir_d,name,image,darks,gain,Tset,mode,bl,t_exp,\
inm,rxa,rya,savefig='yes')
print('Elapsed time:',round((time.time()-tic)/60),'min')
