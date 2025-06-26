################################################################################
################################################################################
# SPGCam LIBRARY
################################################################################
################################################################################

import os, sys, getopt,re
import pandas as pd
import read_functions as rf
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
#from pick import pick
import matplotlib as mpl #dos
from scipy.stats import norm
from scipy.signal import gaussian
from mpl_toolkits.axes_grid1 import make_axes_locatable
STRFMT = '{:8.5f}'
PLT_RNG = 2

import scipy.ndimage.filters as filt

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows',
        'win64' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    platform= platforms[sys.platform]

    if platform == 'OS X':
        software='./sunriseCamTest.sh'
        ledstatus = './ledControl.sh 2'
        ledon='./ledControl.sh 1'
        ledoff='./ledControl.sh 0'
    elif platform == 'Linux':
        software='sunriseCamTest.sh'
        ledon='ledControl.sh 1'
        ledoff='ledControl.sh 2'
    elif platform == 'Windows':
        software='sunriseCamTest.bat'
        ledon='led_on.bat'
        ledoff='led_off.bat'
    else:
        assert False, "Unknown system"
        sys.exit(2)
    print('System: '+platform)
    return

def colorbar(mappable,labelsize=4):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=labelsize)
    return fig.colorbar(mappable, cax=cax)

def fits_open(num):
    '''helper function to load FITS data set'''
    dir = './'
    hdu_list = pyfits.open(dir+'dark_'+'{:02d}ms_100.fits'.format(num))
    print('reading image ',dir+'dark_'+'{:02d}ms_100.fits'.format(num))
    return hdu_list[0].data.astype(np.dtype('d'))

def i_h_e(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape)#, cdf

def list_files(dir,pattern):
    import os, fnmatch
    listOfFiles = os.listdir(dir)
    n_files = 0
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            try:
                files = files + ',' + entry
            except:
                files = files = entry
            n_files += 1
    print(n_files,' files found')
    if n_files == 0:
        return n_files
    lines = files.split(',')
    return lines

def read_raw(dir,pattern,Nx=2048,Ny=2048):
    files = list_files(dir,pattern)
    #Sort files correctly: 0,1,2,3,4,...
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)',text) ]
    files.sort(key=natural_keys)
    #Initialization
    Nx=int(Nx)
    Ny=int(Ny)
    image = np.zeros([Nx,Ny,len(files)],dtype=np.float32)
    for i in range(len(files)):
        #print(dir+files[i])
        im_dummy = np.fromfile(dir+files[i],dtype='>i2')
        im_dummy = im_dummy.reshape([Nx, Ny])
        image[:,:,i] = im_dummy.astype(np.float32)
        print('read ',dir+files[i])
    print('read ',len(files),' ',dir,pattern,' images')
    return image

def convert_to_raw(dir,pattern):
    files = list_files(dir,pattern)
    image = np.zeros([2048,2048,len(files)],dtype=np.float32)
    for i in range(len(files)):
       filein = dir+files[i]
       fileout = dir+files[i]+'.raw'
       cmd = 'readcam/./imread -i '+filein+' -o '+fileout+' -w 2048 -h 2048'
       f = os.popen(cmd)
       now = f.read()
       print("Running.... ", now)

def find_times(dir,dir_d,name,gain,Tset,mode,bl):
    pattern = name+'_g'+gain+'_t*_T'+Tset+'_mod'+mode+'_bl'+bl
    files = list_files(dir+dir_d,pattern+'_0.raw')

    if files == 0:
        pattern = name+'_g'+gain+'_t*_T'+Tset+'_bl'+bl
        files = list_files(dir+dir_d,pattern+'_0.raw')
    if files==0:
        print("No files with given pattern recognized: ",pattern )
        quit()
        pass
    #get exp time from files
    t_exp = np.zeros((len(files)))

    i = 0
    for ss in files:
        left = ss.find('_t')+2
        right = ss.find('_T')
        t_exp[i] = int(ss[left:right])
        i = i + 1
    return np.array(sorted(t_exp))

def init_arrays(Nx,Ny,t_exp):
    Nx=int(Nx)
    Ny=int(Ny)
    try:
        Ntexp=t_exp.size
    except:
        Ntexp=len(t_exp)
    #Mean of the Nim images at a given exp time
    datam = np.zeros([Nx,Ny,Ntexp],dtype=np.float32)
    #RMS of the Nim images at a given exp time (px based std)
    datas = np.zeros([Nx,Ny,Ntexp],dtype=np.float32)
    #Spatial mean value of image at a given exp time (mean(datam))
    spmean = np.zeros([Ntexp],dtype=np.float32)
    #std value of std image at a given exp time (mean(datas))
    spvar = np.zeros([Ntexp],dtype=np.float32)
    #std mean value of image std at a given exp time (mean(std(image(i))))
    spiar = np.zeros([Ntexp],dtype=np.float32)

    #mean of difference between pair of franes at a given exp time
    datad = np.zeros([Nx,Ny,Ntexp],dtype=np.float32)
    #mean value of datad at a given exp time (mean(datad))
    sdmean = np.zeros([Ntexp],dtype=np.float32)
    #std mean value of image diff datad at a given exp time (mean(std(datads)))
    spdar = np.zeros([Ntexp],dtype=np.float32)
    return datam, datas, spmean, spvar, spiar, datad, sdmean, spdar

def get_variables(argv):
    try:
        opts, args = getopt.getopt(argv,"f:g:n:b:ho:v", ["help", "output="])
        # : ->  means args given
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)


    name = '*'
    gain = '*'
    mode='*'
    temp='*'
    nim = '*'
    bl = '*'
    output = None
    verbose = False
    showim = False

    for opt, arg in opts:
        if opt == "-v":
            verbose = True
            showim = True
        elif opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-f"):
            name = str(arg)
        elif opt in ("-g"):
            gain = str(arg)
        elif opt in ("-p"):
            nim = str(arg)
        elif opt in ("-b"):
            bl = str(arg)
        elif opt in ("-o", "--output"):
            output = str(arg)
        else:
            assert False, "unhandled option"
    return name, gain, mode, temp, nim, bl, output, verbose, showim

def choose_file(list):
    if list == 0:
        print("No files with given pattern recognized: ",pattern )
        quit()

    if len(list) > 1:
        title = 'Please choose file: '
        option, index = pick(list, title, indicator='=>', default_index=1)
        print(option, index)
        file = list[index]
    else:
        try:
            file = list[0]
        except FileNotFoundError:
            file = list[0][:-4]
    return file

def load_data(path):
    loaded = np.load(path)
    datam = loaded['datam']
    datas = loaded['datas']
    spmean = loaded['spmean']
    spvar = loaded['spvar']
    spiar = loaded['spiar']
    t_exp = loaded['t_exp']
    try:
        sdmean = loaded['sdmean'] #check if exist
    except:
        sdmean = spmean
    try:
        spdar = loaded['spdar']
    except:
        spdar = spvar
    return datam,datas,spmean,spvar,spiar,t_exp,sdmean,spdar

def save_parameters(dir,name,tmin,tmax,deltat,nim,bl,mode,Tset,lgain,Nx,Ny,deltax,deltay):
    datos=pd.Series()
    datos=datos.append(pd.Series([tmin,tmax,deltat],index=['tmin','tmax','deltat']))
    datos=datos.append(pd.Series([nim,bl,mode,Tset],index=['nim','bl','mode','Tset']))
    datos=datos.append(pd.Series([Nx,Ny,deltax,deltay],\
    index=['hsize','vsize','hoffset','voffset']))
    datos=datos.append(pd.Series([name],index=['name']))
    if len(lgain)==1:
        datos=datos.append(pd.Series([lgain],index=['lgain']))
    else:
        print('Gain parameter could not be saved in CSV file')
    datos.to_csv(dir+name+'.csv',header=False)
    return

def import_parameters(dir,name):
    params=pd.read_csv(dir+name+'.csv',sep=',',index_col=0,squeeze=True,\
    header=None)
    mode=params['mode']
    nim=params['nim']
    bl=params['bl']
    Tset=params['Tset']
    tmin=np.float64(params['tmin'])
    tmax=np.float64(params['tmax'])
    deltat=int(params['deltat'])
    Nx=str(params['hsize'])
    Ny=str(params['vsize'])
    deltax=str(params['hoffset'])
    deltay=str(params['voffset'])
    return mode,nim,bl,Tset,tmin,tmax,deltat,Nx,Ny,deltax,deltay

def get_images(name,mode,lgain,tmin,tmax,deltat,nim,bl,Tset,Nx,Ny,deltax,deltay):
    """
    The main difference with respect to SPGCam_lib in sunriseTestCam_Nov
    is that exposure times are not set logarithmically, but linearly in
    steps of 'delta_t'
    """
    #t_exp_lg = np.floor(np.exp(np.linspace(np.log(tmin), np.log(tmax), Ntime)))
    t_exp_lg=np.arange(tmin, tmax, deltat)
    t_exp_lg = t_exp_lg*1e3 #Conversion from ms to microsec

    for i in range(len(lgain)):
        gain = lgain[i]
        dir='./'+name+'_g'+gain+'/'
        try:
            os.mkdir(dir)
            os.mkdir(dir+'single_images')
        except FileExistsError:
            print(dir+' already created')
        save_parameters(dir,name,int(tmin),int(tmax),int(deltat),nim,bl,\
        mode,Tset,gain,Nx,Ny,deltax,deltay)
        for i in range(t_exp_lg.size):
            time = str(int(t_exp_lg[i]))
            time_ms=str(int(t_exp_lg[i]*1e-3))
            timetrigg=str(int(2*t_exp_lg[i]))

            cmd = 'sunriseCamTest.bat '+nim+' '+time+' '+gain+\
            ' ' + Tset + ' '+name+'_g'+gain+'_t'+time_ms+'_T'+Tset+'_mod'\
            +mode+'_bl'+bl+' '+name+'_g'+gain+ ' 1 '+ ' '+deltax+' '+Nx\
            +' '+deltay+' '+ Ny+' 0 '+bl+ ' '+mode #+ ' ' + timetrigg
            os.system(cmd)
            for im in range(np.int(nim)):
                file = name+'_g'+gain+'_t'+time_ms+'_T'+Tset+'_mod'+mode\
                +'_bl'+bl+'_'+str(int(im))
                rf.imread(dir,file,file+'.raw',Nx,Ny)
                rf.delFiles(name+'_g'+gain+'\\',file) #del files

            image = read_raw(dir,file+'.raw',Nx=Nx,Ny=Ny)
            image=image[:,:,0]
            mu=np.mean(image)
            rms=np.std(image)
            if Nx=='2048':
                plt.imshow(image,cmap='gray',clim=(0.5*mu,1.5*mu))
            else:
                plt.imshow(image,cmap='gray',clim=(mu-2*rms,mu+2*rms))
            plt.colorbar()
            plt.title('Mean='+str(mu))
            plt.savefig(dir+'single_images/'+file+'.png')
            plt.close()

def get_images_bursts(name,mode,lgain,tmin,tmax,deltat,nim,bl,Tset,Nx,Ny,deltax,deltay):
    """
    Similar to get_images but with burts. The maximum number of images is linimited
    to 80. The exposure time should not be larger than 32 ms. The time between
    triggers is set to the maximum value: 64 ms.
    """
    #t_exp_lg = np.floor(np.exp(np.linspace(np.log(tmin), np.log(tmax), Ntime)))
    t_exp_lg=np.arange(tmin, tmax, deltat)
    t_exp_lg = t_exp_lg*1e3 #Conversion from ms to microsec
    timetrigg=str(int(64e3))
    for i in range(len(lgain)):
        gain = lgain[i]
        dir='./'+name+'_g'+gain+'/'
        try:
            os.mkdir(dir)
            os.mkdir(dir+'single_images')
        except FileExistsError:
            print(dir+' already created')
        save_parameters(dir,name,int(tmin),int(tmax),int(deltat),nim,bl,\
        mode,Tset,gain,Nx,Ny,deltax,deltay)
        for i in range(t_exp_lg.size):
            time = str(int(t_exp_lg[i]))
            time_ms=str(int(t_exp_lg[i]*1e-3))

            cmd = 'sunriseCamTest.bat '+ nim +' '+time+' '+gain+\
            ' ' + Tset + ' '+name+'_g'+gain+'_t'+time_ms+'_T'+Tset+'_mod'\
            +mode+'_bl'+bl+' '+name+'_g'+gain+ ' '+ '80' +' '+ ' '+deltax+' '+Nx\
            +' '+deltay+' '+ Ny+' 0 '+bl+ ' '+mode + ' ' + timetrigg
            os.system(cmd)
            for im in range(np.int(nim)):
                for j in range(80):
                    file = name+'_g'+gain+'_t'+time_ms+'_T'+Tset+'_mod'+mode\
                    +'_bl'+bl+'_'+str(int(im*80+j))
                    rf.imread(dir,file,file+'.raw',Nx,Ny)
                    rf.delFiles(name+'_g'+gain+'\\',file) #del files


def adquire_images(name,mode,gain,exp_times,nim,bl,Tset,Nx='2048',Ny='2048',deltax='0',deltay='0',path='./'):
    #Need to put warnings in case of mising par
    exp_times_microsec = exp_times*1e3 #Conversion from ms to microsec

    filename = name+'_g'+gain+'_mod'+mode
    dir=path+filename+'/'
    try:
        print('folder ',dir,' created')
        os.mkdir(dir)
        print('folder ',dir,' single_images',' created')
        os.mkdir(dir+'single_images')
    except FileExistsError:
        print(dir+' already created')
    save_parameters(dir,name,int(np.min(exp_times)),int(np.max(exp_times)),\
        int(exp_times.size),nim,bl,mode,Tset,gain,Nx,Ny,deltax,deltay)
    for i in range(exp_times.size):
        time = str(int(exp_times_microsec[i]))
        time_ms=str("{:10.4f}".format(exp_times[i])).strip()#str(int(exp_times[i]))
        cmd = software+' '+nim+' '+time+' '+gain+\
        ' ' + Tset + ' '+name+'_g'+gain+'_t'+time_ms+'_T'+Tset+'_mod'\
        +mode+'_bl'+bl+' '+path+filename+ ' 1 '+ ' '+deltax+' '+Nx\
        +' '+deltay+' '+ Ny+' 0 '+bl+ ' '+mode
        print('Executing camera... ',cmd)
        os.system(cmd)
        for im in range(np.int(nim)):
            file = name+'_g'+gain+'_t'+time_ms+'_T'+Tset+'_mod'+mode\
            +'_bl'+bl+'_'+str(int(im))
            rf.imread(dir,file,file+'.raw',Nx,Ny,OS=platform)
            rf.delFiles(path+filename,file,OS=platform) #del files

        image = read_raw(dir,file+'.raw',Nx=Nx,Ny=Ny)
        image=image[:,:,0]
        mu=np.mean(image)
        rms=np.std(image)
        if Nx=='2048':
            plt.imshow(image,cmap='gray',clim=(0.5*mu,1.5*mu))
        else:
            plt.imshow(image,cmap='gray',clim=(mu-2*rms,mu+2*rms))
        plt.colorbar()
        plt.title('Mean='+str(mu))
        plt.savefig(dir+'single_images/'+file+'.png')
        plt.close()

def roi(yes_no):
    if yes_no=='yes':
        Nx='400'
        Ny=Nx
        deltax='800' #'1500'
        deltay='1500' #'1000'
        return Nx,Ny,deltax,deltay
    if yes_no=='no':
        return '2048','2048','0','0'

def show_one(img,vmax=None,vmin=None,xlabel='pixel',ylabel='pixel',title='Image no title',cbarlabel='Some units',save=None):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    if vmin == None and vmax == None:
        im = ax.imshow(img, cmap='gray',vmin=img.mean() - PLT_RNG * img.std(),
           vmax=img.mean() + PLT_RNG * img.std(), interpolation='none')
    elif vmin == None:
        im = ax.imshow(img, cmap='gray',vmin=img.mean() - PLT_RNG * img.std(),
           vmax=vmax, interpolation='none')
    elif vmax == None:
        im = ax.imshow(img, cmap='gray',vmin=vmin,
           vmax=img.mean() + PLT_RNG * img.std(), interpolation='none')
    else:
        im = ax.imshow(img, cmap='gray',vmin=vmin,
           vmax=vmax, interpolation='none')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbarlabel)
    if save:
        plt.savefig(save,dpi=300)
        plt.close()
    else:
        plt.show()
    return

def squar(n):
    column = np.int(np.sqrt(n))
    if column < n:
        row = column + 1
        if row*column < n:
            column = column + 1
    else:
        row = column
    return row,column

def show_all(image):
    ishape = image.shape
    row,column = squar(ishape[2])
    fig, maps = plt.subplots(row,column, sharex='col', sharey='row',figsize=(12,12))
    plt.subplots_adjust(top=0.92)
    for i in range(ishape[2]):
        #themedian = median(sp[:,:,i])
        #image = image_histogram_equalization(sp[:,:,i])[0]
        im = maps[divmod(i, column)].imshow(image[:,:,i],\
        cmap='gray',vmin=image[:,:,i].mean() - PLT_RNG * image[:,:,i].std(),
               vmax=image[:,:,i].mean() + PLT_RNG * image[:,:,i].std(),\
                interpolation='none')
        colorbar(im)
    plt.show()
    plt.close()

def exptimes(mode,lgain):
    """
    Minimum, maximum and step for the exposure times depending on the
    mode and gain
    """
    tmin=1
    if mode=='0' or mode=='1':
        if lgain=='0':
            tmax=2250
            deltat=50 #ms
        elif lgain=='1':
            tmax=1500
            deltat=30 #ms
        elif lgain=='3':
            tmax=900

            deltat=20 #ms
    elif mode=='2':
        tmax=400#1000 #Maximum exposure time [ms]
    elif mode=='3':
        if lgain=='4':
            tmax=500#800#1200
        elif lgain=='0':
            tmax=1000
        elif lgain=='1':
            tmax=800
    elif mode=='4':
        if lgain=='7':
            tmax=20
        if lgain=='4':
            tmax=110
    return tmin,tmax,deltat

def get_value(file,stringToMatch):
    with open(file, 'r') as file:
	    for line in file:
		    if stringToMatch in line:
			    matchedLine = line
			    break
    loc_value = matchedLine.find('=')
    value = matchedLine[loc_value+1:loc_value+4]
    return value

def find_last_string(entry,string,verbose=False):
    found = 0
    found_where = []
    while found < len(entry):
        found = entry.find(string, found)
        if found == -1:
            break
        found_where.append(found)
        found += 2 # +2 because len('_T') == 2
    if verbose:
        print('string found at', found_where)
    return found_where

def change_names(dir_d,light,cond,date,temp):
    dir = './'
    print('Folder found?',os.path.isdir(dir_d),dir_d)
    if not os.path.isdir(dir+dir_d):
        print('Folder not found')
        quit()
    half = dir_d[:-1]
    #Detect number of images and esposure times!
    nm = dir+dir_d
    #dirm = half+'*'+'.img'
    dirm = '*'+'.img'
    print('Searching images of the type',dirm)
    list = list_files(nm,dirm)
    list.sort()

    print('1st image:',list[0])
    print('Last image:',list[-1])

    gainl=['1','2','3','4','5']
    k=0
    for gain in gainl:
        if gain=='1':
            time=np.arange(1,700,23) #IN TLINES UNITS!!
        if gain=='2':
            time=np.arange(1,542,18) #IN TLINES UNITS!!
        if gain=='3':
            time=np.arange(1,362,12) #IN TLINES UNITS!!
        if gain=='4':
            time=np.arange(1,1112,37) #IN TLINES UNITS!!
        if gain=='5':
            time=np.arange(1,392,13) #IN TLINES UNITS!!

        for i in time:
            for j in range(2):
                newname=light+'_'+cond+'_'+date+'_'+'g'+gain+'_t'+str(i)\
                +'_T'+temp+'_mod1_bl-704_'+str(j)
                os.rename(dir_d+list[k],dir_d+newname+'.img')
                k+=1
    print(k, 'files renamed')

    #os.rename()
    return


def read_data(dir,name,gain,bl,Tset,mode,Nx=2048,Ny=2048,limit_frames=0,
              time_format='float'):
    """
    Read data from the camera. The function detects the number of images
    and the exposure times. The images are read in the format
    name+'_g'+gain+'_t'+time+'_T'+Tset+'_mod'+mode+'_bl'+bl+'_'+str(im)
    The images are saved in the array image[:,:,i*inm+j] where i is the
    exposure time and j is the image number. The function returns the
    number of images per exposure time (inm), the exposure times (t_exp)
    and the image array (image)."""

    dir_d = name+'_g'+gain+'_mod'+mode+'/'

    #Search directory
    print(os.path.isdir(dir_d),dir_d)
 
    if not os.path.isdir(dir+dir_d):
        dir_d = name+'_g'+gain+'/'
        print(os.path.isdir(dir+dir_d),dir+dir_d)
   
    half = name+'_g'+gain+'_t'
    other_half = '_T'+Tset+'_mod'+mode+'_bl'+bl
    #Detect number of images and esposure times!
    nm = dir+dir_d
    dirm = half+'*'+other_half+'*.img'
    list = list_files(nm,dirm)
    print('Searching for files of the type',dirm)
    list.sort()
    first = 0
    inm = 0
    for file in list:   #E.g.,:'darks_amb_31_Oct_g4_t142_T20_mod1_bl-830_4.img'
        loc_raw = file.find('.img')
        loc_bl = file.find('bl-')
        loc_ = file.find('_',loc_bl)
        inm_counter = int(file[loc_+1:loc_raw])
        if inm_counter > inm:
            inm = inm_counter
        #loc_t = file.find('_t')
        #loc_T = file.find('_T')
        loc_t = find_last_string(file,'_t')
        if len(loc_t) != 1:
            loc_t = loc_t[len(loc_t)-1]
        else:
            loc_t = int(loc_t[0])

        loc_T = find_last_string(file,'_T')
        if len(loc_T) != 1:
            loc_T = loc_T[len(loc_T)-1]
        else:
            loc_T = int(loc_T[0])
        time = file[loc_t+2:loc_T]
        if first == 0:
            t_exp = [time]
            first = 1
        else:
            for x in t_exp:
                if x == time:
                    break
            else:
                t_exp.append(time)
    if time_format=='float':
        t_exp = [float(i) for i in t_exp]  #Changed int to float (19 jan)
    else:
        t_exp=[int(i) for i in t_exp]
    t_exp.sort()
    t_exp = np.asarray(t_exp)
    inm = inm + 1
    print('Number of frames per exposure time: ',inm)
    if limit_frames != 0:
        inm = limit_frames
        print('Number of frames per exposure time has been limited to: ',inm)

    Nx=int(Nx)
    Ny=int(Ny)
    image = np.zeros([Nx,Ny,t_exp.size*inm],dtype=np.float32)


    for i in range(t_exp.size):
        for j in range(inm):
            #file = nm+half+str(t_exp[i])+other_half+'_'+str(j)+'.raw'
            #timeis = str("{:10.4f}".format(t_exp[i])).rstrip('0').rstrip('.').strip() #*****
            if time_format=='float':
                timeis = str("{:10.4f}".format(t_exp[i])).strip()
            else:
                timeis = str("{}".format(t_exp[i])).strip()
            file = nm+half+timeis+other_half+'_'+str(j)+'.img'
            im_dummy = np.fromfile(file,dtype='<i2')
            im_dummy = im_dummy.reshape([Nx, Ny])
            image[:,:,i*inm+j] = im_dummy.astype(np.float32)

          



        #try:
            #file_log = half+timeis+other_half+'_housekeeping.log*'
            #list_log = list_files(nm,file_log)
            #value_t = get_value(nm+list_log[0],'GSENSE400_temperature')
        #except:
            #value_t='not found'

        print('# of images:',inm)
        print('Exposure time:',t_exp[i])
        #print('Temperature:',value_t,'ºC')

    print('----------------------------------')
    print('Total # of images read: ',t_exp.size*inm)
    return inm,t_exp,image

def read_data_dir(name,dir_d,gain,bl,Tset,mode,Nx=2048,Ny=2048,limit_frames=0,time_format='float'):
    dir = './'

    print(os.path.isdir(dir_d),dir_d)
    if not os.path.isdir(dir+dir_d):
        dir_d = name
        print(os.path.isdir(dir_d),dir_d)
    half = name+'_g'+gain+'_t'
    other_half = '_T'+Tset+'_mod'+mode+'_bl'+bl
    #Detect number of images and esposure times!
    nm = dir+dir_d + '/'
    dirm = half+'*'+other_half+'*.raw'
    list = list_files(nm,dirm)
    print(dirm)
    list.sort()
    first = 0
    inm = 0
    for file in list:   #'darks_amb_31_Oct_g4_t142_T20_mod1_bl-830_4.raw'
        loc_raw = file.find('.raw')
        loc_bl = file.find('bl-')
        loc_ = file.find('_',loc_bl)
        inm_counter = int(file[loc_+1:loc_raw])
        if inm_counter > inm:
            inm = inm_counter
        #loc_t = file.find('_t')
        #loc_T = file.find('_T')
        loc_t = find_last_string(file,'_t')
        if len(loc_t) != 1:
            loc_t = loc_t[len(loc_t)-1]
        else:
            loc_t = int(loc_t[0])

        loc_T = find_last_string(file,'_T')
        if len(loc_T) != 1:
            loc_T = loc_T[len(loc_T)-1]
        else:
            loc_T = int(loc_T[0])
#        try:
#            if len(loc_T) != 1:
#                loc_T = loc_T[len(loc_T)-1]
#        except:
#            loc_T = int(loc_T[0][0])
        time = file[loc_t+2:loc_T]
        if first == 0:
            t_exp = [time]
            first = 1
        else:
            for x in t_exp:
                if x == time:
                    break
            else:
                t_exp.append(time)
    if time_format=='float':
        t_exp = [float(i) for i in t_exp]  #Changed int to float (19 jan)
    else:
        t_exp=[int(i) for i in t_exp]
    t_exp.sort()
    t_exp = np.asarray(t_exp)

    inm = inm + 1
    print('Number of frames per exposure time: ',inm)
    if limit_frames != 0:
        inm = limit_frames
        print('Number of frames per exposure time has been limited to: ',inm)

    Nx=int(Nx)
    Ny=int(Ny)
    image = np.zeros([Nx,Ny,t_exp.size*inm],dtype=np.float32)
    for i in range(t_exp.size):
        for j in range(inm):
            #file = nm+half+str(t_exp[i])+other_half+'_'+str(j)+'.raw'
            #timeis = str("{:10.4f}".format(t_exp[i])).rstrip('0').rstrip('.').strip() #*****
            if time_format=='float':
                timeis = str("{:10.4f}".format(t_exp[i])).strip()
            else:
                timeis = str("{}".format(t_exp[i])).strip()
            file = nm+half+timeis+other_half+'_'+str(j)+'.raw'
            im_dummy = np.fromfile(file,dtype='>i2')
            im_dummy = im_dummy.reshape([Nx, Ny])
            #image[:,:,i*2+j] = im_dummy.astype(np.float32)
            image[:,:,i*inm+j] = im_dummy.astype(np.float32)

        file_log = half+timeis+other_half+'_housekeeping.log*'
        list_log = list_files(nm,file_log)
        try:
            value_t = get_value(nm+list_log[0],'GSENSE400_temperature')
        except:
            value_t='not found'
        print('# of images:',inm)
        print('Exposure time:',t_exp[i])
        print('Temperature:',value_t,'ºC')
    print('----------------------------------')
    print('Total # of images read: ',t_exp.size*inm)
    return inm,t_exp,image


def estadistica_data(t_exp,inm,data,rx=[0,2048],ry=[0,2048]):
    rxn = rx[1]-rx[0]+1
    ryn = ry[1]-ry[0]+1
    #mean image at a given exp time
    datam = np.zeros([rxn,ryn,t_exp.size],dtype=np.float32)
    #std image at a given exp time (px based std)
    datas = np.zeros([rxn,ryn,t_exp.size],dtype=np.float32)
    #mean value of image at a given exp time (mean(datam))
    spmean = np.zeros([t_exp.size],dtype=np.float32)
    #mean value of std image at a given exp time (mean(datas)) - px based std
    spvar = np.zeros([t_exp.size],dtype=np.float32)
    #mean value of std image std at a given exp time (mean(std(image(i))))
    spiar = np.zeros([t_exp.size],dtype=np.float32)
    #mean of difference between pair of franes at a given exp time
    datad = np.zeros([rxn,ryn,t_exp.size],dtype=np.float32)
    #mean value of datad at a given exp time (mean(datad))
    sdmean = np.zeros([t_exp.size],dtype=np.float32)
    #std mean value of image diff datad at a given exp time (mean(std(datads)))
    spdar = np.zeros([t_exp.size],dtype=np.float32)

    dummy_pair = np.zeros([rxn,ryn,inm//2],dtype=np.float32)
    dummy_std  = np.zeros([inm//2],dtype=np.float32)

    for i in range(t_exp.size):
        image = data[rx[0]:rx[1]+1,ry[0]:ry[1]+1,inm*i:inm*i+inm]
        datam[:,:,i] = np.mean(image,axis=(2)) #noise goes down by sqrt(inm)
        datas[:,:,i] = np.var(image,axis=(2)) #variance for single pixels
        spmean[i] = np.mean(datam[:,:,i],axis=(0,1))
        #mean value of image (mean introduce artifacts)
        spvar[i] = np.mean(datas[:,:,i],axis=(0,1))
        #mean value of variance (along images - Photon)
        spiar[i] = np.mean(np.var(image[:,:,:],axis=(0,1)))
        media=datam[:,:,i]

        #mean value of variance of image (spatial - FPN)
        for j in range(inm//2):
            dummy_pair[:,:,j] = image[:,:,2*j] - image[:,:,2*j+1] #noise image NO FPN
            dummy_std[j] = np.var(dummy_pair[:,:,j],axis=(0,1)) #STD of NOISE IMAGE
        datad[:,:,i] = np.mean(dummy_pair,axis=(2)) #mean value of NOISE image
        sdmean[i] = np.mean(datad[:,:,i],axis=(0,1))
        spdar[i] = np.mean(dummy_std) #Same as spvar
    return datam,datas,spmean,spvar,spiar,datad,sdmean,spdar

def limit_statistics(mode,gain):
    limit_inf = 0
    if mode == '1':
        if gain == '0':
            limit = 800
        elif gain == '3':
            limit = 3500
        elif gain == '4':
            limit = 1200
            limit_inf = 500
        else:
            limit = 2000
    elif mode == '3':
        limit = 800
    elif mode == '0':
        limit = 2000
    return limit,limit_inf

def plots_and_statistics(dir,dir_d,name,image,darks,gain,Tset,mode,bl,t_exp,inm,rxa,rya,savefig='yes'):
    """
    STATISTICS FOR ILLUMINATED FRAMES
    """
    #Correct images from darks
    image_dc = image - darks  #remove all darks
    plt.imshow(image[:,:,3]-darks[:,:,3],cmap='gray')#,vmin=140,vmax=150)

    # # Data analysis
    #find_index = np.where(t_exp > 80.)
    #index = np.nonzero(t_exp>80)[0][0]*inm
    find_index = np.where(t_exp > 0.05)
    index = np.nonzero(t_exp>0.05)[0][0]*inm

    plt.imshow( darks[rxa[0]:rxa[1],rya[0]:rya[1],index] )

    rx=[0,2048]
    ry=[0,2048]

    #Figure with six subplots: dark, image, image-dark, etc.
    med = np.mean(darks[rxa[0]:rxa[1],rya[0]:rya[1],index])
    medi = np.mean(image[:,:,index])
    fig = plt.figure(figsize=(25,15))
    ax1 = fig.add_subplot(231)
    fig1 = ax1.imshow(darks[rx[0]:rx[1],ry[0]:ry[1],index],cmap='gray',vmin=med-5,vmax=med+5)
    colorbar(fig1,labelsize=10)
    ax2 = fig.add_subplot(232)
    fig2 = ax2.imshow(image[rx[0]:rx[1],ry[0]:ry[1],index],cmap='gray',vmin=medi-medi*0.2,vmax=medi+medi*0.2)
    colorbar(fig2,labelsize=10)
    ax3 = fig.add_subplot(233)
    fig3 = ax3.imshow(image[rx[0]:rx[1],ry[0]:ry[1],index]-darks[rx[0]:rx[1],ry[0]:ry[1],index+1],cmap='gray',vmin=medi-med-medi*0.2,vmax=medi-med+medi*0.2)
    colorbar(fig3,labelsize=10)
    ax4 = fig.add_subplot(234)
    fig4 = ax4.imshow(image_dc[rx[0]:rx[1],ry[0]:ry[1],index],cmap='gray',vmin=medi-med-medi*0.2,vmax=medi-med+medi*0.2)
    colorbar(fig4,labelsize=10)
    ax5 = fig.add_subplot(235)
    fig5 = ax5.imshow(darks[rx[0]:rx[1],ry[0]:ry[1],index]-darks[rx[0]:rx[1],ry[0]:ry[1],index+1],cmap='gray',vmin=-4,vmax=4)
    colorbar(fig5,labelsize=10)
    ax6 = fig.add_subplot(236)
    fig6 = ax6.imshow(image[rx[0]:rx[1],ry[0]:ry[1],index]-image[rx[0]:rx[1],ry[0]:ry[1],index+1],cmap='gray',vmin=-4,vmax=4)
    colorbar(fig6,labelsize=10)

    ax1.set_title('dark '+str(t_exp[index//2])+' ms')
    ax2.set_title('image')
    ax3.set_title('image-dark')
    ax4.set_title('image-dc')
    ax5.set_title('darks-darks')
    ax6.set_title('image-image')
    fig.suptitle(name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl)
    if savefig=='yes':
        plt.savefig(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+\
        '.png',dpi=600)


    plt.imshow(image[rxa[0]:rxa[1],rya[0]:rya[1],index]-darks[rxa[0]:rxa[1],\
    rya[0]:rya[1],index],cmap='gray')#,vmin=140,vmax=150)

    #Statistics of data
    datam,datas,spmean,spvar,spiar,datad,sdmean,spdar = \
    estadistica_data(t_exp,inm,image,rx=rxa,ry=rya)

    plt.imshow(datam[:,:,1])
    plt.colorbar()

    dd = np.abs(spmean-np.median(spmean))
    index=np.where(dd == np.min(dd))
    #plt.close()
    #plt.imshow(np.squeeze(image_dc[400:900,300:800,index[0][0]])/np.mean(\
    #image_dc[400:900,300:800,index[0]]),cmap='gray')#,vmin=spmean[index]*0.9,vmax=spmean[index]*1.1,cmap='gray')
    #plt.colorbar()
    #plt.show()

    PRNU = np.std(image_dc[rxa[0]:rxa[1],rya[0]:rya[1],index[0]])/np.mean(\
    image_dc[rxa[0]:rxa[1],rya[0]:rya[1],index[0]])*100



    """
    Fit of data to obtain gain, read noise, FPN ...
    """
    #Fit linearity
    limit,limit_inf=limit_statistics(mode,gain)
    which_ones = np.where((spmean < limit) & (spmean > limit_inf))
    #which_ones=np.arange(1,20,1)

    fitlinear = np.polyfit(t_exp[which_ones],spmean[which_ones], 1)
    fitlinear_fns = np.poly1d(fitlinear)
    linearity=(np.max( (spmean[which_ones] - fitlinear_fns(t_exp[which_ones])))\
    + np.abs(np.min( (spmean[which_ones] - fitlinear_fns(t_exp[which_ones])))))\
     / np.max(spmean[which_ones])*100 #Linearity in %
    off_set =fitlinear[0]

    #Fit to get gain and readout noise
    fitgain = np.polyfit(spdar[which_ones]/2,spmean[which_ones], 1)
    fitgain_fns = np.poly1d(fitgain)
    calc_gain = 1/fitgain[0] #DN/e
    read_noise = fitgain[1] #DN (read noise)
    rn = np.sqrt(read_noise*calc_gain) #Read noise in electrons

    #Fit to get FPN
    fitFPN = np.polyfit(spiar[which_ones],spmean[which_ones], 2)
    #fitFPN = np.polyfit(spiar[:],spmean[:], 2)
    fitFPN_fns = np.poly1d(fitFPN)

    #Plots
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(131)
    ax1.plot(t_exp,spmean,'ok', ms=4)
    ax1.plot(t_exp,fitlinear_fns(t_exp),'k', ms=2,color='r')
    ax2 = fig.add_subplot(132)
    ax2.plot(spdar/2,spmean,'ok', ms=4)
    ax2.plot(spdar/2,fitgain_fns(spdar/2),'k', ms=2,color='r')
    ax3 = fig.add_subplot(133)
    ax3.plot(spiar,spmean,'ok', ms=4)
    ax3.plot(spiar,fitFPN_fns(spiar),'k', ms=2,color='r')
    ax1.grid(True,which="both",ls="--",color='0.65')
    ax2.grid(True,which="both",ls="--",color='0.65')
    ax3.grid(True,which="both",ls="--",color='0.65')

    ax1.set_xlabel('Exp Time')
    ax1.set_ylabel('Counts [DN]')
    ax2.set_xlabel('Photon Noise [DN$^2$]')
    ax2.set_ylabel('Counts [DN]')
    ax3.set_xlabel('Noise (FPN)')
    ax3.set_ylabel('Counts [DN]')

    xlim1 = [np.floor(np.min(t_exp))*0.9 , np.ceil(np.max(t_exp))*1.1]
    ylim1 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    xlim2 = [np.floor(np.min(spdar/2))*0.9 , np.ceil(np.max(spdar/2))*1.1]
    ylim2 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    xlim3 = [np.floor(np.min(spiar))*0.9 , np.ceil(np.max(spiar))*1.1]
    ylim3 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]

    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)
    ax3.set_xlim(xlim3)
    ax3.set_ylim(ylim3)

    ax1.set_title('Mode='+mode+'; gain='+gain+'; BL='+bl+' T='+Tset)
    ax1.set_title('Linearity= '+str("{:10.4f}".format(linearity)).strip()\
    +' [%]'+'  R/N='+str("{:2.2f}".format(rn))+' e$^-$')
    ax2.set_title('Off-set='+str("{:2.2f}".format(off_set))+' Gain='\
    +str("{:2.4f}".format(calc_gain))+' [DN/e$^-$]')
    ax3.set_title('FPN[%] = '\
    +str("{:2.2f}".format(np.sqrt(np.abs(fitFPN[0])*100.) ) )\
    +' PRNU[%] = '+str("{:2.2f}".format(PRNU)))

    fig.suptitle(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl\
    +' -- Max: '+str("{:2.4f}".format(1/fitgain[0]*90e3))+' DR='+str("{:2.2f}".format(9e4/rn)))
    if savefig=='yes':
        plt.savefig(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'\
        +bl+'_curves.png',dpi=600)

    # %%
    fig = plt.figure(figsize=(14,8))
    #ax1 = fig.add_subplot(131)
    #ax1.plot(t_exp,spmean,'ok', ms=4)
    #ax1.plot(t_exp,fitlinear_fns(t_exp),'k', ms=2,color='r')
    ax2 = fig.add_subplot(122)
    ax2.plot(spmean,np.sqrt(spdar/2),'ok', ms=4,color='r')
    #ax2.plot(fitgain_fns(spdar/2),spdar/2,'k', ms=2,color='r')
    ax2.plot(spmean,np.sqrt(spiar),'ok', ms=4,color='b')
    #ax2.plot(fitFPN_fns(spiar),spiar,'k', ms=2,color='b')
    #ax1.grid(True,which="both",ls="--",color='0.65')
    ax2.grid(True,which="both",ls="--",color='0.65')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim([np.floor(np.min([np.sqrt(spdar/2),np.sqrt(spiar)]))*0.9 ,\
     np.ceil(np.max([np.sqrt(spdar/2),np.sqrt(spiar)]))*1.1])
    ax2.set_xlim([np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1])

    #ax1.set_xlabel('Exp Time')
    #ax1.set_ylabel('Mean of image [DN]')
    ax2.set_ylabel('Photon Noise [DN]')
    ax2.set_xlabel('Mean of image')

    from scipy.optimize import curve_fit
    def myExpFunc(x, a, b):
        return a * np.power(x, b)
    x = np.linspace(0.1, 1000, 1000)
    newX = np.logspace(1, 10, base=10)

    popt1, pcov1 = curve_fit(myExpFunc, spmean[which_ones], np.sqrt(spiar[which_ones]))
    ax2.plot(newX, myExpFunc(newX, *popt1), linestyle='--',
        label="m = 1 FPN Regime fit ({0:.3f}*x**{1:.3f})".format(*popt1))
    Kg = 1/popt1[0]**(1/popt1[1])


    popt2, pcov2 = curve_fit(myExpFunc, spmean[which_ones], \
    np.sqrt(spdar[which_ones]/2))
    ax2.plot(newX, myExpFunc(newX, *popt2), linestyle='--',
        label="m = 1 Photon Regime fit ({0:.3f}*x**{1:.3f})".format(*popt2))
    Np = 1/popt2[0]**(1/popt2[1])


    fitFPN = np.polyfit(spmean[which_ones],np.sqrt(spiar[which_ones]), 2)
    fitFPN_fns = np.poly1d(fitFPN)
    ax2.plot(newX,fitFPN_fns(newX),'k', ms=5,color='r', linestyle='-',\
    label="m = 1 FPN Regime fit")

    ax2.legend(loc='best',fontsize=10)
    fig.suptitle(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+\
    ' -- Max: '+str("{:2.4f}".format(1/fitgain[0]*90e3)))
    if savefig=='yes':
        plt.savefig(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+\
        '_curves_log.png',dpi=600)


    """
    STATISTICS FOR DARKS
    """
    datam,datas,spmean,spvar,spiar,datad,sdmean,spdar=estadistica_data(t_exp,inm,darks,rx=rxa,ry=rya)

    sx = (rxa[1]-rxa[0])+1
    sy = (rya[1]-rya[0])+1
    #cond = t_exp > 19
    cond = t_exp > 0.05
    t_exp_fit = t_exp[cond]
    fitmean = np.transpose(np.polyfit(t_exp_fit,
        np.transpose(datam[:,:,cond].reshape(sx * sy, len(t_exp_fit))), 1))

    fitmean = fitmean.reshape(sx, sy, 2)
    fitstd = np.transpose(np.polyfit(t_exp_fit,
        np.transpose(datas[:,:,cond].reshape(sx * sy, len(t_exp_fit))), 1))
    fitstd = fitstd.reshape(sx, sy, 2)

    dcmap = fitmean[:, :, 0]
    """
    plt.close(fig='all')
    conflevel=4
    rms0=np.std(dcmap)
    mu0=np.mean(dcmap)
    max=np.max(dcmap)
    minim=mu0-conflevel*rms0
    maxim=mu0+conflevel*rms0
    print('minim',minim)
    print('maxim',maxim)
    if minim<0:
        minim=0

    #Histogram
    hst=np.ndarray.flatten(dcmap)
    n,bins,patches=plt.hist(hst,bins=np.linspace(minim,1.5*maxim,200),density=True)
    plt.xlabel('DN')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.show()
    quit()

    #Gaussian fit
    mu_hst,sigma_hst=norm.fit(hst)
    fitting=norm.pdf(bins,mu_hst,sigma_hst)
    plt.plot(bins,fitting,label='Gaussian fit')
    """

    show_one(dcmap*1e3,title='Dark current, mean=' +
              STRFMT.format(dcmap.mean(dtype=np.float64)*1e3) +
              ' ; std =' + STRFMT.format(dcmap.std(dtype=np.float64)*1e3),
              cbarlabel = 'DC [DN/s]',save=dir+dir_d+name+'_g'+gain+'_T'+\
              Tset+'_mod'+mode+'_bl'+bl+'_DC-map.png')

    offsetmap = fitmean[:, :, 1]
    show_one(offsetmap,title='Off-set map, mean=' +
              STRFMT.format(offsetmap.mean(dtype=np.float64)) +
              ' ; std =' + STRFMT.format(offsetmap.std(dtype=np.float64)),
              cbarlabel = 'Off-set [DN]',save=dir+dir_d+name+'_g'+gain+'_T'+\
              Tset+'_mod'+mode+'_bl'+bl+'_off-set-map.png')


    limit = 500
    which_ones2 = np.where((spmean < limit) )

    fitlinear2 = np.polyfit(t_exp[which_ones2],spmean[which_ones2], 1)
    fitlinear_fns2 = np.poly1d(fitlinear2)
    linearity2  =  (np.max( (spmean[which_ones2] - fitlinear_fns2(t_exp[which_ones2]))) /
                   + np.abs(np.min( (spmean[which_ones2] - fitlinear_fns2(t_exp[which_ones2])))) ) / np.max(spmean[which_ones2])*100

    fitgain2 = np.polyfit(spdar[which_ones2]/2,spmean[which_ones2], 1)
    fitgain_fns2 = np.poly1d(fitgain2)
    calc_gain2 = 1/fitgain2[0] #e/DN
    read_noise2 = fitgain2[1] #DN (read noise)
    rn = np.sqrt(read_noise2*calc_gain2) #electrons

    fitFPN2= np.polyfit(spiar[which_ones2],spmean[which_ones2], 2)

    fitFPN_fns2 = np.poly1d(fitFPN2)

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(131)
    ax1.plot(t_exp,spmean,'ok', ms=4)
    ax1.plot(t_exp,fitlinear_fns2(t_exp),'k', ms=2,color='r')
    ax2 = fig.add_subplot(132)
    ax2.plot(spdar/2,spmean,'ok', ms=4)
    ax2.plot(spdar/2,fitgain_fns2(spdar/2),'k', ms=2,color='r')
    ax3 = fig.add_subplot(133)
    ax3.plot(spiar,spmean,'ok', ms=4)
    ax3.plot(spiar,fitFPN_fns2(spiar),'k', ms=2,color='r')
    ax1.grid(True,which="both",ls="--",color='0.65')
    ax2.grid(True,which="both",ls="--",color='0.65')
    ax3.grid(True,which="both",ls="--",color='0.65')

    ax1.set_xlabel('Exp Time')
    ax1.set_ylabel('Counts [DN]')
    ax2.set_xlabel('Photon Noise [DN$^2$]')
    ax2.set_ylabel('Counts [DN]')
    ax3.set_xlabel('Noise (FPN)')
    ax3.set_ylabel('Counts [DN]')

    xlim1 = [np.floor(np.min(t_exp))*0.9 , np.ceil(np.max(t_exp))*1.1]
    ylim1 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    xlim2 = [np.floor(np.min(spdar/2))*0.9 , np.ceil(np.max(spdar/2))*1.1]
    ylim2 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    xlim3 = [np.floor(np.min(spiar))*0.9 , np.ceil(np.max(spiar))*1.1]
    ylim3 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    if mode == '4':
        if gain == '7':
            xlim2 = [0,1]
            ylim2 = [100,300]
    elif mode == '1':
        if gain == '0':
            xlim3 = [0,500]

    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)
    ax3.set_xlim(xlim3)
    ax3.set_ylim(ylim3)

    ax1.set_title('Mode='+mode+'; gain='+gain+'; BL='+bl+' T='+Tset)
    ax1.set_title('Linearity= '+str("{:10.4f}".format(linearity2)).strip()+' [%]'+'  R/N='+str("{:2.2f}".format(rn))+' e$^-$')
    ax2.set_title('Off-set='+str("{:2.2f}".format(fitgain2[1]))+' Gain='+str("{:2.4f}".format(1/fitgain2[0]))+' [DN/e$^-$]')
    ax3.set_title('FPN[%] = '+str("{:2.2f}".format(np.sqrt(np.abs(fitFPN2[0])*100.) ) )+' PRNU[%] = '+str("{:2.2f}".format(PRNU)))

    fig.suptitle(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+' -- Max: '+str("{:2.4f}".format(1/fitgain[0]*90e3))+' DR='+str("{:2.2f}".format(9e4/rn)))
    if savefig=='yes':
        plt.savefig(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+'_curves_dc.png',dpi=600)


    """
    PRINT AND SAVE CALCULATED PARAMETERS
    """
    print('Shape of images',datam.shape)
    print('Gain [DN/e]:',round(calc_gain,3))
    print('Read noise [e]:',round(rn,2))
    print('DR:',round(9e4/rn))
    print('PRNU [%]: ',round(PRNU,2))
    print('FPN [%]:',round(np.sqrt(np.abs(fitFPN2[0])*100.),2))
    print('Linearity [%]',round(linearity,3))
    print('Mean off-set [DN]:',round(offsetmap.mean(dtype=np.float64),2))
    print('STD off-set [DN]:',round(offsetmap.std(dtype=np.float64),2))
    print('Mean dark current [DN/s]:',round(dcmap.mean(dtype=np.float64)*1e3,2))
    print('STD dark current [DN/s] ',round(dcmap.std(dtype=np.float64)*1e3,2))

    #Save data
    datos=pd.Series()
    datos=datos.append(pd.Series([round(calc_gain,3),round(rn,2),\
    round(9e4/rn)],index=['Gain','Read noise','Dynamic range']))
    datos=datos.append(pd.Series([round(PRNU,2)],index=['PRNU']))
    datos=datos.append(pd.Series([round(np.sqrt(np.abs(fitFPN2[0])*100.),2)],\
    index=['FPN']))
    datos=datos.append(pd.Series([round(linearity,3)],index=['Linearity']))
    datos=datos.append(pd.Series([round(offsetmap.mean(dtype=np.float64),2)],\
    index=['Mean off-set']))
    datos=datos.append(pd.Series([round(offsetmap.std(dtype=np.float64),2)],\
    index=['STD off-set']))
    datos=datos.append(pd.Series([round(dcmap.mean(dtype=np.float64)*1e3,2)],\
    index=['Mean dark current']))
    datos=datos.append(pd.Series([round(dcmap.std(dtype=np.float64)*1e3,2)],\
    index=['STD dark current']))
    datos=datos.append(pd.Series([name],index=['name']))
    datos.to_csv(dir+dir_d+name+'_T'+Tset+'_results'+'.csv',header=False)
    return

def curves_darks(dir,dir_d,name,darks,gain,Tset,mode,bl,t_exp,inm,rxa,rya):
    """
    Calculates only dark current curves
    """
    # %%
    datam,datas,spmean,spvar,spiar,datad,sdmean,spdar=estadistica_data(t_exp,inm,darks,rx=rxa,ry=rya)


    # %%
    sx = (rxa[1]-rxa[0])+1
    sy = (rya[1]-rya[0])+1

    cond = t_exp > 19
    t_exp_fit = t_exp[cond]
    fitmean = np.transpose(np.polyfit(t_exp_fit,
        np.transpose(datam[:,:,cond].reshape(sx * sy, len(t_exp_fit))), 1))

    fitmean = np.transpose(np.polyfit(t_exp_fit,
        np.transpose(datam[:,:,cond].reshape(sx * sy, len(t_exp_fit))), 1))
    fitmean = fitmean.reshape(sx, sy, 2)
    fitstd = np.transpose(np.polyfit(t_exp_fit,
        np.transpose(datas[:,:,cond].reshape(sx * sy, len(t_exp_fit))), 1))
    fitstd = fitstd.reshape(sx, sy, 2)

    dcmap = fitmean[:, :, 0]


    # %%
    limit = 500
    which_ones2 = np.where((spmean < limit) )

    fitlinear2 = np.polyfit(t_exp[which_ones2],spmean[which_ones2], 1)
    fitlinear_fns2 = np.poly1d(fitlinear2)
    linearity2  =  (np.max( (spmean[which_ones2] - fitlinear_fns2(t_exp[which_ones2]))) /
                   + np.abs(np.min( (spmean[which_ones2] - fitlinear_fns2(t_exp[which_ones2])))) ) / np.max(spmean[which_ones2])*100

    fitgain2 = np.polyfit(spdar[which_ones2]/2,spmean[which_ones2], 1)
    fitgain_fns2 = np.poly1d(fitgain2)
    calc_gain2 = 1/fitgain2[0] #DN/e
    read_noise2 = fitgain2[1] #DN (read noise)
    rn = np.sqrt(np.abs(read_noise2*calc_gain2)) #electrons

    fitFPN2= np.polyfit(spiar[which_ones2],spmean[which_ones2], 2)
    #fitFPN = np.polyfit(spiar,spmean, 2)
    fitFPN_fns2 = np.poly1d(fitFPN2)
    #print(fitFPN,'gain= ',1/fitFPN[1],'FPN= ',fitFPN[0],'Read noise= ',np.sqrt(fitFPN[2]))

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(131)
    ax1.plot(t_exp,spmean,'ok', ms=4)
    ax1.plot(t_exp,fitlinear_fns2(t_exp),'k', ms=2,color='r')
    ax2 = fig.add_subplot(132)
    ax2.plot(spdar/2,spmean,'ok', ms=4)
    ax2.plot(spdar/2,fitgain_fns2(spdar/2),'k', ms=2,color='r')
    ax3 = fig.add_subplot(133)
    ax3.plot(spiar,spmean,'ok', ms=4)
    ax3.plot(spiar,fitFPN_fns2(spiar),'k', ms=2,color='r')
    ax1.grid(True,which="both",ls="--",color='0.65')
    ax2.grid(True,which="both",ls="--",color='0.65')
    ax3.grid(True,which="both",ls="--",color='0.65')

    ax1.set_xlabel('Exp Time')
    ax1.set_ylabel('Counts [DN]')
    ax2.set_xlabel('Photon Noise [DN$^2$]')
    ax2.set_ylabel('Counts [DN]')
    ax3.set_xlabel('Noise (FPN)')
    ax3.set_ylabel('Counts [DN]')

    xlim1 = [np.floor(np.min(t_exp))*0.9 , np.ceil(np.max(t_exp))*1.1]
    ylim1 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    xlim2 = [np.floor(np.min(spdar/2))*0.9 , np.ceil(np.max(spdar/2))*1.1]
    ylim2 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    xlim3 = [np.floor(np.min(spiar))*0.9 , np.ceil(np.max(spiar))*1.1]
    ylim3 = [np.floor(np.min(spmean))*0.9 , np.ceil(np.max(spmean))*1.1]
    if mode == '4':
        if gain == '7':
            xlim2 = [0,1]
            ylim2 = [100,300]
    elif mode == '1':
        if gain == '0':
            xlim3 = [0,500]

    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)
    ax3.set_xlim(xlim3)
    ax3.set_ylim(ylim3)

    ax1.set_title('Mode='+mode+'; gain='+gain+'; BL='+bl+' T='+Tset)
    ax1.set_title('Linearity= '+str("{:10.4f}".format(linearity2)).strip()+' [%]'+'  R/N='+str("{:2.2f}".format(rn))+' e$^-$')
    ax2.set_title('Off-set='+str("{:2.2f}".format(fitgain2[1]))+' Gain='+str("{:2.4f}".format(1/fitgain2[0]))+' [DN/e$^-$]')

    fig.suptitle(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+' -- Max: '+str("{:2.4f}".format(1/fitgain2[0]*90e3))+' DR='+str("{:2.2f}".format(9e4/rn)))
    if savefig=='yes':
        plt.savefig(dir+dir_d+name+'_g'+gain+'_T'+Tset+'_mod'+mode+'_bl'+bl+'_curves_dc.png',dpi=200)

    return

def search_hotpixels(name,folder,path,conflevel,rx,ry,hsize,vsize,show='no'):
    image = read_raw('./'+folder+'/',name+'_0'+'.raw',Nx=hsize,Ny=vsize)
    image=image[:,:,0]

    #Plot image
    xmin=rx[0]
    xmax=rx[1]
    ymin=ry[0]
    ymax=ry[1]

    rms0=np.std(image[xmin:xmax,ymin:ymax])
    mu0=np.mean(image[xmin:xmax,ymin:ymax])
    max=np.max(image[xmin:xmax,ymin:ymax])
    minim=mu0-conflevel*rms0
    maxim=mu0+conflevel*rms0
    if minim<0:
        minim=0

    #plt.subplot(1,2,2)
    Hot_pos=np.argwhere(image[xmin:xmax,ymin:ymax]>maxim) #Position of hot pixels

    Nhot=np.size(Hot_pos)/2 #Number of hot pixels
    perhot=Nhot/((2048+rx[1]-rx[0])*(2048+ry[1]-ry[0]))*100 #percentage of pixels
    #print('# of hot pixels:',Nhot)
    #print('% of hot pixels:',round(perhot,4))

    plt.subplot(131)
    plt.imshow(image[xmin:xmax,ymin:ymax],vmin=mu0-conflevel*rms0,vmax=mu0+conflevel*rms0,cmap='gray')
    plt.colorbar()
    plt.title('Mean:%g; RMS:%g; Max:%g'%(mu0,rms0,max))


    #Histogram
    plt.subplot(133)
    hst=np.ndarray.flatten(image[xmin:xmax,ymin:ymax])
    n,bins,patches=plt.hist(hst,bins=np.arange(int(minim),int(maxim+1)),density=True)
    plt.xlabel('DN')
    plt.ylabel('Frequency')

    #Gaussian fit
    mu_hst,sigma_hst=norm.fit(hst)
    fitting=norm.pdf(bins,mu_hst,sigma_hst)
    plt.plot(bins,fitting,label='Gaussian fit')



    #Location of hot pixels
    plt.subplot(132)
    zeromat=np.zeros((2048-xmin+xmax,2048-xmin+xmax))
    zeromat[np.where(image[xmin:xmax,ymin:ymax]>maxim)]=1
    plt.imshow(zeromat)
    if show=='yes':
        plt.show()
        plt.close()
    return round(perhot,4)

def linearity(dir,dir_d,name,image,darks,gain,Tset,mode,bl,t_exp,inm,rxa,rya):
    """
    STATISTICS FOR ILLUMINATED FRAMES
    """
    #Correct images from darks
    image_dc = image - darks  #remove all darks

    # # Data analysis
    find_index = np.where(t_exp > 80.)
    index = np.nonzero(t_exp>80)[0][0]*inm

    rx=[0,2048]
    ry=[0,2048]

    datam,datas,spmean,spvar,spiar,datad,sdmean,spdar = estadistica_data(t_exp,\
    inm,image,rx=rxa,ry=rya)

    dd = np.abs(spmean-np.median(spmean))
    index=np.where(dd == np.min(dd))


    """
    Fit of data to obtain gain, read noise, FPN ...
    """
    limit,limit_inf=limit_statistics(mode,gain)
    which_ones = np.where((spmean < limit) & (spmean > limit_inf))
    fitlinear = np.polyfit(t_exp[which_ones],spmean[which_ones], 1)
    fitlinear_fns = np.poly1d(fitlinear)
    linearity=(np.max( (spmean[which_ones] - fitlinear_fns(t_exp[which_ones])))\
    + np.abs(np.min( (spmean[which_ones] - fitlinear_fns(t_exp[which_ones])))))\
     / np.max(spmean[which_ones])*100
    off_set =fitlinear[0]
    return linearity

def save_as_npy(dir,dir_d,image,darks,t_exp,inm,dark_corr=True):
    """
    Save dark-corrected images in a numpy array with format (Nx,Ny,frame,inm)
    """
    data_npy = np.zeros((image.shape[0],image.shape[1],t_exp.size,inm),dtype=np.float32)
    for i in range(t_exp.size):
        for j in range(inm):
            if dark_corr==True:
                data_npy[:,:,i,j]=np.abs(image[:,:,i*inm+j]-darks[:,:,i*inm+j])
            else:
                data_npy[:,:,i,j]=np.abs(image[:,:,i*inm+j])
    if dark_corr==True:
        np.save(dir+dir_d+'data_dark_corrected.npy',data_npy)
    else:
        np.save(dir+dir_d+'data.npy',data_npy)
    return
