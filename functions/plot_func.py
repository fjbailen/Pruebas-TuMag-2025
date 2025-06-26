from matplotlib import pyplot as pt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
pt.rcParams['animation.ffmpeg_path']='C:/FFmpeg/bin/ffmpeg'

def im2(I,Q,title=['','','',''],show='no'):
    """
    2 subplots in one.
        I,Q: arrays to be plotted
        title: list with titles for each subplot
        show: show plot?
    """
    colormap='gray'
    pt.subplot(1,2,1)
    pt.imshow(I,cmap=colormap)
    pt.colorbar()
    pt.title(title[0])
    pt.subplot(1,2,2)
    pt.imshow(Q,cmap=colormap)
    pt.colorbar()
    pt.title(title[1])
    if show=='yes':
        pt.show()

def im4(I,Q,U,V,title=['','','',''],show='no'):
    """
    4 subplots in one.
        I,Q,U,V: arrays to be plotted
        title: list with titles for each subplot
        show: show plot?
    """
    pt.subplot(2,2,1)
    pt.imshow(I)
    pt.colorbar()
    pt.title(title[0])
    pt.subplot(2,2,2)
    pt.imshow(Q)
    pt.colorbar()
    pt.title(title[1])
    pt.subplot(2,2,3)
    pt.imshow(U)
    pt.colorbar()
    pt.title(title[2])
    pt.subplot(2,2,4)
    pt.imshow(V)
    pt.colorbar()
    pt.title(title[3])
    if show=='yes':
        pt.show()

def movie(image3D,filename,resol=720,axis=2,fps=15,cbar='no',clims='no',vmin='no',vmax='no'):
    """
    Creates a movie from a 3D image
    """
    metadata = dict(title='Movie', artist='FJBM',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    n=image3D.shape[axis]
    fig,ax=pt.subplots()
    def animate(i):
        if axis==2:
            if vmin!='no':
                pt.imshow(image3D[:,:,i],cmap='gray',vmin=vmin,vmax=vmax)
            else:
                pt.imshow(image3D[:,:,i],cmap='gray')
        elif axis==1:
            pt.imshow(image3D[:,i,:],cmap='gray')
        elif axis==0:
            pt.imshow(image3D[i,:,:],cmap='gray')
        if cbar=='yes':
            pt.colorbar()
    ani = FuncAnimation(fig, animate, frames=n, repeat=False)
    print('./'+filename)
    ani.save('./'+filename, writer=writer)

def movie2(im1,im2,filename,resol=720,axis=2,fps=15,title=['',''],cmap='gray'):
    """
    Creates a movie from two 3D images
    """
    metadata = dict(title='Movie', artist='FJBM',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    n=im1.shape[axis]
    fig=pt.figure()
    min1=np.min(im1[:,:,:])
    max1=np.max(im1[:,:,:])
    min2=np.min(im2[:,:,:])
    max2=np.max(im2[:,:,:])
    print(min1,max1,min2,max2)
    #To use colorbars
    if axis==2:
        ax1=fig.add_subplot(1,2,1)
        pt.imshow(im1[:,:,0],cmap=cmap,vmin=min1,vmax=max1)
        pt.colorbar(orientation='horizontal')
        ax2=fig.add_subplot(1,2,2)
        pt.imshow(im2[:,:,0],cmap=cmap,vmin=min2,vmax=max2)
        pt.colorbar(orientation='horizontal')
        pt.clim(min2,max2)
    #Refresh frames
    def animate(i):
        if axis==2:
            ax1=fig.add_subplot(1,2,1)
            pt.imshow(im1[:,:,i],cmap=cmap,vmin=min1,vmax=max1)
            pt.title(title[0])
            ax2=fig.add_subplot(1,2,2)
            pt.imshow(im2[:,:,i],cmap=cmap,vmin=min2,vmax=max2)
            pt.title(title[1])
        elif axis==1:
            pt.subplot(2,1,1)
            pt.imshow(im1[:,i,:],cmap=cmap,vmin=min1,vmax=max1)
        elif axis==0:
            pt,subplot(2,1,1)
            pt.imshow(im1[i,:,:],cmap=cmap,vmin=min1,vmax=max1)
    ani = FuncAnimation(fig, animate, frames=n, repeat=False)
    ani.save('./'+filename, writer=writer)
    pt.close()

def plot4(x,I,Q,U,V,title=['','','',''],show='no',xlabel='',ylabel=['','','','']):
    """
    4 subplots in one.
        I,Q,U,V: 1D vectors
        title: list with titles for each subplot
        show: show plot?
    """
    pt.subplot(2,2,1)
    pt.plot(x,I)
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[0])
    pt.title(title[0])
    pt.subplot(2,2,2)
    pt.plot(x,Q)
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[1])
    pt.title(title[1])
    pt.subplot(2,2,3)
    pt.plot(x,U)
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[2])
    pt.title(title[2])
    pt.subplot(2,2,4)
    pt.plot(x,V)
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[3])
    pt.title(title[3])
    if show=='yes':
        pt.show()

def twoplot4(x,I1,I2,Q1,Q2,U1,U2,V1,V2,\
title=['','','',''],show='no',xlabel='',ylabel=['','','',''],figlabel=['','']):
    """
    4 subplots in one with two curves each one.
        I1,I2,Q1,Q2,etc: 1D vectors
        title: list with titles for each subplot
        show: show plot?
    """
    pt.subplot(2,2,1)
    pt.plot(x,I1,label=figlabel[0])
    pt.plot(x,I2,label=figlabel[1])
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[0])
    pt.legend()
    pt.title(title[0])

    pt.subplot(2,2,2)
    pt.plot(x,Q1,label=figlabel[0])
    pt.plot(x,Q2,label=figlabel[1])
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[1])
    pt.title(title[1])
    pt.legend()

    pt.subplot(2,2,3)
    pt.plot(x,U1,label=figlabel[0])
    pt.plot(x,U2,label=figlabel[1])
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[2])
    pt.title(title[2])
    pt.legend()

    pt.subplot(2,2,4)
    pt.plot(x,V1,label=figlabel[0])
    pt.plot(x,V2,label=figlabel[1])
    pt.xlabel(xlabel)
    pt.ylabel(ylabel[3])
    pt.title(title[3])
    pt.legend()
    if show=='yes':
        pt.show()
