import sys, os
import cv2
import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import os
import math
import skimage
from skimage import measure
from skimage import filter, morphology
from itertools import combinations
from scipy import ndimage
#from mahotas.polygon import fill_convexhull
import time
import resource
from mpi4py import MPI
#sys.path.append(os.path.abspath('../myFunctions'))
import handyCythonLib
#import unhandyCythonLib

comm= MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

def RefTom(R):
    xv,yv=numpy.mgrid[-R:R+1,-R:R+1]
    r=numpy.sqrt(xv**2+yv**2)
    a,b=numpy.where(r<=R)
    a=a-R
    b=b-R
    c=numpy.zeros(a.shape,a.dtype)
    k=numpy.vstack((a,b,c)).T
    return k

def RandomVolume(R, CreateSliceOnly=True ):
    xv,yv,zv=numpy.mgrid[-R:R+1,-R:R+1,-R:R+1]

    if CreateSliceOnly==False:
        W=numpy.random.random_integers(0,10000,(2*R+1,2*R+1,2*R+1))
        r=numpy.sqrt(xv**2+yv**2+zv**2)
        mask= r<=R
        W=W*mask
    else:
        W=numpy.zeros((2*R+1,2*R+1,2*R+1),dtype='uint32')
        xv=xv[:,:,R]
        yv=yv[:,:,R]
        r=numpy.sqrt(xv**2+yv**2)
        mask= r<=R
        randomslice=(numpy.random.random_integers(0,10000,(2*R+1,2*R+1))*mask).astype(W.dtype)
        W[:,:,R]=randomslice
    return W

def readlist(path,i):
    filename = 'quaternion'+str(i)+'.dat'
    filename = os.path.join(path, filename)
    with open(filename,'r') as f:
        [w]=[int(x) for x in f.readline().split()]
        array=numpy.asarray([numpy.asarray([numpy.double(x) for x in line.split()]) for line in f])
    return array,w

def memorycons():
    consumption=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    return consumption

def mkdir(dirName):
    if (os.path.exists(dirName) == False):
        os.makedirs(dirName)
def showcuts(rbk,ce,title):
    fig=plt.figure()
    ax1=fig.add_subplot(131)
    ax1.imshow(rbk[ce,:,:])
    ax2=fig.add_subplot(132)
    ax2.imshow(rbk[:,ce,;])
    ax3=fig.add_subplot(133)
    ax3.imshow(rbk[:,:,ce])
    fig.savefig(title,bbox='tight')
    plt.close()

def plotcuts(yf, label, dropzero=False,ylim=False, xlim=False)
    n=yf.shape[0]
    rm=yf.shape[1]
    t=int(rm/numpy.sqrt(3))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(n):
        if dropzero==False:
            index=i
        else:
            index=i+1
        ax.plot(yf[i], label='quat '+str(index))
        if ylim==True:
            ax.set_ylim([0,1])
        if xlim==True:
            ax.set_xlim([0,t])
    ax.legend()
    fig.savefig(saving+label+'.png')
    plt.close()



inputDir = '/Volumes/Untitled/3D-reconstruction/29_20150504_141003_2015-05-05-14-05/'
path= '/mnt/cbis/home/barzhas/working_dir/PyEMC/quaternions/'
saving='/mnt/cbis/home/barzhas/working_dir/PyEMC/testresults1/'
L=375
ce=(L-1)/2
R=48
sigma=R/2



Ro=numpy.zeros((L,L,L),dtype='double')
rbk=numpy.zeros((L,L,L),dtype='double')
local_n=numpy.zeros(1)

xv,yv,zv=numpy.mgrid[-ce:ce+1,-ce:ce+1,-ce:ce+1]
kv=numpy.double(numpy.sqrt(xv**2+yv**2+zv**2))
r3=numpy.round(kv).astype('int')
kradius=numpy.arange(r3.max()+1)

k0=RefTom(ce)

if rank ==0:
    W = RandomVolume(R,CreateSliceOnly=False)
    Ro[ce-R:ce+R+1,ce-R:ce+R+1,ce-R:ce+R+1]=W
    gau= numpy.exp(-kv*kv/(2*sigma**2))
    Rok=numpy.fft.fftshift(numpy.fft.fftn(Ro))
    Rbk=Rok*gau
    rbk=numpy.double(numpy.abs(Rbk))
    Rbr=numpy.abs(numpy.fft.ifftn(Rbk))
    quatx,n=readlist(path,10)
    local_n=n/size


comm.Bcast(Ro, root=0)
comm.Bcast(rbk, root=0)
comm.Bcast(local_n,root=0)

local_quat=numpy.zeros((local_n,5),dtype='double')

comm.Scatter(quatx,local_quat,root=0)

print ('process '+str(rank)+'has quat of shape '+ str(local_quat.shape))


#slices=handyCythonLib.expand(rbk,quat,k0)
#rbk1=handyCythonLib.compress(slices,quat,k0)



