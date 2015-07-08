import numpy
import h5py
import sys,os
import math
from matplotlib import pyplot as plt
from mpi4py import MPI
sys.path.append(os.path.abspath('../myFunctions'))
import handyCythonLib



def qDistFrames(filename, dmax, dmin, dbin,timesep=100,fourierSpace=True):
    global rank
    global commSize
    
    f=h5py.File(filename+"h5","r")
    qfile=h5py.File(filename+" qDistFrames.h5","w")

    shiftlist=numpy.arange(1,timesep+1)
    local_index   = [numpy.array_split(numpy.arange(len(shiftlist)), commSize)][rank]
    binss=numpy.arange(dmin, dmax+dbin, dbin)
    for key in f.keys():
        
        t,row,col = f[key].shape
        qMax=min(int(row/2),int(col/2))+1
        stack=numpy.empty((qMax,timesep,len(binss)-1),dtype='float')
        for index in local_index:
            shift=shiftlist[index]
            tempstack=numpy.empty((t-shift,qMax))
            for i in range(t-shift):
                if fourierSpace=True:
                    img=f[key][i]
                    img1=f[key][i+shift]
                else:
                    img=numpy.abs(numpy.fft.fftshift(numpy.fft.fftn(f[key][i])))
                    img1=numpy.abs(numpy.fft.fftshift(numpy.fft.fftn(f[key][i+shift])))
                tempstack[i]=handyCythonLib.qNorm(img,img1)
            for q in range(qMax):
                a,b=numpy.histogram(tempstack[:,q],bins=binss)
                stack[q,index]=a
        qfile.create_dataset(key,data=stack,compression='gzip', compression_opts=9)

    f.close()
    qfile.close()





