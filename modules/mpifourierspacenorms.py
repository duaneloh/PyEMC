import numpy
import h5py
import sys,os
from matplotlib import pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
commSize=comm.Get_size()
rank= comm.Get_rank()

inputDir='/mnt/cbis/home/barzhas/data/3drec'


f=h5py.File(inputDir+"/fcutouts.h5","r")
qfile=h5py.File(inputDir+"/EucDist","w")
p1=f["particle1"]
p2=f["particle2"]
l=p1.keys()
m=p2.keys()
shiftlist=numpy.arange(1,101)
s=len(shiftlist)

print "Rank %d done with reading data"%(rank)

localshift   = numpy.array_split(shiftlist, commSize)[rank]

for shift in localshift:

    stack=numpy.empty((len(l)-shift,63),dtype='float')
    for i in range(len(l)-shift):
        print l[i]
        img=p1[l[i]].value
        img1=p1[l[i+shift]].value
        stack[i]=handyCythonLib.qNorm(img,img1)
    
    qfile.create_dataset("particle1/"+str(shift), data=stack , compression="gzip", compression_opts=9)
    tack=numpy.empty((len(m)-shift,63),dtype='float')
    for i in range(len(m)-shift):
        print m[i]
        img=p2[m[i]].value
        img1=p2[m[i+shift]].value
        tack[i]=handyCythonLib.qNorm(img,img1)
    qfile.create_dataset("particle2/"+str(shift), data=tack , compression="gzip", compression_opts=9)
    print "Rank %d done with generating stack for shift  %s "%(rank, shift)

f.close()
qfile.close()

