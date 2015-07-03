import h5py
import sys,os
from matplotlib import pyplot as plt
from mpi4py import MPI
import handyCythonLib
import scipy
import numpy
comm = MPI.COMM_WORLD
commSize=comm.Get_size()
rank= comm.Get_rank()

inputDir='/mnt/cbis/home/barzhas/data/3drec'
imageDir1='/mnt/cbis/images/barzhas/3drec/particle1'
imageDir2='/mnt/cbis/images/barzhas/3drec/particle2'
f=h5py.File(inputDir+"/fcutouts.h5","r")
qfile=h5py.File(inputDir+"/EucDist","r")
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
        if i%500==0:
            print l[i]
        img=p1[l[i]].value
        img1=p1[l[i+shift]].value
        stack[i]=handyCythonLib.qNorm(img,img1)
    numpy.save(inputDir+"/particle1-"+str(shift)+".npy",stack)
    tack=numpy.empty((len(m)-shift,63),dtype='float')
    for i in range(len(m)-shift):
        if i%500==0:
            print m[i]
        img=p2[m[i]].value
        img1=p2[m[i+shift]].value
        tack[i]=handyCythonLib.qNorm(img,img1)
     
    numpy.save(inputDir+"/particle2-"+str(shift)+".npy",tack)
    print "Rank %d done with generating stack for shift  %s "%(rank, shift)
f.close()

if rank==0:
    for shift in shiftlist:
        print "saving shift",shift
        stack=numpy.load(inputDir+"/particle1-"+str(shift)+".npy")
        qfile.create_dataset("particle1/"+str(shift), data=stack , compression="gzip", compression_opts=9)
        tack=numpy.load(inputDir+"/particle2-"+str(shift)+".npy")
        qfile.create_dataset("particle2/"+str(shift), data=tack , compression="gzip", compression_opts=9) 
        os.remove(inputDir+"/particle1-"+str(shift)+".npy")
        os.remove(inputDir+"/particle2-"+str(shift)+".npy") 


local_skewness1=numpy.zeros((63,s),dtype='float')
total_skewness1=numpy.zeros_like(local_skewness1)
local_skewness2=numpy.zeros((63,s),dtype='float')
total_skewness2=numpy.zeros_like(local_skewness1)

index= [len(rng) for rng in numpy.array_split(numpy.arange(s), commSize)]
indexcumsum=numpy.cumsum(index)
indexrank=range(indexcumsum[rank]-index[rank],indexcumsum[rank])
for i in indexrank:
    stack=qfile["particle1/"+str(shiftlist[i])].value
    tack=qfile["particle2/"+str(shiftlist[i])].value
    for q in range(63):
        qDist1=stack[:,q]
        local_skewness1[q,i]=scipy.stats.skew(qDist1)
        qDist2=tack[:,q]
        local_skewness2[q,i]=scipy.stats.skew(qDist2)

comm.Reduce(local_skewness1,total_skewness1, op=MPI.SUM)
comm.Reduce(local_skewness2,total_skewness2, op=MPI.SUM)

if rank==0:
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('time')
    ax.set_ylabel('skewness')
    for q in range(63):
        print q
        ax.plot(total_skewness1[q])
        ax.set_xlabel('time')
        ax.set_ylabel('skewness')
        fig.savefig(imageDir1+'/'+str(q)+'.png')
        ax.cla()
        ax.set_xlabel('time')
        ax.set_ylabel('skewness')
        ax.plot(total_skewness2[q])
        fig.savefig(imageDir2+'/'+str(q)+'.png')
        ax.cla()




