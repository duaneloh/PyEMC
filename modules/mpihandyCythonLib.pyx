import numpy
import time
cimport numpy
from scipy import ndimage
import skimage
from skimage import measure
import matplotlib.pyplot as plt
#cimport cv
import cv2

DTYPE = numpy.double
ctypedef numpy.double_t DTYPE_t

cdef extern from "math.h":
    float sqrt(float x)
    float floor(float x)
    float atan(float x)
    float fabs(float x)
    float round(float x)

cdef extern from "complex.h":
    double cabs(complex x)

cimport cython
from cpython cimport bool
from cython.parallel import parallel, prange
@cython.boundscheck(False)





def RandomVolume(int R, bool CreateSliceOnly=True ):
    cdef numpy.ndarray[numpy.int64_t, ndim=3] xv=numpy.mgrid[-R:R+1,-R:R+1,-R:R+1][0]
    cdef numpy.ndarray[numpy.int64_t, ndim=3] yv=numpy.mgrid[-R:R+1,-R:R+1,-R:R+1][1]
    cdef numpy.ndarray[numpy.int64_t, ndim=3] zv=numpy.mgrid[-R:R+1,-R:R+1,-R:R+1][2]
    cdef numpy.ndarray[numpy.float_t, ndim=3] r=numpy.sqrt(xv**2+yv**2+zv**2)
    cdef numpy.ndarray[numpy.uint8_t, ndim=3] mask= r<=R
    
    cdef numpy.ndarray[numpy.float_t, ndim=3] W=numpy.zeros((2*R+1,2*R+1,2*R+1),dtype='float')
    
    cdef numpy.ndarray[numpy.int64_t, ndim=2] xv2=xv[:,:,R]
    cdef numpy.ndarray[numpy.int64_t, ndim=2] yv2=yv[:,:,R]
    cdef numpy.ndarray[numpy.float_t, ndim=2] r2=numpy.sqrt(xv**2+yv**2)
    cdef numpy.ndarray[numpy.int_t, ndim=2] mask2=r2<=R
    cdef numpy.ndarray[numpy.float_t, ndim=2]randomslice=numpy.zeros((2*R+1,2*R+1),dtype='float')
    
    if CreateSliceOnly==False:
        W=numpy.random.random_integers(0,10000,(2*R+1,2*R+1,2*R+1))
        W=W*mask
    else:
        randomslice=(numpy.random.random_integers(0,10000,(2*R+1,2*R+1))*mask).astype(W.dtype)
        W[:,:,R]=randomslice
    return W

def RefTom(int R):
    cdef numpy.ndarray[numpy.int_t, ndim=2] xv=numpy.mgrid[-R:R+1,-R:R+1][0]
    cdef numpy.ndarray[numpy.int_t, ndim=2] yv=numpy.mgrid[-R:R+1,-R:R+1][1]
    cdef numpy.ndarray[numpy.int_t, ndim=2] r=numpy.sqrt(xv**2+yv**2)
    cdef numpy.ndarray[numpy.int_t, ndim=1] a=numpy.where(r<=R)[0]-R
    cdef numpy.ndarray[numpy.int_t, ndim=1] b=numpy.where(r<=R)[1]-R
    cdef numpy.ndarray[numpy.int_t, ndim=1] c=numpy.zeros(a.shape[0],dtype='int')
    cdef numpy.ndarray[numpy.int_t, ndim=2] k=numpy.vstack((a,b,c)).T
    return k

def MaxTom(int Rmin, int Rmax):
    cdef numpy.ndarray[numpy.int_t, ndim=2] xv=numpy.mgrid[-Rmax:Rmax+1,-Rmax:Rmax+1][0]
    cdef numpy.ndarray[numpy.int_t, ndim=2] yv=numpy.mgrid[-Rmax:Rmax+1,-Rmax:Rmax+1][1]
    cdef numpy.ndarray[numpy.int_t, ndim=2] r=numpy.sqrt(xv**2+yv**2)
    cdef numpy.ndarray[numpy.int_t, ndim=2] boolka=numpy.logical_and(r>Rmin,r<Rmax)
    cdef numpy.ndarray[numpy.int_t, ndim=1] a=numpy.where(boolka)[0]-Rmax
    cdef numpy.ndarray[numpy.int_t, ndim=1] b=numpy.where(boolka)[1]-Rmax
    cdef numpy.ndarray[numpy.int_t, ndim=2] k=numpy.vstack((a,b)).T
    return k

def CalcRotMat(numpy.ndarray[numpy.float_t, ndim=1] q):
    assert q.dtype==numpy.float
    cdef numpy.ndarray[numpy.float_t, ndim=2] rotmat=numpy.zeros((3,3),dtype='float')
    cdef float q1=q[0]
    cdef float q2=q[1]
    cdef float q3=q[2]
    cdef float q4=q[3]
    
    rotmat[0,0]=q1**2+q2**2-q3**2-q4**2
    rotmat[0,1]=2*q2*q3-2*q1*q4
    rotmat[0,2]=2*q2*q4+2*q1*q3
    rotmat[1,0]=2*q2*q3+2*q1*q4
    rotmat[1,1]=q1**2-q2**2+q3**2-q4**2
    rotmat[1,2]=2*q3*q4-2*q1*q2
    rotmat[2,0]=2*q2*q4-2*q1*q3
    rotmat[2,1]=2*q3*q4+2*q1*q2
    rotmat[2,2]=q1**2-q2**2-q3**2+q4**2

    return rotmat

def QuatList(int number):
    cdef numpy.ndarray[numpy.float_t, ndim =2] quat=numpy.zeros((number,4), dtype='float')
    cdef float q=1.1
    cdef float q1,q2,q3,q4
    cdef numpy.ndarray[numpy.float_t, ndim=1] randnum= numpy.zeros(4, dtype='float')
    cdef int i
    
    for i in range(number):
        q=1.1
        while q>1:
            randnum=-1+2*numpy.random.random_sample(4)
            q1=randnum[0]
            q2=randnum[1]
            q3=randnum[2]
            q4=randnum[3]
            q=numpy.sqrt(q1**2+q2**2+q3**2+q4**2)
        q1=q1/q
        q2=q2/q
        q3=q3/q
        q4=q4/q
        quat[i] =numpy.asarray([q1,q2,q3,q4])
    return quat



def expand(numpy.ndarray[numpy.float_t, ndim=3] W, numpy.ndarray[numpy.float_t, ndim=2]quat,numpy.ndarray[numpy.int_t, ndim=2]k0):

    assert W.dtype==numpy.float and k0.dtype==numpy.int and quat.dtype==numpy.float
    cdef int R=(W.shape[0]-1)/2
    cdef int N      = quat.shape[0]
    cdef numpy.ndarray[numpy.float_t, ndim=2]       stack=numpy.zeros((quat.shape[0],k0.shape[0]), dtype='float')
    cdef numpy.ndarray[numpy.float_t,ndim=2] rotmat=numpy.zeros((3,3), dtype='float')
    cdef int n,j
    cdef float rx,ry,rz,dx1,dx,dy1,dy,dz1,dz
    cdef float v000,v100,v010,v110,v001,v101,v011,v111
    cdef int ix,iy,iz,x,y,z
    for n from 0 <= n < N:
        rotmat=CalcRotMat(quat[n])
        for j from 0 <= j < k0.shape[0]:
            ix=k0[j,0]
            iy=k0[j,1]
            rx=rotmat[0,0]*ix+rotmat[0,1]*iy
            ry=rotmat[1,0]*ix+rotmat[1,1]*iy
            rz=rotmat[2,0]*ix+rotmat[2,1]*iy
            if sqrt(rx*rx+ry*ry+rz*rz)>0.99*R:
                    continue
            rx=rx+R
            ry=ry+R
            rz=rz+R
            x=int(rx)
            y=int(ry)
            z=int(rz)
            dx1=rx-x
            dy1=ry-y
            dz1=rz-z
            dx=x+1.0-rx
            dy=y+1.0-ry
            dz=z+1.0-rz
            v000=W[x,y,z];v100=W[x+1,y,z];v010=W[x,y+1,z];v110=W[x+1,y+1,z]
            v001=W[x,y,z+1];v101=W[x+1,y,z+1];v011=W[x,y+1,z+1];v111=W[x+1,y+1,z+1]
            stack[n,j]=v000*dx*dy*dz+v100*dx1*dy*dz+v010*dx*dy1*dz+v110*dx1*dy1*dz+v001*dx*dy*dz1+v101*dx1*dy*dz1+v011*dx*dy1*dz1+v111*dx1*dy1*dz1

    return stack

def maximize(numpy.ndarray[numpy.float_t,ndim=2] projections,numpy.ndarray[numpy.float_t, ndim=2] tomograms,numpy.ndarray[numpy.int_t, ndim=2] k0, int Rmin, int Rmax):
    cdef int pix_index,quat_index,frame_index
    cdef float  distance
    cdef numpy.ndarray[numpy.float_t, ndim=2] distances=numpy.empty((tomograms.shape[0],projections.shape[0]),dtype='float')
    cdef int ix,iy
#    cdef numpy.ndarray[numpy.int_t, ndim=2] k=MaxTom(Rmin,Rmax)
    for quat_index from 0 <= quat_index < tomograms.shape[0]:
        for frame_index from 0 <= frame_index < projections.shape[0]:
            distance=0
            for pix_index from 0 <= i < k0.shape[0]:
                ix=k0[pix_index,0]
                iy=k0[pix_index,1]
                if sqrt(ix*ix+iy*iy)>0.99*Rmax or sqrt(ix*ix+iy*iy)<0.99*Rmin:
                    continue
                distance+=(tomograms[quat_index,pix_index]-projections[frame_index,pix_index])**2
            distances[quat_index,frame_index]=distance
    
    return distances





def compress(numpy.ndarray[numpy.float_t,ndim=2] slices,numpy.ndarray[numpy.float_t, ndim=2] quat,numpy.ndarray[numpy.int_t, ndim=2] k0):
    assert slices.dtype==numpy.float and k0.dtype==numpy.int and quat.dtype==numpy.float
        
    cdef int N=quat.shape[0]
    cdef int R=k0[-1,0]
    cdef numpy.ndarray[numpy.float_t,ndim=3] weights=numpy.zeros((2*R+1,2*R+1,2*R+1),dtype='float')
    cdef numpy.ndarray[numpy.float_t, ndim=3] W=numpy.zeros((2*R+1,2*R+1,2*R+1),dtype='float')
    cdef int x,y,z,ix,iy
    cdef numpy.ndarray[numpy.float_t,ndim=2] rotmat=numpy.zeros((3,3), dtype='float')
    cdef float point,rx,ry,rz,dx,dy,dz,dx1,dy1,dz1
    cdef int n,j,i,k
    cdef float w1,w2,w3,w4,w5,w6,w7,w8
    for n from 0 <= n < N:
        rotmat=CalcRotMat(quat[n])
        for j from 0 <= j < k0.shape[0]:
            ix=k0[j,0]
            iy=k0[j,1]
            rx=rotmat[0,0]*ix+rotmat[0,1]*iy
            ry=rotmat[1,0]*ix+rotmat[1,1]*iy
            rz=rotmat[2,0]*ix+rotmat[2,1]*iy

            if sqrt(rx*rx+ry*ry+rz*rz)>0.99*R:
                    continue
            
            rx=rx+R
            ry=ry+R
            rz=rz+R
            x=int(rx)
            y=int(ry)
            z=int(rz)
            point=slices[n,j]
            dx1=rx-x
            dy1=ry-y
            dz1=rz-z
            dx=x+1.0-rx
            dy=y+1.0-ry
            dz=z+1.0-rz
            w1=dx*dy*dz
            w2=dx1*dy*dz
            w3=dx*dy1*dz
            w4=dx1*dy1*dz
            w5=dx*dy*dz1
            w6=dx1*dy*dz1
            w7=dx*dy1*dz1
            w8=dx1*dy1*dz1
            
            weights[x,y,z]+=w1
            weights[x+1,y,z]+=w2
            weights[x,y+1,z]+=w3
            weights[x+1,y+1,z]+=w4
            weights[x,y,z+1]+=w5
            weights[x+1,y,z+1]+=w6
            weights[x,y+1,z+1]+=w7
            weights[x+1,y+1,z+1]+=w8
            W[x,y,z]+=point*w1
            W[x+1,y,z]+=point*w2
            W[x,y+1,z]+=point*w3
            W[x+1,y+1,z]+=point*w4
            W[x,y,z+1]+=point*w5
            W[x+1,y,z+1]+=point*w6
            W[x,y+1,z+1]+=point*w7
            W[x+1,y+1,z+1]+=point*w8

    return W, weights

def array_division(numpy.ndarray[numpy.float_t,ndim=3] W, numpy.ndarray[numpy.float_t,ndim=3] weights):

    cdef int i,j,k
    cdef numpy.ndarray[numpy.float_t,ndim=3] result= numpy.zeros_like(weights)
    
    for i from 0<= i < weights.shape[0]:
        for j from 0<= j < weights.shape[1]:
            for k from 0<= k < weights.shape[2]:
                if weights[i,j,k]>0:
                    result[i,j,k]=W[i,j,k]/weights[i,j,k]
    return result


def angAve(numpy.ndarray[numpy.float_t, ndim=3] W1, numpy.ndarray[numpy.int_t, ndim=3] r3):
    assert W1.dtype==numpy.float and r3.dtype==numpy.int
    
    cdef int i,j,k
    cdef int rlim=r3.max()+1
    cdef numpy.ndarray[numpy.float_t, ndim=1] sumint=numpy.zeros(rlim, dtype='float')
    cdef numpy.ndarray[numpy.int_t,ndim=1] count=numpy.zeros(rlim, dtype='int')
    cdef int a= W1.shape[0]
    cdef int b= W1.shape[1]
    cdef int c= W1.shape[2]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                sumint[r3[i,j,k]]+=W1[i,j,k]*W1[i,j,k]
                count[r3[i,j,k]]+=1
    for i in range(rlim):
        if count[i]>0:
            sumint[i]=sqrt(sumint[i]/count[i])
    return sumint





