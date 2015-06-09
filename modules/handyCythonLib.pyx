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
	double sqrt(double x)
	double floor(double x)
	double atan(double x)
	double fabs(double x)
	double round(double x)

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
	cdef numpy.ndarray[numpy.double_t, ndim=3] r=numpy.sqrt(xv**2+yv**2+zv**2)
	cdef numpy.ndarray[numpy.uint8_t, ndim=3] mask= r<=R
	
	cdef numpy.ndarray[numpy.double_t, ndim=3] W=numpy.zeros((2*R+1,2*R+1,2*R+1),dtype='double')
	
	cdef numpy.ndarray[numpy.int64_t, ndim=2] xv2=xv[:,:,R]
	cdef numpy.ndarray[numpy.int64_t, ndim=2] yv2=yv[:,:,R]
	cdef numpy.ndarray[numpy.double_t, ndim=2] r2=numpy.sqrt(xv**2+yv**2)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] mask2=r2<=R
	cdef numpy.ndarray[numpy.double_t, ndim=2]randomslice=numpy.zeros((2*R+1,2*R+1),dtype='double')
	
	if CreateSliceOnly==False:
		W=numpy.random.random_integers(0,10000,(2*R+1,2*R+1,2*R+1))
		W=W*mask
	else:
		randomslice=(numpy.random.random_integers(0,10000,(2*R+1,2*R+1))*mask).astype(W.dtype)
		W[:,:,R]=randomslice
	return W

def RefTom(int R):
	cdef numpy.ndarray[numpy.int64_t, ndim=2] xv=numpy.mgrid[-R:R+1,-R:R+1][0]
	cdef numpy.ndarray[numpy.int64_t, ndim=2] yv=numpy.mgrid[-R:R+1,-R:R+1][1]
	cdef numpy.ndarray[numpy.double_t, ndim=2] r=numpy.sqrt(xv**2+yv**2)
	cdef numpy.ndarray[numpy.double_t, ndim=1] a=numpy.where(r<=R)[0]-R
	cdef numpy.ndarray[numpy.double_t, ndim=1] b=numpy.where(r<=R)[1]-R
	cdef numpy.ndarray[numpy.double_t, ndim=1] c=numpy.zeros(a.shape[0],dtype='double')
	cdef numpy.ndarray[numpy.double_t, ndim=2] k=numpy.vstack((a,b,c)).T
	return k

def CalcRotMat(numpy.ndarray[numpy.double_t, ndim=1] q):
	assert q.dtype==numpy.double
	cdef numpy.ndarray[numpy.double_t, ndim=2] rotmat=numpy.zeros((3,3),dtype='double')
	cdef double q1=q[0]
	cdef double q2=q[1]
	cdef double q3=q[2]
	cdef double q4=q[3]
	
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
	cdef numpy.ndarray[numpy.double_t, ndim =2] quat=numpy.zeros((number,4), dtype='double')
	cdef double q=1.1
	cdef double q1,q2,q3,q4
	cdef numpy.ndarray[numpy.double_t, ndim=1] randnum= numpy.zeros(4, dtype='double')
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

def ExtractSliceValues(numpy.ndarray[numpy.double_t, ndim=3] W,numpy.ndarray[numpy.double_t, ndim=1] q, numpy.ndarray[numpy.double_t, ndim=2] k0):
	assert W.dtype==numpy.double and q.dtype==numpy.double and k0.dtype==numpy.double
	cdef int R=int((W.shape[0]-1)/2)
	cdef numpy.ndarray[numpy.double_t, ndim=1] i
	cdef int j
	cdef numpy.ndarray[numpy.double_t, ndim=2] 	slice=numpy.zeros((2*R+1,2*R+1),dtype='double')
	cdef numpy.ndarray[numpy.double_t, ndim=2] k=numpy.zeros(k0.shape,k0.dtype)
	cdef numpy.ndarray[numpy.double_t, ndim=2]	rotmat=CalcRotMat(q)
	cdef int x,y,z
	cdef double wx1, wx2, wy1,wx3,wx4,wy2
	
	for j in range(k0.shape[0]):
	
		k[j]=numpy.dot(rotmat,k0[j])
		i=k[j]
		i=i+R
		x=i.astype(int)[0]
		y=i.astype(int)[1]
		z=i.astype(int)[2]
		
		wx1=W[x,y,z]*(x+1-i[0])+W[x+1,y,z]*(i[0]-x)
		wx2=W[x,y+1,z]*(x+1-i[0])+W[x+1,y+1,z]*(i[0]-x)
		wy1=wx1*(y+1-i[1])+wx2*(i[1]-y)
		wx3=W[x,y,z+1]*(x+1-i[0])+W[x+1,y,z+1]*(i[0]-x)
		wx4=W[x,y+1,z+1]*(x+1-i[0])+W[x+1,y+1,z+1]*(i[0]-x)
		wy2=wx3*(y+1-i[1])+wx4*(i[1]-y)
		slice[R+k0[j,0],R+k0[j,1]]=wy1*(z+1-i[2])+wy2*(i[2]-z)

	return slice


def expand(numpy.ndarray[numpy.double_t, ndim=3] Volume,numpy.ndarray[numpy.double_t, ndim=2]k0, numpy.ndarray[numpy.double_t, ndim=2]quat):

	assert Volume.dtype==numpy.double and k0.dtype==numpy.double and quat.dtype==numpy.double
	cdef int len0=Volume.shape[0]
	cdef int len1=Volume.shape[1]
	cdef int len2=Volume.shape[2]
	cdef int lena= quat.shape[0]
	cdef numpy.ndarray[numpy.double_t, ndim=3]	stack=numpy.zeros((quat.shape[0],Volume.shape[1],Volume.shape[2]), dtype='double')
	cdef  numpy.ndarray[numpy.double_t, ndim=1] q
	cdef bool condition1= len0==len1
	cdef bool condition2= len0==len2
	cdef int i
	if (condition1 or condition2)==False:
		print "Error! Volume must have cubic shape"

	for i in range(lena):
		q=quat[i]
		stack[i]=ExtractSliceValues(Volume,q,k0)

	return stack


def compress(numpy.ndarray[numpy.double_t,ndim=3] slices,numpy.ndarray[numpy.double_t, ndim=2] quat,numpy.ndarray[numpy.double_t, ndim=2] k0):
	assert slices.dtype==numpy.double and k0.dtype==numpy.double and quat.dtype==numpy.double
	
	cdef int N=slices.shape[0]
	cdef numpy.ndarray[numpy.double_t,ndim=3] weights=numpy.zeros((slices.shape[1],slices.shape[1],slices.shape[1]),dtype='double')
	cdef numpy.ndarray[numpy.double_t, ndim=3] W=numpy.zeros((slices.shape[1],slices.shape[1],slices.shape[1]),dtype='double')
	cdef int R=int((slices.shape[1]-1)/2)
	cdef numpy.ndarray[numpy.double_t,ndim=2] slice
	cdef numpy.ndarray[numpy.double_t, ndim=1] q
	cdef numpy.ndarray[numpy.double_t, ndim=2] rotmat
	cdef numpy.ndarray[numpy.double_t, ndim=2] k
	cdef numpy.ndarray[numpy.double_t, ndim=1] i
	cdef int x,y,z
	
	cdef double point
	cdef int n,j
	
	for n in range(N):
		slice=slices[n]
		q=quat[n]
		rotmat= CalcRotMat(q)
		k=numpy.zeros(k0.shape,k0.dtype)
		
		for j in range(k0.shape[0]):
			k[j]=numpy.dot(rotmat,k0[j])
			i=k[j]
			i=i+R
			x=i.astype(int)[0]
			y=i.astype(int)[1]
			z=i.astype(int)[2]
			point=slice[R+k0[j,0],R+k0[j,1]]
			W[x,y,z]+=point*(1+x-i[0])*(1+y-i[1])*(1+z-i[2])
			W[x+1,y,z]+=point*(i[0]-x)*(1+y-i[1])*(1+z-i[2])
			W[x,y+1,z]+=point*(1+x-i[0])*(i[1]-y)*(1+z-i[2])
			W[x+1,y+1,z]+=point*(i[0]-x)*(i[1]-y)*(1+z-i[2])
			W[x,y,z+1]+=point*(1+x-i[0])*(1+y-i[1])*(i[2]-z)
			W[x+1,y,z+1]+=point*(i[0]-x)*(1+y-i[1])*(i[2]-z)
			W[x,y+1,z+1]+=point*(1+x-i[0])*(i[1]-y)*(i[2]-z)
			W[x+1,y+1,z+1]+=point*(i[0]-x)*(i[1]-y)*(i[2]-z)
			
			weights[x,y,z]+=(1+x-i[0])*(1+y-i[1])*(1+z-i[2])
			weights[x+1,y,z]+=(i[0]-x)*(1+y-i[1])*(1+z-i[2])
			weights[x,y+1,z]+=(1+x-i[0])*(i[1]-y)*(1+z-i[2])
			weights[x+1,y+1,z]+=(i[0]-x)*(i[1]-y)*(1+z-i[2])
			weights[x,y,z+1]+=(1+x-i[0])*(1+y-i[1])*(i[2]-z)
			weights[x+1,y,z+1]+=(i[0]-x)*(1+y-i[1])*(i[2]-z)
			weights[x,y+1,z+1]+=(1+x-i[0])*(i[1]-y)*(i[2]-z)
			weights[x+1,y+1,z+1]+=(i[0]-x)*(i[1]-y)*(i[2]-z)
			
	weights[weights==0]=1
	W=numpy.divide(W,weights)
	return W


