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
#	rotmat[0,2]=2*q2*q4+2*q1*q3
	rotmat[1,0]=2*q2*q3+2*q1*q4
	rotmat[1,1]=q1**2-q2**2+q3**2-q4**2
#	rotmat[1,2]=2*q3*q4-2*q1*q2
	rotmat[2,0]=2*q2*q4-2*q1*q3
	rotmat[2,1]=2*q3*q4+2*q1*q2
#	rotmat[2,2]=q1**2-q2**2-q3**2+q4**2

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

def ExtractSliceValues(numpy.ndarray[numpy.double_t, ndim=3] W,numpy.ndarray[numpy.double_t, ndim=1] q, numpy.ndarray[numpy.int_t, ndim=2] k0):
	assert W.dtype==numpy.double and q.dtype==numpy.double and k0.dtype==numpy.int
	cdef int R=int((W.shape[0]-1)/2)
	cdef numpy.ndarray[numpy.double_t, ndim=1] i
	cdef int j
	cdef int l=k0.shape[0]
	cdef numpy.ndarray[numpy.double_t, ndim=2] 	slice=numpy.zeros((2*R+1,2*R+1),dtype='double')
	cdef numpy.ndarray[numpy.double_t, ndim=2] k=numpy.zeros((l,3),dtype='double')
	cdef numpy.ndarray[numpy.double_t, ndim=2]	rotmat=CalcRotMat(q)
	cdef numpy.ndarray[numpy.int_t, ndim=2] vlist=numpy.zeros((8,3),dtype='int')
	cdef numpy.ndarray[numpy.double_t, ndim=1] v=numpy.zeros(8,dtype='double')
	cdef numpy.ndarray[numpy.int_t, ndim=1] u=numpy.zeros(8,dtype='int')
	cdef int x,y,z,h,g, loc
	cdef double wx1, wx2, wy1,wx3,wx4,wy2
#	cdef double

	for j in range(k0.shape[0]):
	
		k[j]=numpy.dot(rotmat,k0[j])
		i=k[j]
		i=i+R
		x=i.astype(int)[0]
		y=i.astype(int)[1]
		z=i.astype(int)[2]
		vlist=numpy.array([[x,y,z],[x+1,y,z],[x,y+1,z],[x+1,y+1,z],[x,y,z+1],[x+1,y,z+1],[x,y+1,z+1],[x+1,y+1,z+1]])
		for h in range(8):
			for g in range(3):
				loc=vlist[h,g]
				if loc<0 or loc>124:
					u[h]+=1
		for h in range(8):
			if u[h]==0:
				v[h]=W[vlist[h,0],vlist[h,1],vlist[h,2]]
		slice[R+k0[j,0],R+k0[j,1]]=v[0]*(x+1-i[0])*(y+1-i[1])*(z+1-i[2])+v[1]*(i[0]-x)*(y+1-i[1])*(z+1-i[2])+v[2]*(x+1-i[0])*(i[1]-y)*(z+1-i[2])+v[3]*(i[0]-x)*(i[1]-y)*(z+1-i[2])+	v[4]*(x+1-i[0])*(y+1-i[1])*(i[2]-z)+v[5]*(i[0]-x)*(y+1-i[1])*(i[2]-z)+	v[6]*(x+1-i[0])*(i[1]-y)*(i[2]-z)+	v[7]*(i[0]-x)*(i[1]-y)*(i[2]-z)
#		wx1=W[x,y,z]*(x+1-i[0])+W[x+1,y,z]*(i[0]-x)
#		wx2=W[x,y+1,z]*(x+1-i[0])+W[x+1,y+1,z]*(i[0]-x)
#		wy1=wx1*(y+1-i[1])+wx2*(i[1]-y)
#		wx3=W[x,y,z+1]*(x+1-i[0])+W[x+1,y,z+1]*(i[0]-x)
#		wx4=W[x,y+1,z+1]*(x+1-i[0])+W[x+1,y+1,z+1]*(i[0]-x)
#		wy2=wx3*(y+1-i[1])+wx4*(i[1]-y)
#		slice[R+k0[j,0],R+k0[j,1]]=wy1*(z+1-i[2])+wy2*(i[2]-z)

	return slice


def expand(numpy.ndarray[numpy.double_t, ndim=3] W, numpy.ndarray[numpy.double_t, ndim=2]quat,numpy.ndarray[numpy.int_t, ndim=2]k0):

	assert W.dtype==numpy.double and k0.dtype==numpy.int and quat.dtype==numpy.double
	cdef int R=(W.shape[0]-1)/2
	cdef int N	= quat.shape[0]
	cdef numpy.ndarray[numpy.double_t, ndim=3]	stack=numpy.zeros((quat.shape[0],W.shape[1],W.shape[2]), dtype='double')
	cdef numpy.ndarray[numpy.double_t,ndim=2] rotmat=numpy.zeros((3,3), dtype='double')
	cdef int n,j
	cdef double rx,ry,rz,dx1,dx,dy1,dy,dz1,dz
	cdef double v000,v100,v010,v110,v001,v101,v011,v111
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
			ix=ix+R
			iy=iy+R
			x=int(rx)
			y=int(ry)
			z=int(rz)
			dx1=rx-x
			dy1=ry-y
			dz1=rz-z
			dx=x+1-rx
			dy=y+1-ry
			dz=z+1-rz
			v000=W[x,y,z];v100=W[x+1,y,z];v010=W[x,y+1,z];v110=W[x+1,y+1,z]
			v001=W[x,y,z+1];v101=W[x+1,y,z+1];v011=W[x,y+1,z+1];v111=W[x+1,y+1,z+1]
			stack[n,ix,iy]=v000*dx*dy*dz+v100*dx1*dy*dz+v010*dx*dy1*dz+v110*dx1*dy1*dz+v001*dx*dy*dz1+v101*dx1*dy*dz1+v011*dx*dy1*dz1+v111*dx1*dy1*dz1
	return stack


def compress(numpy.ndarray[numpy.double_t,ndim=3] slices,numpy.ndarray[numpy.double_t, ndim=2] quat,numpy.ndarray[numpy.int_t, ndim=2] k0):
	assert slices.dtype==numpy.double and k0.dtype==numpy.int and quat.dtype==numpy.double
	
	cdef int N=quat.shape[0]
	cdef numpy.ndarray[numpy.double_t,ndim=3] weights=numpy.zeros((slices.shape[1],slices.shape[1],slices.shape[1]),dtype='double')
	cdef numpy.ndarray[numpy.double_t, ndim=3] W=numpy.zeros((slices.shape[1],slices.shape[1],slices.shape[1]),dtype='double')
	cdef int R=int((slices.shape[1]-1)/2)
	cdef int x,y,z,ix,iy
	
	cdef double point,rx,ry,rz,dx,dy,dz,dx1,dy1,dz1
	cdef int n,j,i,k
	
	for n from 0 <= n < N:
		for j from 0 <= j < k0.shape[0]:
			ix=k0[j,0]
			iy=k0[j,1]
			rx=(quat[n,0]**2+quat[n,1]**2-quat[n,2]**2-quat[n,3]**2)*ix+(2*quat[n,1]*quat[n,2]-2*quat[n,0]*quat[n,3])*iy
			ry=(2*quat[n,1]*quat[n,2]+2*quat[n,0]*quat[n,3])*ix+(quat[n,0]**2-quat[n,1]**2+quat[n,2]**2-quat[n,3]**2)*iy
			rz=(2*quat[n,1]*quat[n,3]-2*quat[n,0]*quat[n,2])*ix+(2*quat[n,2]*quat[n,3]+2*quat[n,0]*quat[n,1])*iy
			if sqrt(rx*rx+ry*ry+rz*rz)>0.99*R:
				continue
			rx=rx+R
			ry=ry+R
			rz=rz+R
			ix=ix+R
			iy=iy+R
			x=int(rx)
			y=int(ry)
			z=int(rz)
			point=slices[n,ix,iy]
			dx1=rx-x
			dy1=ry-y
			dz1=rz-z
			dx=x+1-rx
			dy=y+1-ry
			dz=z+1-rz
			weights[x,y,z]+=dx*dy*dz
			weights[x+1,y,z]+=dx1*dy*dz
			weights[x,y+1,z]+=dx*dy1*dz
			weights[x+1,y+1,z]+=dx1*dy1*dz
			weights[x,y,z+1]+=dx*dy*dz1
			weights[x+1,y,z+1]+=dx1*dy*dz1
			weights[x,y+1,z+1]+=dx*dy1*dz1
			weights[x+1,y+1,z+1]+=dx1*dy1*dz1
			W[x,y,z]+=point*weights[x,y,z]
			W[x+1,y,z]+=point*weights[x+1,y,z]
			W[x,y+1,z]+=point*weights[x,y+1,z]
			W[x+1,y+1,z]+=point*weights[x+1,y+1,z]
			W[x,y,z+1]+=point*weights[x,y,z+1]
			W[x+1,y,z+1]+=point*weights[x+1,y,z+1]
			W[x,y+1,z+1]+=point*weights[x,y+1,z+1]
			W[x+1,y+1,z+1]+=point*weights[x+1,y+1,z+1]

#			vlist=numpy.array([[x,y,z],[x+1,y,z],[x,y+1,z],[x+1,y+1,z],[x,y,z+1],[x+1,y,z+1],[x,y+1,z+1],[x+1,y+1,z+1]])
#			u[:]=0
#			for h in range(8):
#				for g in range(3):
#					locpoint=vlist[h][g]
#					if locpoint<0 or locpoint>2*R:
#						u[h]+=1
#			for h in range(8):
#				loc=vlist[h]
#				if u[h]==0:

#			weights[loc[0],loc[1],loc[2]]=(1-abs(i[0]-loc[0]))*(1-abs(i[1]-loc[1]))*(1-abs(i[2]-loc[2]))
#			W[loc[0],loc[1],loc[2]]=point*(1-abs(i[0]-loc[0]))*(1-abs(i[1]-loc[1]))*(1-abs(i[2]-loc[2]))

#			point=slice[R+k0[j,0],R+k0[j,1]]
#			W[x,y,z]+=point*(1+x-i[0])*(1+y-i[1])*(1+z-i[2])
#			W[x+1,y,z]+=point*(i[0]-x)*(1+y-i[1])*(1+z-i[2])
#			W[x,y+1,z]+=point*(1+x-i[0])*(i[1]-y)*(1+z-i[2])
#			W[x+1,y+1,z]+=point*(i[0]-x)*(i[1]-y)*(1+z-i[2])
#			W[x,y,z+1]+=point*(1+x-i[0])*(1+y-i[1])*(i[2]-z)
#			W[x+1,y,z+1]+=point*(i[0]-x)*(1+y-i[1])*(i[2]-z)
#			W[x,y+1,z+1]+=point*(1+x-i[0])*(i[1]-y)*(i[2]-z)
#			W[x+1,y+1,z+1]+=point*(i[0]-x)*(i[1]-y)*(i[2]-z)
#			
#			weights[x,y,z]+=(1+x-i[0])*(1+y-i[1])*(1+z-i[2])
#			weights[x+1,y,z]+=(i[0]-x)*(1+y-i[1])*(1+z-i[2])
#			weights[x,y+1,z]+=(1+x-i[0])*(i[1]-y)*(1+z-i[2])
#			weights[x+1,y+1,z]+=(i[0]-x)*(i[1]-y)*(1+z-i[2])
#			weights[x,y,z+1]+=(1+x-i[0])*(1+y-i[1])*(i[2]-z)
#			weights[x+1,y,z+1]+=(i[0]-x)*(1+y-i[1])*(i[2]-z)
#			weights[x,y+1,z+1]+=(1+x-i[0])*(i[1]-y)*(i[2]-z)
#			weights[x+1,y+1,z+1]+=(i[0]-x)*(i[1]-y)*(i[2]-z)
	for i from 0<= i < slices.shape[1]:
		for j from 0<= j < slices.shape[1]:
			for k from 0<= k < slices.shape[1]:
				if weights[i,j,k]>0:
					W[i,j,k]=W[i,j,k]/weights[i,j,k]

	return W

def angAve3D(numpy.ndarray[numpy.double_t, ndim=3] W1,numpy.ndarray[numpy.double_t, ndim=3] W2, numpy.ndarray[numpy.int64_t, ndim=3] r3):
	assert W1.dtype==numpy.double and r3.dtype==numpy.int
	
	cdef int i,j,k
	cdef int rlim=r3.max()+1
	cdef numpy.ndarray[numpy.double_t, ndim=1] sumint=numpy.zeros(rlim, dtype='double')
	cdef numpy.ndarray[numpy.int_t,ndim=1] count=numpy.zeros(rlim, dtype='int')
	cdef int a= W1.shape[0]
	cdef int b= W1.shape[1]
	cdef int c= W1.shape[2]
#	a,b,c=Volume.shape
#	xv,yv,zv=numpy.mgrid[-(a-1)/2:(a+1)/2,-(b-1)/2:(b+1)/2,-(c-1)/2:(c+1)/2]
	for i in range(a):
		for j in range(b):
			for k in range(c):
				sumint[r3[i,j,k]]+=(W1[i,j,k]-W2[i,j,k])**2
				count[r3[i,j,k]]+=1
	for i in range(rlim):
		if count[i]>0:
			sumint[i]=sqrt(sumint[i]/count[i])
	return sumint






