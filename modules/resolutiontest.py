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

#sys.path.append(os.path.abspath('../myFunctions'))
import handyCythonLib
#import unhandyCythonLib

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

L=375
ce=(L-1)/2
R=48
sigma=R/2
quatlists=range(1,11)#[10]
structure=numpy.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])


Ro=numpy.zeros((L,L,L),dtype='double')
W = RandomVolume(R,CreateSliceOnly=False)
#W=numpy.random.random_integers(0,10000,(2*R+1,2*R+1,2*R+1))
Ro[ce-R:ce+R+1,ce-R:ce+R+1,ce-R:ce+R+1]=W
#mask=Ro>=filter.threshold_otsu(Ro[Ro>0])
#mask=ndimage.binary_fill_holes(mask)
#mask=ndimage.binary_opening(mask, structure=structure)
#mask=ndimage.binary_closing(mask, structure=morphology.ball(3), iterations=1)
#Ro=ndimage.filters.gaussian_filter(Ro, sigma=5)#*mask
inputDir = '/Volumes/Untitled/3D-reconstruction/29_20150504_141003_2015-05-05-14-05/'
path= '/mnt/cbis/home/barzhas/working_dir/PyEMC/quaternions/'
saving='/mnt/cbis/home/barzhas/working_dir/PyEMC/testresults1/'
xv,yv,zv=numpy.mgrid[-ce:ce+1,-ce:ce+1,-ce:ce+1]
kv=numpy.double(numpy.sqrt(xv**2+yv**2+zv**2))
r3=numpy.round(kv).astype('int')
kradius=numpy.arange(r3.max()+1)
gau= numpy.exp(-kv*kv/(2*sigma**2))
Rok=numpy.fft.fftshift(numpy.fft.fftn(Ro))
Rbk=Rok*gau
rbk=numpy.double(numpy.abs(Rbk))
Rbr=numpy.abs(numpy.fft.ifftn(Rbk))
timelist=numpy.zeros((len(quatlists),2),dtype='double')
memorylist=numpy.zeros((len(quatlists),3),dtype='double')
numquat=numpy.zeros(len(quatlists), dtype='int')
ResResErr=numpy.zeros((len(quatlists), len(kradius)), dtype='double')
angave=numpy.zeros((len(quatlists)+1, len(kradius)), dtype='double')
ratio=numpy.zeros((len(quatlists), len(kradius)), dtype='double')
xy=numpy.zeros((len(quatlists)+1, L), dtype='double')
xz=numpy.zeros((len(quatlists)+1, L), dtype='double')
yz=numpy.zeros((len(quatlists)+1, L), dtype='double')
angave[0]=handyCythonLib.angAve(rbk,r3)
k0=RefTom(ce)
xy[0]=log(rbk[ce,ce,:])
xz[0]=log(rbk[ce,:,ce])
yz[0]=log(rbk[:,ce,ce])
title=os.path.join(saving,'center cut original.png')
showcuts(log(rbk),ce,title)

for i in range(len(quatlists)):
	quatx,n=readlist(path,quatlists[i])
	print n
	quat=quatx[:,:4]
	numquat[i]=n
	t0=time.time()
	m0=memorycons()
	print 'expansion'
	slices=handyCythonLib.expand(rbk,quat,k0)
	t1=time.time()
	m1=memorycons()
	print 'time executed for expansion is',t1-t0
	print 'memory executed for expansion is',m1-m0
	print 'compression'
	rbk1=handyCythonLib.compress(slices,quat,k0)
	t2=time.time()
	m2=memorycons()
	print 'time executed for compression is',t2-t1
	print 'memory executed for compression is',m2-m1
	title=os.path.join(saving,'center cut '+str(quatlists[i])+'.png')
	showcuts(log(rbk1),ce,title)
	memorylist[i]=numpy.asarray([m0,m1,m2])
	timelist[i]=numpy.asarray([t1-t0,t2-t1])
	ResResErr[i]=handyCythonLib.angAveDif(rbk1,rbk,r3)/angave[0]
	ratio[i]=handyCythonLib.angAve(rbk1/rbk,r3)
	angave[i+1]=handyCythonLib.angAve(rbk1,r3)
	xy[i+1]=log(rbk1[ce,ce,:])
	xz[i+1]=log(rbk1[ce,:,ce])
	yz[i+1]=log(rbk1[:,ce,ce])



plotcuts(yf=xy,label='x=0,y=0')
plotcuts(yf=xy,label='x=0,z=0')
plotcuts(yf=yz,label='z=0,y=0')
plotcuts(yf=ResResErr, label='error-L=375-R=20', dropzero=True,xlim=True, ylim=True)
plotcuts(yf=angave/angave[0], label='ratio of angular average', xlim=True)
plotcuts(yf=ratio, label=' angular average of ratio',dropzero=True, xlim=True)
iangave=angave[:,:ce]
plotcuts(yf=iangave[0]/iangave, label='inverse angular average of ratio',dropzero=True)





#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(numquat,memorylist[:,0],numquat,memorylist[:,1],numquat,memorylist[:,2])
#ax.set_ylabel('memory usage')
#ax.set_xlabel('number of quaternions')
#fig.savefig(saving+'/memory usage.png')
#plt.close(fig)
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(numquat,timelist[:,0])
#ax.set_ylabel('time used for expansion')
#ax.set_xlabel('number of quaternions')
#fig.savefig(saving+'/time usage expansion.png')
#plt.close(fig)
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(numquat,timelist[:,1])
#ax.set_ylabel('time used for compression')
#ax.set_xlabel('number of quaternions')
#fig.savefig(saving+'/time usage compression.png')
#plt.close(fig)

