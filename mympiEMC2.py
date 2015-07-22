from mpi4py import MPI
import numpy as np
from optparse import OptionParser
import sys,os
import h5py

sys.path.append(os.path.abspath('modules'))
import mpihandyCythonLib
#Introduce command line arguments
#Also specify some default directory names
srcDir = os.getcwd()
quatDir = os.path.join(srcDir, "quaternions")
modelDir = os.path.join(srcDir, "models")
parser = OptionParser()
parser.add_option("-Q", "--quatDir", action="store", type="string", dest="quatDir", help="absolute path to input quaternions", metavar="", default=quatDir)
parser.add_option("-M", "--modelDir", action="store", type="string", dest="modelDir", help="absolute path to store generated models", metavar="", default=modelDir)
#parser.add_option("-d", action="store_true", dest="detailed", default=False, help="do detailed reconstruction")
(op, args) = parser.parse_args()

comm            = MPI.COMM_WORLD
commSize        = comm.Get_size()
rank            = comm.Get_rank()

g_all_quat          = None
g_my_quat           = None
g_curr_model        = None
g_my_ref_tomo       = None
g_all_tomograms1    = None
g_all_tomograms2    = None
g_my_tomograms1     = None
g_my_tomograms2     = None
g_my_moments        = None
g_my_weights        = None
g_total_moments     = None
g_total_weights     = None
g_updated_model     = None
g_dtype             = np.float64
qMax                = 62
dataDir=""
def readData(dataDir):
    global g_my_ref_tomo
    global dataStack
    
    data=h5py.File(dataDir,"r")
    movie=data[data.keys()[0]].value
    a,b,c=movie.shape
    dataStack=np.empty((a,g_my_ref_tomo.shape[0]),dtype='float')
    average_data=0
    print 'rank', rank, 'is reading the data'
    for i in range(a):
        for j in range(g_my_ref_tomo.shape[0]):
            dataStack[i,j]=movie[i,g_my_ref_tomo[j,0],g_my_ref_tomo[j,1]]
            average_data+=dataStack[i,j]
    data.close()
    print 'rank', rank, 'has finished reading the data'
    average_data/=a
    return dataStack,average_data

def readQuaternion(quatFN):
    fp          = open(quatFN)
    lines       = fp.readlines()
    all_quat    = np.array([ll.strip("\n").split("\t") for ll in lines[1:]])
    fp.close()
    print "rank %d done reading %s"%(rank, quatFN)
    return all_quat.astype(g_dtype)

def readModel(fn=None):
    global g_curr_model 
    g_curr_model=np.empty((125,125,125),dtype=g_dtype)
    if fn is None:
        if rank==0:
            g_curr_model = np.random.rand(125,125,125).astype(g_dtype)
            modelFN = os.path.join(op.modelDir, "model0.h5")
            f=h5py.File(modelFN,"w")
            f.create_dataset("data",data=g_curr_model,compression="gzip", compression_opts=9)
            f.close()
            print 'hi! this is rank=0 and I am done with creating and saving random volume function!'
            comm.Bcast(g_curr_model)
            print 'hello once again! that is rank=0 and I have broadcasted the current model to everyone'
    else:
        #read 3D model from h5 file here
        modelFN = os.path.join(op.modelDir, fn)
        f=h5py.File(modelFN,"r")
        g_curr_model=f["data"].value
        f.close()

def saveModel(i):
        global g_updated_model
        #write 3D model to h5 file here
        modelFN = os.path.join(op.modelDir, "model"+str(i+1)+".h5")
        f=h5py.File(modelFN,"w")
        f.create_dataset("data",data=g_updated_model,compression="gzip", compression_opts=9)
        f.close()

def makeRefTomogram(qMax=None):
    global g_my_ref_tomo
    if qMax is None or qMax<2:
        g_my_ref_tomo = np.random.rand(int(3.14*62*62),3).astype(g_dtype)
    else:
        xv,yv=np.mgrid[-qMax:qMax+1,-qMax:qMax+1]
        r=np.sqrt(xv**2+yv**2)
        a,b=np.where(r<=qMax)
        a=a-qMax
        b=b-qMax
        c=np.zeros(a.shape,a.dtype)
        g_my_ref_tomo=np.vstack((a,b,c)).T

def expand():
    #Implement your expansion algorithm here
    #But remember to pre-allocate g_my_tomo
    global g_my_tomograms1
    global g_my_ref_tomo
    global g_my_quat
    global g_curr_model
    print "Rank %d done with tomo shape %s"%(rank, g_my_ref_tomo.shape)
    g_my_tomograms1=mpihandyCythonLib.expand(g_curr_model,g_my_quat, g_my_ref_tomo)
    print "Rank %d done with expanding %s tomograms"%(rank, g_my_tomograms1.shape)

def maximize(Rmin, Rmax, g_len_all_quat, g_num_frames,job_len,job_start):
    global g_my_tomograms1
    global g_my_tomograms2
    global g_my_ref_tomo
    
    g_my_tomograms2 = g_my_tomograms1.copy()
    print "Rank %d done with copying %s tomograms"%(rank, g_my_tomograms1.shape)
    g_my_distances=mpihandyCythonLib.maximize(dataStack,g_my_tomograms1,g_my_ref_tomo, Rmin, Rmax)
    g_total_distances=np.empty((g_len_all_quat,g_num_frames), dtype='float')
    print 'stacking distance arrays together'
    comm.Reduce(g_my_distances, g_total_distances, op= MPI.SUM)
#    comm.Barrier()
    if rank==0:
        print 'converting total distance array to binary'
        min_distances = np.argmin(g_total_distances,axis=0)
        g_total_binary_dist = np.zeros_like(g_total_distances)
        del g_total_distances
        del g_my_distances
        g_total_binary_dist[min_distances, np.arange(g_num_frames)]=1
        g_total_binary_dist=g_total_binary_dist.flatten()
    else:
        del g_total_distances
        del g_my_distances
        g_total_binary_dist = None
    g_my_binary_dist=np.empty(len(g_my_tomograms1)*g_num_frames,dtype='float')
    print "scattering total binary array to the processes"
    comm.Scatterv([g_total_binary_dist,tuple(job_len*g_num_frames),tuple(job_start*g_num_frames),MPI.FLOAT],g_my_binary_dist)
    g_my_binary_dist=np.resize(g_my_binary_dist,(len(g_my_tomograms1),g_num_frames))
    print 'rank',rank,'is substituting the tomograms with the data'
    for i in range(len(g_my_tomograms2)):
        if np.sum(g_my_binary_dist[i,:])==0:
            continue
        similarFrameIndex=np.nonzero(g_my_binary_dist[i,:])[0]
        sumSimilarFrames=np.sum(dataStack[similarFrameIndex],axis=0)
        g_my_tomograms2[i]=1.0*sumSimilarFrames/np.sum(g_my_binary_dist[i,:])
        print 'rank',rank,'has updated tomogram',i,' by the average of',np.sum(g_my_binary_dist[i,:]),'frames'
    del g_my_binary_dist


def compress():
    global g_my_quat
    global g_my_tomograms2
    global g_my_moments
    global g_my_weights
    global g_total_moments
    global g_total_weights
    global g_updated_model
    
    g_my_moments, g_my_weights= mpihandyCythonLib.compress(g_my_tomograms2,g_my_quat, g_my_ref_tomo)
    
    #summing up all moments and weights for different processes
    
    comm.Allreduce(g_my_moments, g_total_moments, op= MPI.SUM)
    comm.Allreduce(g_my_weights, g_total_weights, op= MPI.SUM)
    
    #final division of moments by weights
    
    g_updated_model= mpihandyCythonLib.array_division(g_total_moments, g_total_weights)
    
    #total weights and moments should be set to zero(for the next iteration)
    
    g_total_weights[:,:,:]=0
    g_total_moments[:,:,:]=0


def measureModelChange():
    pass

start_t = MPI.Wtime()
makeRefTomogram(qMax=qMax)
dataStack,average_data = readData(dataDir)
g_num_frames=len(dataStack)
print average_data
readModel("model0.h5")
quatFN = os.path.join(op.quatDir, "quaternion4.dat")
g_all_quat = readQuaternion(quatFN)
g_len_all_quat = len(g_all_quat)
g_num_pix_in_ref_tomo = len(g_my_ref_tomo)

#if rank == 0:
#    g_all_tomograms1 = np.empty((g_len_all_quat, g_num_pix_in_ref_tomo), dtype=g_dtype)
#    g_all_tomograms2 = np.empty((g_len_all_quat, g_num_pix_in_ref_tomo), dtype=g_dtype)

g_total_weights=np.zeros_like(g_curr_model)
g_total_moments=np.zeros_like(g_curr_model)
job_len   = [len(rng) for rng in np.array_split(np.arange(g_len_all_quat), commSize)]
job_start=np.cumsum(job_len)-job_len
#g_my_quat=g_all_quat[job_lencumsum[rank]-job_len[rank]:job_lencumsum[rank]]
g_my_quat=g_all_quat[np.array_split(np.arange(g_len_all_quat), commSize)[rank]]
g_my_indeces=np.array_split(np.arange(g_len_all_quat), commSize)[rank]
g_my_tomograms1 = np.zeros((job_len[rank], g_num_pix_in_ref_tomo), dtype=g_dtype)
end_t   = MPI.Wtime()
print("Rank %d done with setup in %lf seconds"%(rank, end_t-start_t))

print "rank %d has model with %lf sum"%(rank, g_curr_model.sum())
#This is where you would insert the loop.
average_model=np.zeros(1)
for iter_num in range(5):
    start_t = MPI.Wtime()
    if iter_num==0:
        expand()
        local_average_model=np.sum(g_my_tomograms1)
        comm.Allreduce(local_average_model,average_model)
        average_model/=g_len_all_quat
        if rank==0:
            print average_model
        g_my_tomograms2=g_my_tomograms1*(average_data/average_model)
        compress()
    else:
        expand()
        maximize(Rmin, Rmax, g_len_all_quat, g_num_frames,job_len,job_start)
        compress()
    end_t   = MPI.Wtime()
    
    if rank==0:
        saveModel(iter_num)
    g_curr_model=g_updated_model.copy()
    print "Rank %d: I'm done with iteration %d in %lf seconds"%(rank, iter_num, end_t-start_t)






















