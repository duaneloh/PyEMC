from mpi4py import MPI
import numpy as np
from optparse import OptionParser
import os
import h5py

#Introduce command line arguments
#Also specify some default directory names
srcDir = os.getcwd()
quatDir = os.path.join(srcDir, "quaternions")
parser = OptionParser()
parser.add_option("-Q", "--quatDir", action="store", type="string", dest="quatDir", help="absolute path to input quaternions", metavar="", default=quatDir)
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
g_dtype             = np.float32

def readData():
    pass

def readQuaternion(quatFN):
    fp          = open(quatFN)
    lines       = fp.readlines()
    all_quat    = np.array([ll.strip("\n").split("\t") for ll in lines[1:]])
    fp.close()
    print "rank %d done reading %s"%(rank, quatFN)
    return all_quat.astype(g_dtype)

def readModel(fn=None):
    global g_curr_model
    if fn is None:
        g_curr_model = np.random.rand(125,125,125).astype(g_dtype)
    else:
        #read 3D model from h5 file here
        pass

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
    g_my_tomograms1=handyCythonLib.expand(g_curr_model,g_my_quat, g_my_ref_tomo)
    print "Rank %d done with expanding %s tomograms"%(rank, g_my_tomograms1.shape)

def maximize():
    global g_my_tomograms1
    global g_my_tomograms2
    g_my_tomograms2 = g_my_tomograms1.copy()
    print "Rank %d done with copying %s tomograms"%(rank, g_my_tomograms1.shape)

def compress():
    global g_my_quat
    global g_my_tomograms2
    global g_my_moments
    global g_my_weights
    global g_total_moments
    global g_total_weights
    global g_updated_model
    
    g_my_moments, g_my_weights= handyCythonLib.compress(g_my_tomograms2,g_my_quat, g_my_ref_tomo)
    
    #summing up all moments and weights for different processes
    
    comm.Allreduce(g_my_moments, g_total_moments, op= MPI.SUM)
    comm.Allreduce(g_my_weights, g_total_weights, op= MPI.SUM)
    
    #final division of moments by weights
    
    g_updated_model= handyCythonLib.array_division(g_total_moments, g_total_weights)
    
    #total weights and moments should be set to zero(for the next iteration)
    
    g_total_weights[:,:,:]=0
    g_total_moments[:,:,:]=0


def measureModelChange():
    pass

start_t = MPI.Wtime()
readData()
quatFN = os.path.join(op.quatDir, "quaternion10.dat")
makeRefTomogram()
g_all_quat = readQuaternion(quatFN)
g_len_all_quat = len(g_all_quat)
g_num_pix_in_ref_tomo = len(g_my_ref_tomo)

if rank == 0:
    g_all_tomograms1 = np.empty((g_len_all_quat, g_num_pix_in_ref_tomo), dtype=g_dtype)
    g_all_tomograms2 = np.empty((g_len_all_quat, g_num_pix_in_ref_tomo), dtype=g_dtype)


job_len   = [len(rng) for rng in np.array_split(g_len_all_quat, commSize)]
job_lencumsum=numpy.cumsum([len(rng) for rng in np.array_split(g_len_all_quat, commSize)])
g_my_quat=g_all_quat[job_lencumsum[rank]-job_len[rank]:job_lencumsum[rank]]
g_my_tomograms1 = np.empty((job_len, g_num_pix_in_ref_tomo), dtype=g_dtype)
end_t   = MPI.Wtime()
print("Rank %d done with setup in %lf seconds"%(rank, end_t-start_t))


#This is where you would insert the loop.
for iter_num in range(5):
    start_t = MPI.Wtime()
    expand()
    maximize()
    compress()
    end_t   = MPI.Wtime()
    print "Rank %d: I'm done with iteration %d in %lf seconds"%(rank, iter_num, end_t-start_t)






















