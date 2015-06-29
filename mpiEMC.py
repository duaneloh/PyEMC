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
    if qMax is None:
        g_my_ref_tomo = np.random.rand(int(3.14*62*62),3).astype(g_dtype)

def expand():
    #Implement your expansion algorithm here
    #But remember to pre-allocate g_my_tomo
    global g_my_tomograms1
    global g_num_pix_in_ref_tomo
    global job_len
    print "randomizing"
    tmp =10*np.random.rand(job_len, g_num_pix_in_ref_tomo).astype(g_dtype)
    g_my_tomograms1[:] = tmp.copy()
    print "Rank %d tot: %e"%(rank, tmp.mean())

def maximize():
    global g_my_tomograms1
    global g_my_tomograms2
    g_my_tomograms2 = g_my_tomograms1.copy()
    print "Rank %d done with copying %s tomograms"%(rank, g_my_tomograms1.shape)

def compress():
    global g_my_tomograms1
    global g_my_tomograms2
    print "rank %d sum %e --> %e"%(rank, g_my_tomograms1.mean(), g_my_tomograms2.mean())

def measureModelChange():
    pass

start_t = MPI.Wtime()
readData()
quatFN = os.path.join(op.quatDir, "quaternion1.dat")
makeRefTomogram()
g_all_quat = readQuaternion(quatFN)
g_len_all_quat = len(g_all_quat)
g_num_pix_in_ref_tomo = len(g_my_ref_tomo)

if rank == 0:
    g_all_tomograms1 = np.zeros((g_len_all_quat, g_num_pix_in_ref_tomo), dtype=g_dtype)
    g_all_tomograms2 = np.zeros((g_len_all_quat, g_num_pix_in_ref_tomo), dtype=g_dtype)


job_len   = [len(rng) for rng in np.array_split(np.arange(g_len_all_quat), commSize)][rank]
g_my_tomograms1 = np.zeros((job_len, g_num_pix_in_ref_tomo), dtype=g_dtype)
g_my_tomograms2 = np.zeros((job_len, g_num_pix_in_ref_tomo), dtype=g_dtype)
end_t   = MPI.Wtime()
print("Rank %d done with setup in %lf seconds"%(rank, end_t-start_t))

#This is where you would insert the loop.
for iter_num in range(2):
    start_t = MPI.Wtime()
    comm.Scatter([g_all_tomograms1, MPI.FLOAT], [g_my_tomograms1, MPI.FLOAT], root=0)
    comm.Scatter([g_all_tomograms2, MPI.FLOAT], [g_my_tomograms2, MPI.FLOAT], root=0)
    expand()
    maximize()
    compress()
    print "Rank %d compressiong tomograms 2"%rank
    comm.Gather([g_my_tomograms2, MPI.FLOAT], [g_all_tomograms2, MPI.FLOAT], root=0)
    print "Rank %d compressiong tomograms 1"%rank
    comm.Gather([g_my_tomograms1, MPI.FLOAT], [g_all_tomograms1, MPI.FLOAT], root=0)
    end_t   = MPI.Wtime()
    print "Rank %d: I'm done with iteration %d in %lf seconds"%(rank, iter_num, end_t-start_t)
