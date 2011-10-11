import numpy
import itertools

from scipy.io import loadmat
datarun = loadmat('datarun.mat')
datarun = datarun['datarun']

T = datarun['duration'][0][0][0][0]

Nspikes = [s.size for s in datarun['spikes'][0][0][0]]

# list of cones connected to each RGC
selected_cones = [[int(x) for x in numpy.nonzero(numpy.squeeze(sc[:]))[0]]
                  for sc in datarun['selected_cones'][0][0][0]]
all_cones = sorted(set(itertools.chain(*selected_cones)))
N_cones   = len(all_cones)
index = dict((c,i) for i,c in enumerate(all_cones))

indices = [numpy.array([index[i] for i in s],dtype=int) for s in selected_cones]

cones = numpy.squeeze( datarun['cones'][0][0][0]['centers'][0] )
cones = cones[all_cones,:]

cone_types = [s for s in datarun['cones'][0][0][0]['types'][0]] # L, M, or S
cone_types = [cone_types[i+1] for i in all_cones]   # <-- indices start at 1 !(?)!

STAs = [numpy.zeros(N_cones) for _ in Nspikes]
for sta, values, indice in zip( STAs, datarun['cone_stas'][0][0][0], indices):
    sta[indice] = values

STCs = [numpy.zeros((N_cones,N_cones)) for _ in Nspikes]
for stc, values, indice in zip( STCs, datarun['cone_stcs'][0][0][0], indices):
    stc[numpy.ix_(indice,indice)] = numpy.squeeze( values )

cov_stimulus = numpy.zeros((N_cones,N_cones))    # <--  missing most pairs here !!
for values, indice in zip( datarun['stim_cov'][0][0][0], indices):
    cov_stimulus[numpy.ix_(indice,indice)] = numpy.squeeze( values )

mean_stimulus = [numpy.zeros(N_cones) for _ in Nspikes]
for mean, values, indice in zip( mean_stimulus, datarun['stim_mean'][0][0][0], indices):
    mean[indice] = values

# center of mass of the ith RGC's receptive field
RGC_center_of_mass = [numpy.squeeze(com) for com in datarun['rf_coms'][0][0][0]]