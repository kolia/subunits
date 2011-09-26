from   time  import time

import simulate_retina
reload(simulate_retina)

import numpy.random as Rand

import sparse_group_lasso as sgl
reload(sgl)

import numpy as np
import pylab as p
from scipy.linalg import schur, block_diag
from   matplotlib.ticker import *

from IPython.Debugger import Tracer; debug_here = Tracer()

from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

def plot_filters(X,same_scale=True):
    for i in range(X.shape[0]-1,-1,-1):
        ax = p.subplot(X.shape[0]*2,1,i*2+1)
        p.plot(np.arange(X.shape[1]),X[i,:])
        ax.yaxis.set_major_locator( MaxNLocator(nbins=1) )
        ax.xaxis.set_major_locator( IndexLocator(10,0) )

def print_coeffs(V,precision=1e-2):
    lasti = -10
    for i,v in enumerate(V):
        if lasti == i-2: print
        if np.sum(np.abs(v))>precision:    
            print i,' : ', v
            lasti = i    

############################
# Setting up simulated data

N_cells=[40,20,12]

V2 = 0.5
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

# Quantities of interest
N_filters = N_cells[1]*10
filters = np.concatenate(
    [simulate_retina.weights(sigma=n, shape=(N_filters,N_cells[0]))
    for n in [10.]] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
def simulator( v2, nonlinearity, N_cells , sigma_spatial , N_timebins ):
    return simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells ,
                          sigma_spatial=sigma_spatial, N_timebins=N_timebins,
                          average_me={'features':lambda x: NL(np.dot(filters,x))} )
simulate = memory.cache(simulator)
R = simulate( V2, nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,3.],
              N_timebins = 1000000 )


total = float( np.sum([Nspikes for Nspikes in R['N_spikes']]) )

Nspikes = R['N_spikes']/total

dSTA  = np.concatenate(
        [STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis]
        for STA in R['statistics']['features']['STA']], axis=1)

D,Z = schur(R['statistics']['features']['cov']/2)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA ) / 2

predictors    = [ block_diag(*[column*np.sqrt(Nspke) for Nspke in Nspikes]).T
                  for column in P.T]


start = time()

group_weights = [0.05 for _ in predictors]
weights       = [0.01*np.ones(pp.shape[1]) for pp in predictors]

r,coeffs  = sgl.initialize_group_lasso(predictors, (np.sqrt(Nspikes)*y).T.flatten())
print r
iterations = sgl.sparse_group_lasso(predictors, group_weights, weights, 
                                    r, coeffs, maxiter=10000, disp_every=100, ftol=1e-8)
finish = time()
print iterations,'Iterations of sparse group LASSO in ',finish-start,' seconds for n: ',n
print 'infered x '
#print np.concatenate( [sgl.inflate(coeffs), np.arange(vv.shape[0])[:,np.newaxis]], axis=1)
print 'coeffs:'
print_coeffs(sgl.inflate(coeffs),precision=1e-1)