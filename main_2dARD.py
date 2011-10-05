import kolia_base as kb
reload(kb)

import kolia_theano
reload(kolia_theano)

import simulate_retina
reload(simulate_retina)

import optimize
reload(optimize)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import numpy        as np
from scipy.linalg   import schur

import pylab        as p

from IPython.Debugger import Tracer; debug_here = Tracer()

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

############################
# Setting up simulated data
 
#N_cells=[40,20,12]
#N_cells=[30,15,9]
#N_cells=[20,10,6]
N_cells=[20,13,7]

V2 = 0.5
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
 
# Quantities of interest
overcompleteness = 10
N_filters = N_cells[1]*overcompleteness
filters = np.concatenate(
    [simulate_retina.weights(sigma=n, shape=(N_filters,N_cells[0]))
    for n in [10.]] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
def simulator( v2, N_filters, nonlinearity, N_cells , sigma_spatial , N_timebins ):
    retina = simulate_retina.LNLNP_ring_model( nonlinearity = nonlinearity , 
                                               N_cells = N_cells , 
                                               sigma_spatial = sigma_spatial )
#    stimulus = simulate_retina.white_gaussian_stimulus( dimension  = N_cells[0] , 
#                                                         sigma = 1. )
    stimulus = simulate_retina.Stimulus( simulate_retina.white_gaussian )
    return simulate_retina.run_LNLEP( retina , stimulus = stimulus , 
                   N_timebins = N_timebins ,
                   average_me = {'features':lambda x: NL(np.dot(filters,x))} )
simulate = memory.cache(simulator)
R = simulate( V2, N_filters, nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,3.],
              N_timebins = 1000000 )
 
#testR = simulate( V2, nonlinearity=NL, N_cells=N_cells , sigma_spatial=[20.,3.],
#                  N_timebins = 90000 )
 

total = float( np.sum([Nspikes for Nspikes in R['N_spikes']]) )

Nspikes = R['N_spikes']/total

dSTA  = np.concatenate(
        [STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis]
        for STA in R['statistics']['features']['STA']], axis=1)

Cin = R['statistics']['features']['cov']/2

#lam2 = 200.
#prior = np.fromfunction( lambda i,j: (i-j>0)*(i-j<6) , Cin.shape)
##prior = np.dot( filters , filters.T )
##prior = (prior - np.diag(np.diag(prior)))
##prior = prior / np.sqrt( np.sum( prior**2. ) )
#Cin = Cin + lam2 * prior

D,Z = schur(Cin)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA ) / 2

irls = memory.cache(IRLS)
V, iW = irls( y, P, x=0, disp_every=1000, lam=0.005, maxiter=1000000 , 
              ftol=1e-5, nonzero=1e-1)
print 'V'
kb.print_sparse_rows( V, precision=1e-1 )

keepers = np.array( [sum(abs(v))>1e-1 for v in V] )
U        = filters[keepers,:]
V1       = V[keepers,:].T