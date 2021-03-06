import simulate_retina
reload(simulate_retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

from theano import config

import numpy        as np
import pylab        as p
from scipy.linalg   import schur
import numpy.random as Rand

import cPickle

import LQuadLExP
reload(LQuadLExP)
from LQuadLExP import posterior_dU, posterior_dV2, posterior_dV1, posterior_dUV1

from IPython.Debugger import Tracer; debug_here = Tracer()

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

def print_coeffs(V,precision=1e-2):
    lasti = -10
    for i,v in enumerate(V):
        if lasti == i-2: print
        if np.sum(np.abs(v))>precision:
            print i,' : ', v
            lasti = i

def plot_filters(X,same_scale=True):
    for i in range(X.shape[0]):
        p.subplot(X.shape[0],1,i+1)
        p.plot(np.arange(X.shape[1]),X[i,:])
        p.show()

def save(result,name):
    savefile = open("../../../Desktop/%s.pyckle" % name,'w')
    cPickle.dump(R,savefile)
    savefile.close()

def load(name):
    savefile = open('../../../Desktop/%s.pyckle' % name,'r')
    return cPickle.load(savefile)
    

############################
# Setting up simulated data

N_cells=[40,20,12]

V2 = 0.3
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

# Quantities of interest
N_filters = N_cells[1]*10
filters = np.concatenate(
    [simulate_retina.weights(sigma=n, shape=(N_filters,N_cells[0]))
    for n in [10.]] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
def simulator(nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,2.],
              N_timebins = 100000):
    return simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,2.],
              average_me={'features':lambda x: NL(np.dot(filters,x))},
              N_timebins = 100000 )
simulate = memory.cache(simulator)
R = simulate( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,2.],
              N_timebins = 100000 )

#R = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,2.],
#              average_me={'features':lambda x: NL(np.dot(filters,x))},
#              N_timebins = 100000 )

            
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

irls = memory.cache(IRLS)
V, iW = irls( y, P, x=0, disp_every=1000, lam=0.012, maxiter=1000000 , 
              ftol=1e-5, nonzero=1e-1)
print 'V'
print_coeffs( V, precision=1e-1 )

keepers = np.array( [sum(abs(v))>1e-1 for v in V] )

Ncones   = R['N_cells'][0]
Nsub     = R['N_cells'][1]
NRGC     = R['N_cells'][2]
N_spikes = R['N_spikes']
STA      = R['statistics']['stimulus']['STA']
STC      = R['statistics']['stimulus']['STC']
Nproj    = np.sum(keepers)
U        = filters[keepers,:]
V1       = V[keepers,:].T
V2       = V2 * np.ones(Nproj)

#config.compute_test_value = 'warn'
config.compute_test_value = 'off'

#data     = [ V2, N_spikes , STA , STC ]
#baye_ARD = posterior_dUV1( data, Nproj )
#init_params = np.concatenate([ filters[keepers,:].flatten() , \
#                        np.maximum(V1,0.001*np.ones_like(V1)).flatten() ])
#params = baye_ARD.optimize(init_params,maxiter=5000)

i=0
Vs = {i: V1}
Us = {i: U }
baye_V1= posterior_dV1( [ U, V2, N_spikes , STA , STC], Nproj)
V1 = baye_V1.optimize(0.01 * Rand.rand(V1.size), maxiter=2000)
#    V1 = baye_V1.optimize(np.maximum(0.001,V1.flatten())/100., maxiter=5000)
V1 = np.reshape(V1,(NRGC,Nproj))
i += 1
Vs[i] = V1
plot_filters( V1[:12,:] )
#baye_U = posterior_dU ( [V2, V1, N_spikes , STA , STC], Nproj)
#U = baye_U.optimize(U.flatten(), maxiter=5000)
#U = np.reshape( U, (Nproj,Ncones))
#Us[i] = U
