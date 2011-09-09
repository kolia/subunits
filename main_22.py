import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, UVs , eig_barrier

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
import pylab        as p
from scipy.linalg   import schur
import numpy.random as Rand

import cPickle

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
#V2       = V2 * np.ones(Nproj)


def callback( objective , params ):
    print 'Obj: ' , objective.f(params) , '  barrier: ', objective.barrier(params)


#true = {'U' : R['U'] , 'V1': R['V'] }
true = {'V1': V1 }
data = {'STAs':np.vstack(STA) , 'STCs':np.vstack([stc[np.newaxis,:] for stc in STC]), 
        'V2':V2*np.ones(Nproj) , 'U': U , 'N':NRGC , 'N_spikes':N_spikes }

targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }

targets = kolia_theano.reparameterize(targets,UVs(NRGC))

objective = kolia_theano.Objective( init_params=true, differentiate=['f'], 
                          callback=callback, **targets )

optimizer = optimize.optimizer( objective.where(**data) )

#trupar = true
#for i in range(2):
#    trupar = optimizer(init_params=trupar)
#trupar = objective.unflat(trupar)


#init_params = {'U' : 0.0001+0.05*Rand.random(size=R['U'].shape ) ,
#               'V1': 0.0001+0.05*Rand.random(size=R['V'].shape) }

init_params = {'V1': 0.01 * Rand.rand(*V1.shape) }

params = init_params
for i in range(10):
    params = optimizer(init_params=params)
params = objective.unflat(params)

