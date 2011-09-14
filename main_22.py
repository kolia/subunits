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
from scipy.linalg   import schur
import numpy.random as Rand

import pylab        as p
from   matplotlib.ticker import *

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
    for i in range(X.shape[0]-1,-1,-1):
        ax = p.subplot(X.shape[0]*2,1,i*2+1)
        p.plot(np.arange(X.shape[1]),X[i,:])
        ax.yaxis.set_major_locator( MaxNLocator(nbins=1) )

#def save(result,name):
#    savefile = open("../../../Desktop/%s.pyckle" % name,'w')
#    cPickle.dump(result,savefile)
#    savefile.close()

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

testR = simulate( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,2.],
                  N_timebins = 90000 )

            
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
#N_spikes = R['N_spikes']
#STA      = R['statistics']['stimulus']['STA']
#STC      = R['statistics']['stimulus']['STC']
Nproj    = np.sum(keepers)
U        = filters[keepers,:]
V1       = V[keepers,:].T
V2       = V2 * np.ones(Nproj)

iterations = [0]
def callback( objective , params ):
    fval = objective.f(params)
    fbar = objective.barrier(params)
    print 'Obj: ' , fval , '  barrier: ', fbar
    if np.remainder( iterations[0] , 30 ) == 0:
        result = objective.unflat(params)
        if np.remainder( iterations[0] , 100 ) == 0: p.close('all')
        p.figure(1)
        plot_filters(result['V1'])
        p.title('V1 filters')
        p.savefig('/Users/kolia/Desktop/V1.svg',format='svg')
        p.savefig('/Users/kolia/Desktop/V1.pdf',format='pdf')
        p.figure(2)
        plot_filters(result['U'][:10,:])
        p.title('1st ten U filters')
        p.savefig('/Users/kolia/Desktop/U.svg',format='svg')
        p.savefig('/Users/kolia/Desktop/U.pdf',format='pdf')
    iterations[0] = iterations[0] + 1
#    if not np.isfinite( fval ):
#        eigsM = objective.eigsM( params )
#        invM  = objective.invM( params )
#        debug_here()


#true = {'U' : R['U'] , 'V1': R['V'] }

##def objective_U(): 
##    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
##    targets = kolia_theano.reparameterize(targets,UVs(NRGC))
##    return kolia_theano.Objective( init_params={'U': U }, differentiate=['f'], 
##                                        callback=callback, **targets )


def objective_U():
    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
    targets = kolia_theano.reparameterize(targets,UVs(NRGC))
#    list_targets= kolia_theano.reparameterize({'eigsM':eigsM, 'invM':invM },UVs(NRGC),
#                                               reducer=lambda r,x: r + [x], zero=[])
#    targets.update( list_targets )
    
    return    kolia_theano.Objective( init_params={'U': U }, differentiate=['f'], 
                              callback=callback, **targets )    
#obj_U   = objective_U()
#
##@memory.cache
#def optimize_U( v1,init_U, V2=V2  ):
#    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) ,
#            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
#            'V2':V2 , 'V1': v1 , 'N':NRGC , 'N_spikes':R['N_spikes'] }
#    
#    optimizer = optimize.optimizer( obj_U.where(**data) )
#    params = optimizer(init_params={'U': init_U },maxiter=5000)
#    opt_U = obj_U.unflat(params)
#    return opt_U['U']


def objective_UV1():
    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
    targets = kolia_theano.reparameterize(targets,UVs(NRGC))    
    return kolia_theano.Objective( init_params={'V1': V1 , 'U': U }, differentiate=['f'], callback=callback, **targets )
obj_UV1  = objective_UV1()

@memory.cache
def optimize_UV1( v1,u,V2=V2 ):
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) ,
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':V2 , 'N':NRGC , 'N_spikes':R['N_spikes'] }
    optimizer = optimize.optimizer( obj_UV1.where(**data) )    
    params = optimizer(init_params={'V1': v1 , 'U': u },maxiter=3000,gtol=1.1e-7)
    opt_U = obj_UV1.unflat(params)
    return opt_U

print 'Optimizing U,V1'
opt_UV1 = optimize_UV1( 0.001 + 0.001 * Rand.rand(*V1.shape), U, V2=V2 )
#opt_UV1 = optimize_UV1( opt_UV1['V1'], opt_UV1['U'], V2=V2 )

#opt_U = U
#for i in range(2):
#    print 'Optimizing V1'
#    opt_V1 = optimize_V1( opt_U)
#    print 'Optimizing U'
#    opt_U  = optimize_U( opt_V1, opt_U )

testdata = {'STAs':np.vstack(testR['statistics']['stimulus']['STA']) ,
            'STCs':np.vstack([stc[np.newaxis,:] for stc in testR['statistics']['stimulus']['STC']]), 
            'V2':V2 , 'N':NRGC , 'N_spikes':testR['N_spikes'] }
test_objective = obj_UV1.where(**data)

print 'Unregularized ARD + alternating full model ll:', test_objective.f( {'V1': opt_V1 , 'U': opt_U } )

def objective_V1():
    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
    targets = kolia_theano.reparameterize(targets,UVs(NRGC))
#    list_targets= kolia_theano.reparameterize({'eigsM':eigsM, 'invM':invM },UVs(NRGC),
#                                               reducer=lambda r,x: r + [x], zero=[])
#    targets.update( list_targets )
    
    return    kolia_theano.Objective( init_params={'V1': V1 }, differentiate=['f'], 
                              callback=callback, **targets )    
obj_V1  = objective_V1()

@memory.cache
def optimize_V1( u,V2=V2 ):
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) , 
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':V2*np.ones(Nproj) , 'U': u , 'N':NRGC , 'N_spikes':R['N_spikes'] }
    optimizer = optimize.optimizer( obj_V1.where(**data) )    
    init_params = {'V1': 0.001 + 0.001 * Rand.rand(*V1.shape) }
    params = optimizer(init_params=init_params,maxiter=5000)
    opt_V1 = obj_V1.unflat(params)
    return opt_V1['V1']

glm = optimize_V1( np.eye(Ncones), V2=np.zeros(V2.shape) )
testdata['U'] = np.eye(Ncones)
print 'Unregularized GLM (STA) ll:', obj_V1.where(**testdata).f( glm.flatten() )