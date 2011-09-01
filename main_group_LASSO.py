from   time  import time

import simulate_retina
reload(simulate_retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import numpy.random as Rand

import sparse_group_lasso as sgl
reload(sgl)

import numpy as np
import pylab as p
from scipy.linalg import schur, block_diag

import cPickle

# from matplotlib.ticker import *

from IPython.Debugger import Tracer; debug_here = Tracer()

def save():
    savefile = open('../../../Desktop/stimulus_stats.pyckle','w')
    del R['stimulus']
    cPickle.dump(R,savefile)
    savefile.close()

def load():
    savefile = open('../../../Desktop/stimulus_stats.pyckle','r')
    return cPickle.load(savefile)
    
############################
# Setting up simulated data

N_cells=[12,6,3]

V2 = 0.0
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

# Quantities of interest
N_filters = N_cells[1]*2
filters = np.concatenate(
    [simulate_retina.weights(sigma=n, shape=(N_filters,N_cells[0]))
    for n in [10.]] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
#R = load()
try:
    R = load()
except:
    R = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,1.],
                               average_me={'features':lambda x: NL(np.dot(filters,x))},
                               N_timebins = 5000000 )
    save()

#Nspikes_norm  = np.sum(np.sqrt(R['N_spikes']))/len(R['N_spikes'])
#
#dSTA = np.concatenate(
#            [np.sqrt(Nspikes/2)/Nspikes_norm * 
#             (STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis])
#            for Nspikes,STA in 
#            zip(R['N_spikes'],R['statistics']['features']['STA'])], axis=1)
            
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

def objective(v,R,y,predictors):
    total = float( np.sum([Nspikes for Nspikes in R['N_spikes']]) )
    Nspikes = R['N_spikes']/total
    dSTA  = np.concatenate(
            [STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis]
            for STA in R['statistics']['features']['STA']], axis=1)    
    direct = np.sum(v*Nspikes*( dSTA - 0.5*np.dot(R['statistics']['features']['cov'],v)))
    r,coeffs  = sgl.initialize_group_lasso(predictors, (np.sqrt(Nspikes)*y).flatten())
    group_weights = [0. for _ in predictors]
    weights       = [np.zeros(pp.shape[1]) for pp in predictors]
    return [ direct , sgl.objective(group_weights,weights,r,coeffs) ]

def sobjective(v,R):
    total = float( np.sum([Nspikes for Nspikes in R['N_spikes']]) )
    Nspikes = R['N_spikes']/total
    dSTA  = np.concatenate(
            [STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis]
            for STA in R['statistics']['features']['STA']], axis=1)
    D,Z = schur(R['statistics']['features']['cov']/2)
    DD  = np.diag(D)
    keep= DD>1e-10
    P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
    y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA ) /2
    
    direct = np.sum(v*Nspikes*( dSTA - 0.5*np.dot(R['statistics']['features']['cov'],v)))

#    print np.max( D - np.diag(np.diag(D)) )
#    print 'max( Z D Z.T - C ) ', \
#    np.max(np.abs(np.dot(Z,np.dot(D,Z.T)) - R['statistics']['features']['cov']/2))
#    print '2*dot(P.T,y) - dSTA ', np.max(np.abs( 2*np.dot(P.T,y) - dSTA ))

    return [ direct , np.sum( - (np.sqrt(Nspikes)*(y-np.dot(P,v)))**2 + 
                                (np.sqrt(Nspikes)*y)**2 ) ]


print sobjective(Rand.randn(12,3),R)
print sobjective(Rand.randn(12,3),R)
print sobjective(Rand.randn(12,3),R)
print sobjective(0*Rand.randn(12,3),R)

#V, iW = IRLS( y, P, x=0, disp_every=1000, lam=0.15, maxiter=100000 , 
#              ftol=1e-10, nonzero=1e-1)

start = time()
predictors    = [ block_diag(*[column*np.sqrt(Nspke) for Nspke in Nspikes]).T
                  for column in P.T]
                  
print
print
print objective(Rand.randn(12,3),R,y,predictors)
                  
                  
group_weights = [0.1 for _ in predictors]
weights       = [0.001*np.ones(pp.shape[1]) for pp in predictors]

r,coeffs  = sgl.initialize_group_lasso(predictors, (np.sqrt(Nspikes)*y).flatten())
print r
iterations = sgl.sparse_group_lasso(predictors, group_weights, weights, 
                                    r, coeffs, maxiter=10000, disp_every=10)
finish = time()
print iterations,'Iterations of sparse group LASSO in ',finish-start,' seconds for n: ',n
print 'infered x '
print sgl.inflate(coeffs)

def plot_filters(X,same_scale=True):
    for i in range(X.shape[0]):
        p.subplot(X.shape[0],1,i+1)
        p.plot(np.arange(X.shape[1]),X[i,:])
