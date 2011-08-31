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

Nspikes_norm  = np.sum(np.sqrt(R['N_spikes']))/len(R['N_spikes'])

dSTA = np.concatenate(
            [np.sqrt(Nspikes/2)/Nspikes_norm * 
             (STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis])
            for Nspikes,STA in 
            zip(R['N_spikes'],R['statistics']['features']['STA'])], axis=1)
D,Z = schur(R['statistics']['features']['cov']/2)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA )

def objective(v,R,y,P):
    total = np.sum([Nspikes for Nspikes in R['N_spikes']])
    Nspikes_norm  = np.sum(np.sqrt(R['N_spikes']))/len(R['N_spikes'])
    P = block_diag(*[P*np.sqrt(Nspikes/2)/Nspikes_norm for Nspikes in R['N_spikes']])
    y = y.flatten()
    dSTA  = np.concatenate(
            [Nspikes/total*
            (STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis])
            for Nspikes,STA in 
            zip(R['N_spikes'],R['statistics']['features']['STA'])], axis=1)
    direct = np.sum(v*( dSTA - 0.5*np.dot(R['statistics']['features']['cov'],v)))
    one    = np.ones(v.size)
    norm   = direct / np.sum( (y-np.dot(P,one))**2 - y**2 )
    return [ direct , norm*np.sum( (y-np.dot(P,v.flatten()))**2 - y**2 ) ]

def sobjective(v,R):
    v = v[:,0]

    dSTA  = R['statistics']['features']['STA'][0][:,np.newaxis]- \
            R['statistics']['features']['mean'][:,np.newaxis]

    direct = np.sum(v*( dSTA - 0.5*np.dot(R['statistics']['features']['cov'],v)))

    D,Z = schur(R['statistics']['features']['cov']/2)

#    print (np.dot(Z,np.dot(D,Z.T)) - R['statistics']['features']['cov']/2)

    D = np.diag(D)
#    print D

    P   =  (Z * np.sqrt(D)).T
    y   =  np.dot ( (Z * 1/np.sqrt(D)).T , dSTA ) / 2

    return [ direct , np.sum( (y-np.dot(P,v))**2 - y**2 ) ]


print sobjective(Rand.randn(12,3),R)
print sobjective(Rand.randn(12,3),R)
print sobjective(Rand.randn(12,3),R)
print sobjective(0*Rand.randn(12,3),R)
print objective(Rand.randn(12,3),R,y,P)

#V, iW = IRLS( y, P, x=0, disp_every=1000, lam=0.15, maxiter=100000 , 
#              ftol=1e-10, nonzero=1e-1)

#start = time()
#predictors    = [ block_diag(*[column*np.sqrt(Nspikes/2)/Nspikes_norm
#                               for Nspikes in R['N_spikes']]).T
#                  for column in P.T]
#group_weights = [0.1 for _ in predictors]
#weights       = [0.01*np.ones(pp.shape[1]) for pp in predictors]
#
#r,coeffs  = sgl.initialize_group_lasso(predictors, y.flatten())
#print r
#iterations = sgl.sparse_group_lasso(predictors, group_weights, weights, 
#                                    r, coeffs, maxiter=10000, disp_every=10)
#finish = time()
#print iterations,'Iterations of sparse group LASSO in ',finish-start,' seconds for n: ',n
#print 'infered x ',coeffs


def plot_filters(X,same_scale=True):
    for i in range(X.shape[0]):
        p.subplot(X.shape[0],1,i+1)
        p.plot(np.arange(X.shape[1]),X[i,:])
