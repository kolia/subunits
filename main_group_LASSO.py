from   time  import time

import simulate_retina
reload(simulate_retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import sparse_group_lasso as sgl
reload(sgl)

import numpy as np
import pylab as p
from scipy.linalg import schur

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

V2 = 0.5
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

# Quantities of interest
N_filters = N_cells[1]*2
filters = np.concatenate(
    [simulate_retina.weights(sigma=n, shape=(N_filters,N_cells[0]))
    for n in [2.,3.,5.]] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
#R = load()
try:
    R = load()
except:
    R = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[3.,1.],
                               average_me={'features':lambda x: NL(np.dot(filters,x))},
                               N_timebins = 5000000 )
    save()

dSTA = np.concatenate(
            [np.sqrt(Nspikes)*
            (STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis])
            for Nspikes,STA in 
            zip(R['N_spikes'],R['statistics']['features']['STA'])], axis=1)/2
D,Z = schur(R['statistics']['features']['cov']/2)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA )

V, iW = IRLS( y, P, x=0, disp_every=1000, lam=100., maxiter=10000000 , 
              ftol=1e-7, nonzero=1e-1)

#start = time()
#predictors    = [P[:,s*i:np.minimum(s*(i+1),P.shape[1])] for i in range(np.floor(m/s))]
#group_weights = [5. for _ in predictors]
#weights       = [5.*np.ones(p.shape[1]) for p in predictors]
#r,coeffs  = sgl.initialize_group_lasso(predictors, y)
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
