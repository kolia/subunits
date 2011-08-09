import simulate_retina
reload(simulate_retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import numpy as np
import pylab as p
from scipy.linalg import schur

# from matplotlib.ticker import *

from IPython.Debugger import Tracer; debug_here = Tracer()

############################
# Setting up simulated data

N_cells=[12,6,3]

V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

# Quantities of interest
N_filters = N_cells[1]*2
filters = np.concatenate(
    [simulate_retina.weights(sigma=2.+n*0., shape=(N_filters,N_cells[0]))
    for n in range(1)] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
R = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells , sigma_spatial=2.,
                           average_me={'features':lambda x: NL(np.dot(filters,x))},
                           N_timebins = 100000 )

dSTA = np.concatenate(
            [STA[:,np.newaxis] - R['statistics']['features']['mean'][:,np.newaxis]
            for STA in R['statistics']['features']['STA']], axis=1)
D,Z = schur(R['statistics']['features']['cov'])
DD  = np.diag(D)
keep= DD>1e-6
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA )

V, iW = IRLS( y, P, x=0, disp_every=1000, lam=0.001, maxiter=10000000 , ftol=1e-10, nonzero=1e-3)

def plot_filters(X,same_scale=True):
    for i in range(X.shape[0]):
        p.subplot(X.shape[0],1,i+1)
        p.plot(np.arange(X.shape[1]),X[i,:])