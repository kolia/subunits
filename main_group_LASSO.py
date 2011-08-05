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
N_filters = 10
filters = np.concatenate(
    [simulate_retina.weights(sigma=0.5+n*0.2, shape=(N_filters,N_cells[0]))
    for n in range(10)] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
R = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells ,
                           average_me={'features':lambda x: np.exp(np.dot(filters,x))},
                           N_timebins = 100000 )

dSTA = np.concatenate(
            [STA[:,np.newaxis] - R['statistics']['features']['mean'][:,np.newaxis]
            for STA in R['statistics']['features']['STA']], axis=1)
D,Z = schur(R['statistics']['features']['cov'])
DD  = np.diag(D)
keep= DD>1e-6
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA )

V, iW = IRLS( y, P, x=0, disp_every=500, lam=0.4, maxiter=500000 , ftol=1e-8)