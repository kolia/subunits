import simulate_retina
reload(simulate_retina)

import numpy as np
import pylab as p

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
D = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells ,
                           average_me={'features':lambda x: np.exp(np.dot(filters,x))},
                           N_timebins = 10000 )

# True parameters
true = [{ 'theta' : np.dot( D['U'].T , D['V'][i,:] ) , \
          'M' : 0.1*np.dot( D['U'].T * D['V'][i,:] , D['U'] ) } for i in range(N_cells[2])]

