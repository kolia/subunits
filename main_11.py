import simulate_retina
reload(simulate_retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import numpy as np
from scipy.linalg import schur

from LQuadLExP import posterior_dU, posterior_dV2, posterior_dV1, posterior_dUV1
       
import cPickle

from IPython.Debugger import Tracer; debug_here = Tracer()

def print_coeffs(V,precision=1e-2):
    lasti = -10
    for i,v in enumerate(V):
        if lasti == i-2: print
        if np.sum(np.abs(v))>precision:    
            print i,' : ', v
            lasti = i    
    
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
R = simulate_retina.LNLNP( nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,2.],
                           average_me={'features':lambda x: NL(np.dot(filters,x))},
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

V, iW = IRLS( y, P, x=0, disp_every=1000, lam=0.015, maxiter=1000000 , 
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
U        = filters[keepers,:].flatten()
V1       = V[keepers,:].T.flatten()
V2       = V2 * np.ones(Nproj)


#data     = [ V2, N_spikes , STA , STC ]
#baye_ARD = posterior_dUV1( data, Nproj )
#init_params = np.concatenate([ filters[keepers,:].flatten() , V[keepers,:].T.flatten() ])/1.5
#params = baye_ARD.optimize(init_params,maxiter=5000)

for i in range(2):
    baye_V1= posterior_dV1( [( U, V2, N_spikes , STA , STC)], Nproj)
    V1 = baye_V1.optimize(V1.flatten(), maxiter=5000)

    baye_U = posterior_dU ( [(V2, V1, N_spikes , STA , STC)], Nproj)
    U = baye_U.optimize(U.flatten(), maxiter=5000)
