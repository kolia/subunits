"""
Generate simulated data.
@author: kolia
"""

from numpy  import dot,sin,exp,std,eye,sum,newaxis,sqrt
from numpy  import arange,transpose,fromfunction
import numpy.random   as R
import scipy.linalg   as L

def LNLNP(
    NL            = sin              ,  # subunit nonlinearity
    N             = 10               ,  # number of cones, subunits & RGCs
    Sigma         = 1.               ,  # subunit & RGC connection width
    firing_rate   = 0.1              ,  # RGC firing rate
    T             = 100000           ): # number of time samples
    """Simulate spiking data for a Linear-Nonlinear-Linear-Nonlinear-Poisson model.
    Then use this data to calculate the STAs and STCs.
    """
	
    spatial  = fromfunction( lambda x: exp(-0.5 * ((x-N/2)/Sigma)**2) , (N,) )
    U        = L.circulant( spatial )   # connections btw cones & subunits
    V        = U                        # connections btw subunits & RGCs

    U = U[0:10:2,:]
    V = V[0:10:2,0:10:2]
    NRGC = V.shape[0]

    U = U / sqrt(sum(U*U,axis=1))[:,newaxis]
    V = V / sqrt(sum(V*V,axis=1))[:,newaxis]

    X        = R.randn(N,T)                 # cone activations
#    b        = sin(dot(U,X))                # subunit activations
#    b        = exp(dot(U,X))                # subunit activations
#    b        = dot(U,X)                     # subunit activations
#    b        = dot(U,X) + dot(U,X) ** 2     # subunit activations
    b        = NL(dot(U,X))                 # subunit activations
    Y        = exp(dot(V,b))                # RGC activations
    Y        = Y * NRGC * T * firing_rate / sum(Y)

    print 'std( X ) = ', std(X)
    print 'std( U X ) = ', std(dot(U,X))
    print 'std( V b ) = ', std(dot(V,b))

    spikes   = R.poisson(Y)
    N_spikes = sum(spikes,1)
	
    STA = [ sum(X*spikes[i,:],1) / N_spikes[i] for i in arange(NRGC) ]

    STC = [ dot( X-STA[i][:,newaxis] , transpose((X-STA[i][:,newaxis])*Y[i,:])) \
                 / N_spikes[i] - eye(N)                for i in arange(NRGC) ]

    return ((N_spikes,STA,STC),U,V)
