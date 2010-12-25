from numpy import dot,exp,sum,log1p,std,eye
from numpy import arange,array,transpose,fromfunction
import numpy.random   as R
import scipy.linalg   as L
#import scipy.optimize as Opt
import theano.tensor  as Th
from theano import function


def simulate_LNLNP(
    N             = 10               ,  # number of cones, subunits & RGCs
    Sigma         = 1.               ,  # subunit & RGC connection width
    NL            = lambda x: exp(x) ,  # subunit & RGC static nonlinearity
    dyn_range     = 1.               ,  # range of arg to nonlinearity
    firing_rate   = 0.1              ,  # RGC firing rate
    T             = 100000           ): # number of time samples

    spatial  = fromfunction( lambda x: exp(-0.5 * ((x-N/2)/Sigma)**2) , N )
    spatial  = spatial / sum(spatial)
    U        = L.circulant( spatial )   # connections btw cones & subunits
    V        = U                        # connections btw subunits & RGCs

    X        = R.randn(N,T)             # cone activations

    UX       = dot(U,X)
    UX       = UX * log1p(dyn_range) * 0.5 / std(UX)
    b        = NL(UX)                   # subunit activations

    Vb       = dot(V,b)
    Vb       = Vb * log1p(dyn_range) * 0.5 / std(Vb)
    Y        = NL(Vb)                   # RGC activations
    Y        = Y * N * T * firing_rate / sum(Y)

    spikes   = R.poisson(Y)

    N_spikes = sum(spikes,1)            # total number of spikes
    STA      = [ sum(X*spikes[i,:],1) for i in arange(N) ]
    STA      = transpose( array(STA) ) / N_spikes

    STC      = [ dot( X-STA[:,[i]] , transpose( (X-STA[:,[i]]) * Y[i,:])) \
                 / N_spikes[i] - eye(N)                for i in arange(N) ]

    return (N_spikes,STA,STC)
    
    
def posterior_exp():
    priorG   = Th.dscalar()
    N_spikes = Th.dscalar()
    STA      = Th.dvector()
    STC      = Th.dmatrix()
    
    ll       = N_spikes / (N_spikes+priorG) * (STAB - meanb)