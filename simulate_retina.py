"""
Generate simulated data.
@author: kolia
"""

from numpy  import dot,sin,cos,exp,std,sum,newaxis,sqrt,cov
from numpy  import arange,transpose,fromfunction
import numpy.random   as R

def weights(shape=(10,10), sigma=1):
    U = fromfunction( lambda i,j: \
        exp(sigma * cos(j-(shape[0]*i/shape[1])) ), shape)
    return U / sqrt(sum(U*U,axis=1))[:,newaxis]


def LNLNP(
    sigma_stimulus = 1.           ,  # stimulus standard deviation
    nonlinearity   = sin          ,  # subunit nonlinearity
    N_cells        = [10,5,3]     ,  # number of cones, subunits & RGCs
    sigma_spatial  = 1.5          ,  # subunit & RGC connection width
    firing_rate     = 0.1          ,  # RGC firing rate
    N_timebins     = 1000000      ,  # number of time samples
    average_me     = {}           ):  # calculate STA, STC, stim. avg 
                                     # and cov of these functions of the stimulus
    """Simulate spiking data for a Linear-Nonlinear-Linear-Nonlinear-Poisson model.
    """
    seed     = R.get_state()
    def stimulus():
        R.set_state(seed)
        return sigma_stimulus*R.randn(N_cells[0],N_timebins)
    X        = stimulus()                   # cone activations

    U = weights(sigma=sigma_spatial,shape=(N_cells[1],N_cells[0]))
    V = weights(sigma=sigma_spatial,shape=(N_cells[2],N_cells[1]))

    b        = nonlinearity(dot(U,X))      # subunit activations
    Y        = exp(dot(V,b))               # RGC activations
    Y        = Y * N_cells[2] * N_timebins * firing_rate / sum(Y)

    print 'std( X ) = ', std(X)
    print 'std( U X ) = ', std(dot(U,X))
    print 'std( V b ) = ', std(dot(V,b))

    spikes = R.poisson(Y)
    N_spikes = sum(spikes,1)

    average_me['stimulus'] = lambda x : x
    average_me['subunits'] = lambda x : nonlinearity(dot(U,x))

    def statistics(x):
        d = {}
        d['mean'] = sum(x,axis=1)/N_timebins
        d['cov']  = cov(x)
        d['STA']  = [ sum(x*spikes[i,:],1) / N_spikes[i] for i in arange(N_cells[2]) ]
        d['STC']  = \
            [ dot( x-d['STA'][i][:,newaxis], transpose((x-d['STA'][i][:,newaxis])*spikes[i,:])) \
            / N_spikes[i]               for i in arange(N_cells[2]) ]
        return d
    statistics = dict( (name,statistics(average_me[name](X))) for name in average_me.keys())

    result = locals()
    del result['X']
    del result['Y']
    del result['b']
    del result['seed']
    del result['average_me']
    return result