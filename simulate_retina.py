"""
Generate simulated data.
@author: kolia
"""

from numpy  import dot,sin,cos,exp,sum,newaxis,sqrt,cov,ceil,minimum
from numpy  import arange,transpose,fromfunction,pi,mean,concatenate
import numpy.random   as R

from IPython.Debugger import Tracer; debug_here = Tracer()

def weights(shape=(10,10), sigma=1):
    U = fromfunction( lambda i,j: \
        exp(sigma * cos((j-(shape[1]*i/shape[0]))*2*pi/shape[1]) ), shape)
    return U / sqrt(sum(U*U,axis=1))[:,newaxis]


def LNLNP_chunk(
    sigma_stimulus = 1.           ,  # stimulus standard deviation
    nonlinearity   = sin          ,  # subunit nonlinearity
    N_cells        = [10,5,3]     ,  # number of cones, subunits & RGCs
    sigma_spatial  = [2., 1.]     ,  # subunit & RGC connection width
    firing_rate     = 0.1          ,  # RGC firing rate
    N_timebins     = 100000       ,  # number of time samples
    average_me     = {}           ):  # calculate STA, STC, stim. avg 
                                     # and cov of these functions of the stimulus
    """Simulate spiking data for a Linear-Nonlinear-Linear-Nonlinear-Poisson model.
    """
    seed     = R.get_state()
    def stimulus():
        R.set_state(seed)
        return sigma_stimulus*R.randn(N_cells[0],N_timebins)
    X        = stimulus()                   # cone activations

    U = weights(sigma=sigma_spatial[0],shape=(N_cells[1],N_cells[0]))
    V = weights(sigma=sigma_spatial[1],shape=(N_cells[2],N_cells[1]))

    b        = nonlinearity(dot(U,X))      # subunit activations
    spikes   = exp(dot(V,b))               # RGC activations
    spikes   = spikes * N_cells[2] * N_timebins * firing_rate / sum(spikes)
    
    spikes = R.poisson(spikes)
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
    del result['spikes']
    del result['b']
    del result['seed']
    del result['average_me']
    return result

def LNLNP(
    sigma_stimulus = 1.           ,  # stimulus standard deviation
    nonlinearity   = sin          ,  # subunit nonlinearity
    N_cells        = [10,5,3]     ,  # number of cones, subunits & RGCs
    sigma_spatial  = [2., 1.]     ,  # subunit & RGC connection width
    firing_rate     = 0.3          ,  # RGC firing rate
    N_timebins     = 100000       ,  # number of time samples
    average_me     = {}           ,  # calculate STA, STC, stim. avg 
    picklable      = True         ): # and cov of these functions of the stimulus 
                                     
    chunksize = 50000.
    timebins = [minimum(N_timebins-i*chunksize, chunksize) for i,x in 
                    enumerate(range(int(ceil(N_timebins/chunksize))))]
    for _ in timebins:  print '-',
    print
    result = LNLNP_chunk(sigma_stimulus, nonlinearity, N_cells, sigma_spatial, 
                         firing_rate, N_timebins=timebins[0], average_me=average_me)
    result['stimulus'] = [result['stimulus']]
    for Nt in timebins[1:]:
        ll = LNLNP_chunk(sigma_stimulus, nonlinearity, N_cells, sigma_spatial, 
                         firing_rate, N_timebins=Nt, average_me=average_me)
    #        result['spikes']   = concatenate([l['spikes'] for l in lnp], axis=1)
    
        for topname,topd in result['statistics'].items():
            for name,d in result['statistics'][topname].items():
                if isinstance(result['statistics'][topname][name],type([])):
                    for i,(r,l) in enumerate(zip(result['statistics'][topname][name],
                                                 ll['statistics'][topname][name])):
                        result['statistics'][topname][name][i] = \
                            (r*result['N_timebins'] + l*ll['N_timebins'])/ \
                            (result['N_timebins'] + ll['N_timebins'])                        
                else:
                    result['statistics'][topname][name] = \
                        (result['statistics'][topname][name]*result['N_timebins'] + 
                             ll['statistics'][topname][name]*    ll['N_timebins'])/ \
                        (result['N_timebins'] + ll['N_timebins'])
        result['N_timebins'] = result['N_timebins'] + ll['N_timebins']
        result['stimulus']   = result['stimulus']   + [ll['stimulus']]
        result['N_spikes']   = result['N_spikes']   + ll['N_spikes']
        print '+',
    if not picklable:
        result['stimulus']   = lambda : \
            concatenate([ll['stimulus']() for ll in result['stimulus']])
    else:
        del result['stimulus']
        del result['nonlinearity']
    return result