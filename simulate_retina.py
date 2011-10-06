"""
Generate simulated data.
@author: kolia
"""
import kolia_base as kb
import numpy
#from numpy  import dot,sin,cos,exp,sum,newaxis,sqrt,cov,ceil,minimum
#from numpy  import arange,transpose,fromfunction,pi,log,vstack,array
#from numpy.linalg import pinv
import numpy.random   as R

import pylab

from IPython.Debugger import Tracer; debug_here = Tracer()

def ring_weights(shape=(10,10), sigma=1., offset_in=0., offset_out=0.):
    U = numpy.fromfunction( lambda i,j: \
        numpy.exp(sigma * numpy.cos((j+offset_in-
        (shape[1]*(i++offset_out)/shape[0]))*2*numpy.pi/shape[1]) ), shape)
    return U / numpy.sqrt(numpy.sum(U*U,axis=1))[:,numpy.newaxis]

def LNLEP_ring_model(
    nonlinearity   = numpy.sin    ,  # subunit nonlinearity
    N_cells        = [10,5,3]     ,  # number of cones, subunits & RGCs
    sigma_spatial  = [2., 1.]     ): # subunit & RGC connection width
    """Linear-Nonlinear-Linear-Exponential-Poisson model on a ring.
    """
    U = ring_weights(sigma=sigma_spatial[0],shape=(N_cells[1],N_cells[0]))
    V = ring_weights(sigma=sigma_spatial[1],shape=(N_cells[2],N_cells[1]))
    return locals()

import itertools
def hexagonal_2Dgrid( spacing=1. , field_size_x=10. , field_size_y=10. ):
    x1 = numpy.arange( spacing/2., field_size_x, spacing )
    x2 = numpy.arange( spacing   , field_size_x, spacing )
    d  = spacing*numpy.sqrt(3.)
    return list(itertools.chain.from_iterable(
                [[(x,y) for x in x1] + [(x,y+d/2.) for x in x2] \
                for y in numpy.arange(spacing/2., field_size_y, d)]))

def gaussian2D_weights( centers_in , centers_out , sigma=1. ):
    def make_filter( x, y, sigma , centers_in ):
        f = numpy.array( [numpy.exp(-0.5*((i-x)**2.+(j-y)**2.)/sigma) 
                            for (i,j) in centers_in] )
        return f / numpy.sqrt(numpy.sum(f**2.))
    return numpy.vstack( [make_filter( x, y, sigma , centers_in ) 
                            for (x,y) in centers_out] )
    
def LNLEP_gaussian2D_model(
    # list of cone    location coordinates
    cones    = hexagonal_2Dgrid( spacing=1. , field_size_x=10. , field_size_y=10. ) ,
    # list of subunit location coordinates
    subunits = hexagonal_2Dgrid( spacing=2. , field_size_x=10. , field_size_y=10. ) ,
    # list of RGC     location coordinates
    RGCs     = hexagonal_2Dgrid( spacing=4. , field_size_x=10. , field_size_y=10. ) ,
    nonlinearity   = numpy.sin    ,  # subunit nonlinearity
    sigma_spatial  = [2., 4.]     ,  **other ): # subunit & RGC connection widths
    """Linear-Nonlinear-Linear-Exponential-Poisson model on a ring.
    """
    U = gaussian2D_weights( cones    , subunits , sigma=sigma_spatial[0] )
    V = gaussian2D_weights( subunits , RGCs     , sigma=sigma_spatial[1] )
    return locals()

class Stimulus(object):
    """Stimulus generator that keeps a list of the seeds and N_timebins 
    used at each invocation, so that the same stimulus can be generated 
    without storing the generated stimulus. Instances should be picklable."""
    def __init__( self, generator ):
        self.N_timebin_list = []
        self.seed_list      = []
        self.generator      = generator
    
    def generate( self, *args, **kwargs ):
        if kwargs.has_key('seed'):  R.set_state(kwargs['seed'])
        kwargs = kb.format_kwargs(self.generator,kwargs)
        if not kwargs.has_key('N_timebins'):
            raise NameError('stimulus generator requires an N_timebins argument!')
        self.N_timebin_list += [kwargs['N_timebins']]
        self.seed_list      += [R.get_state()]
        return self.generator( *args, **kwargs )

def white_gaussian( sigma=1., dimension=1, N_timebins = 100000 ):
    """Independent zero mean sigma std gaussian.
    """
    return sigma*R.randn(dimension,N_timebins)
    

def very_nonlinear_LN_stimulus( model , output_sigma = 1. , weight_matrix = 'U' ):
    iU    = model['U'].T
#    iU    = pinv( (model['U'].T / numpy.sqrt(numpy.sum( model['U']**2 , axis=1 ))).T )
    iU    = output_sigma * iU / numpy.sqrt(numpy.sum( iU**2 , axis=0 ))
    N_timebin_list = []
    seed_list      = []
    def generate( N_timebins = 100000 , seed  = R.get_state() ,
                  N_timebin_list = N_timebin_list , seed_list = seed_list ):
        R.set_state(seed)
        N_timebin_list += [N_timebins]
        seed_list      += [seed]
        return numpy.dot( iU , R.randn(iU.shape[1],N_timebins) )
    return locals()    

def run_LNLEP_chunk( model , stimulus ,
    N_timebins     = 10000        ,  # number of stimulus time samples
    firing_rate     = 0.1          ,  # RGC firing rate
    keep           = frozenset([]),  # keep ['X', 'b', 'intensity', 'spikes']?
    average_me     = {}           ): # calculate STA, STC, stim. avg 
                                     # and cov of these functions of the stimulus
    """Simulate spiking data for a Linear-Nonlinear-Linear-Nonlinear-Poisson model.
    """
    X         = stimulus.generate( N_timebins=N_timebins, dimension=model['U'].shape[1] )
    b         = model['nonlinearity'](numpy.dot(model['U'],X))  # subunit activations
    intensity = numpy.exp(numpy.dot(model['V'],b))                                # RGC activations
    constant  = numpy.log( model['V'].shape[0] * N_timebins * firing_rate / numpy.sum(intensity) )
    intensity = intensity * numpy.exp(constant)
    
    spikes = R.poisson(intensity)
    N_spikes = numpy.sum(spikes,1)

    average_me['stimulus'] = lambda x : x
    average_me['subunits'] = lambda x : model['nonlinearity'](numpy.dot(model['U'],x))

    def statistics(x):
        d = {}
        d['mean'] = numpy.sum(x,axis=1)/N_timebins
        d['cov']  = numpy.cov(x)
        d['STA']  = [ numpy.sum(x*spikes[i,:],1) / N_spikes[i] 
                      for i in numpy.arange(model['V'].shape[0]) ]
        d['STC']  = \
            [ numpy.dot( x-d['STA'][i][:,numpy.newaxis], 
                        numpy.transpose((x-d['STA'][i][:,numpy.newaxis])*spikes[i,:])) \
            / N_spikes[i]               for i in numpy.arange(model['V'].shape[0]) ]
        return d
    statistics = dict( (name,statistics(average_me[name](X))) for name in average_me.keys())

    result = locals()
    for thing in frozenset(['X','b','intensity','spikes']).difference(keep):
        del result[thing]
    del result['average_me']
    return result

def run_LNLEP( model , stimulus ,
    sigma_stimulus = 1.           ,  # stimulus standard deviation
    firing_rate     = 0.1          ,  # RGC firing rate
    N_timebins     = 100000       ,  # number of time samples
    keep           = frozenset([]),  # keep ['X', 'b', 'intensity', 'spikes']?
    average_me     = {}           ): # calculate STA, STC, stim. avg 
#    picklable      = True         ): # and cov of these functions of the stimulus 
                                     
    chunksize = 50000.
    timebins = [numpy.minimum(N_timebins-i*chunksize, chunksize) for i,x in 
                    enumerate(range(int(numpy.ceil(N_timebins/chunksize))))]
    for _ in timebins:  print '-',
    print
    result = run_LNLEP_chunk( model , stimulus , N_timebins=timebins[0], keep=keep,
                              firing_rate=firing_rate, average_me=average_me)
    result['stimulus'] = [result['stimulus']]
    for Nt in timebins[1:]:
        ll = run_LNLEP_chunk( model , stimulus, N_timebins=Nt, keep=keep,
                              firing_rate=firing_rate, average_me=average_me)
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
#    if not picklable:
#        result['stimulus']   = lambda : \
#            concatenate([ll['stimulus']() for ll in result['stimulus']])
#    else:
#        del result['stimulus']
    return result