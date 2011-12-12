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

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

#from IPython.Debugger import Tracer; debug_here = Tracer()

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

def place_cells( centers_in , centers_out , shapes ):
    r = numpy.zeros((len(centers_out),len(centers_in),len(shapes)))
    dx = numpy.array([[co[0]-ci[0] for co in centers_out] for ci in centers_in])
    dy = numpy.array([[co[1]-ci[1] for co in centers_out] for ci in centers_in])
#    dy = centers_out[i][1]-centers_in[j][1]
    for k in numpy.arange(r.shape[2]):
        r[:,:,k] = shapes[k](dx,dy).T
    return r
#    return numpy.fromfunction( (lambda i,j,k: 
#        shapes[k](centers_out[i][0]-centers_in[j][0], 
#                  centers_out[i][1]-centers_in[j][1])), 
#        (len(centers_out),len(centers_in),len(shapes)))
#    return numpy.hstack( [numpy.fromfunction( lambda i,j: 
#            shape(centers_out[i][0]-centers_in[j][0], 
#                  centers_out[i][1]-centers_in[j][1]), (len(centers_out),len(centers_in)) )
#            for shape in shapes])

import itertools
def hexagonal_2Dgrid( spacing=1. , field_size_x=10. , field_size_y=10. ):
    x1 = numpy.arange( spacing/2., field_size_x, spacing )
    x2 = numpy.arange( spacing   , field_size_x, spacing )
    d  = spacing*numpy.sqrt(3.)
    return list(itertools.chain.from_iterable(
                [[(x,y) for x in x1] + [(x,y+d/2.) for x in x2] \
                for y in numpy.arange(spacing/2., field_size_y, d)]))

@memory.cache
def gaussian2D_weights( centers_in , centers_out , sigma=1. ):
    def make_filter( x, y, sigma , centers_in ):
        f = numpy.array( [numpy.exp(-0.5*((i-x)**2.+(j-y)**2.)/sigma**2.) 
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

def mean_LQuad_from_STA_STC( U , V2 , STAs , STCs ):
    def meanLQuad( sta , stc ):
        Usta = numpy.dot(U,sta)
        return Usta + 0.5*V2*( numpy.diag(numpy.dot(numpy.dot(U,stc),U.T)) + 
                               numpy.sum( Usta**2 , axis=1 ) )
    return [meanLQuad( sta, stc ) for sta,stc in zip(STAs,STCs)]

#def cov_LQuad_from_STA_STC( U , V2 , STAs , STCs ):
#    def covLQuad( sta , stc ):
#        Usta = numpy.dot(U,sta)
#        return Usta + 0.5*V2*( numpy.diag(numpy.dot(numpy.dot(U,stc),U.T)) + 
#                               numpy.sum( Usta**2 , axis=1 ) )
#    return [covLQuad( sta, stc ) for sta,stc in zip(STAs,STCs)]

#def data_run( 
#    # list of cone    location coordinates
#    cones      = hexagonal_2Dgrid( spacing=1. , field_size_x=10. , field_size_y=10. ) ,
#    # list of subunit location coordinates
#    subunits   = hexagonal_2Dgrid( spacing=2. , field_size_x=10. , field_size_y=10. ) ,
#    # list of RGC     location coordinates
#    sigma      = 1.    , 
#    V2         = 0.5   , 
#    cones_mean = None  , 
#    cones_cov  = None  , 
#    STAs       = None  ,
#    STCs       = None  ):
#    U = gaussian2D_weights( cones , subunits , sigma=sigma )
#    statistics = {'features': {'STA' : mean_LQuad_from_STA_STC( U, V2, STAs, STCs) , 
#                               'mean': numpy.dot( U, cones_mean ) ,
#                               'cov' : cov_LQuad_from_STA_STC( U, V2, cones_mean, cones_cov) }}


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

def simulator_LNLEP( model , stimulus ,
    N_timebins     = 10000        ,  # number of stimulus time samples
    firing_rate     = 0.1          ): # RGC firing rate
    """Simulate spiking data for a Linear-Nonlinear-Linear-Nonlinear-Poisson model.
    """
    chunksize = 5000
    N = 0
    while N < N_timebins:
        X         = stimulus.generate( N_timebins=chunksize, dimension=model['U'].shape[1] )
        b         = model['nonlinearity'](numpy.dot(model['U'],X))  # subunit activations
        intensity = numpy.exp(numpy.dot(model['V'],b))                                # RGC activations
        constant  = numpy.log( model['V'].shape[0] * chunksize * firing_rate / numpy.sum(intensity) )
        intensity = intensity * numpy.exp(constant)        
        spikes    = R.poisson(intensity)
        N += chunksize
        yield {'stimulus':X , 'spikes':spikes}
    raise StopIteration()


from scipy.io import loadmat
def read_data( data_file='data.mat' ):
    datarun = loadmat(data_file)['data']
    data = {}
    data['spikes'] = datarun['spike_rate'][0][0]
    data['rgc_ids']        = datarun['rgc_ids'][0][0][0]    
    data['cone_weights']   = datarun['cone_weights'][0][0]    
    data['cone_types']     = datarun['cone_types'][0][0].tolist()    
    data['cone_locations'] = datarun['cone_locations'][0][0]    
    data['rgc_locations']  = numpy.array([d[0][0] for d in datarun['rgc_locations'][0][0]])    
    data['rgc_types']      = dict((d[0][0],d[1][0].tolist()) 
                                   for d in filter( lambda d : len( d[0] )>0 , [d[0][0] 
                                   for d in datarun['cell_types'][0][0][0]] ))
    return data

import sys
def read_stimulus( spikes, stimulus_pattern='cone_input_%d.mat', 
                   normalizer=None, skip_pattern=None):
    i = 0
    N_timebins = 0
    print 'Reading files: ',
    while 1:
        if skip_pattern is not None:
            if skip_pattern[0]>0:
                # skip every skip_pattern[0]-th i
                if numpy.mod(i,numpy.abs(skip_pattern[0])) == skip_pattern[1]:
                    i += 1
            else:
                # keep every skip_pattern[0]-th i only
                if numpy.mod(i,numpy.abs(skip_pattern[0])) != skip_pattern[1]:
                    i += numpy.abs(skip_pattern[0])-1
        data = {}
        try:
            data['stimulus'] = loadmat(stimulus_pattern % i)['data']
#            # Use normalizer to subtract mean and normalize variance
#            if normalizer is not None:
#                data['stimulus'] = data['stimulus'] - normalizer[0]
#                data['stimulus'] = numpy.dot( data['stimulus'] , normalizer[1] )
            data['stimulus'] = data['stimulus'].T
        except:
            raise StopIteration()
        data['spikes'] = spikes[:,N_timebins:N_timebins+data['stimulus'].shape[1]]
        N_timebins += data['stimulus'].shape[1]
        data['files_read'] = i
        print i,
        sys.stdout.flush()
        i += 1
        yield data
    
def simulate_data( spike_generator, stim_generator=None ):
    while 1:
        stimdata = stim_generator.next()
#        try:
        stimdata['spikes'] = spike_generator( stimdata )
        yield kb.extract( stimdata, ['stimulus','spikes'] )
#        except:
#            raise StopIteration
    
    
def one(x,**other):      return numpy.ones((1,x.shape[1]))
def identity(x,**other): return x
def square(x  ,**other): return numpy.dot(x,x.T)

def temporal_sum(x,**other): return numpy.sum(x,axis=-1)
    
def spike_triggered_sum(x,spikes=None,**other):
    if type(x) is type([]):
        return [ numpy.sum(y*s,1) for y,s in zip(x,spikes) ]
    else:
        return [ numpy.sum(x*s,1) for s in spikes ]

def spikes(x,spikes=None,**other): return spikes

def normalize(x,N_timebins=None,N_spikes=None,**other):
    if type(x) is type([]):
        return [ r/n for r,n in zip(x, N_spikes) ]
    else:
        return x/N_timebins

def covariance(x,N_timebins=None,mean=None,**other):
    return (x - numpy.outer(mean,mean))/N_timebins

def STC(x,N_spikes=None,STA=None,**other):
    return (x - numpy.outer(STA,STA))/N_spikes

localization = {'N_timebins' : [one,      temporal_sum,        identity  ],
                'N_spikes'   : [one,      spike_triggered_sum, identity  ],
                'mean'       : [identity, temporal_sum,        normalize ],
                'cov'        : [square,   identity,            covariance],
                'STA'        : [identity, spike_triggered_sum, normalize ]}

STAC = {'N_spikes'   : [one,      spike_triggered_sum, identity  ],
        'STA'        : [identity, spike_triggered_sum, normalize ],
        'STC'        : [square  , spike_triggered_sum, STC       ]}

def sparse_identity( x, sparse_index=None, **other ): 
    return [x[si,:] for si in sparse_index]

def sparse_STC( x, sparse_index=None, spikes=None, **other ):
    return [ numpy.dot( x[si,:], (x[si,:]*s).T ) 
             for s,si in zip(spikes,sparse_index) ]

# accumulate_statistics must be called with sparse_index kwarg for this
fit_U = {'N_spikes'   : [one,             spike_triggered_sum, identity ],
#        'mean'       : [identity,        temporal_sum,        normalize ],
#        'cov'        : [square,          identity,            covariance],
        'sparse_STA' : [sparse_identity, spike_triggered_sum, normalize],
        'sparse_STC' : [sparse_STC,      identity,            normalize]}

def accumulate_statistics( 
    data_generator = None         ,  # yields {'stimulus','spikes'}
    feature        = lambda x : x ,
    pipelines      = localization , **other):

    def process_chunk( datum ):
        datum.update( other )
        return dict((name,accumulate(transform(datum['stimulus'],**datum), **datum))
                     for name,[transform,accumulate,_] in pipelines.items())

    stats = data_generator.next()
    stats['stimulus'] = feature(stats['stimulus'])
    stats = process_chunk( stats )
    for stat in data_generator:
        stat['stimulus'] = feature(stat['stimulus'])
        stat = process_chunk( stat )
        for name,s in stats.items():
            if isinstance(s,type([])): 
                stats[name] = [ r+l for (r,l) in zip(s,stat[name])]
            else:
                stats[name] += stat[name]
    for name,s in stats.items():
        [_,_,postprocess] = pipelines[name]
        if postprocess is not None:
            stats[name] =  postprocess(s , **stats )
#            if isinstance(s,type([])):
#                stats[name] = [postprocess(ss, **stats ) for ss in s]
#            else:
#                stats[name] =  postprocess(s , **stats )
        else:
            del stats[name]
    return  stats

def run_LNLEP( model , stimulus ,
    sigma_stimulus = 1.           ,  # stimulus standard deviation
    firing_rate     = 0.1          ,  # RGC firing rate
    N_timebins     = 100000       ,  # number of time samples
    average_me     = {}           ): # calculate STA, STC, stim. avg 
#    picklable      = True         ): # and cov of these functions of the stimulus 

    average_me['stimulus'] = lambda x : x
    average_me['subunits'] = lambda x : model['nonlinearity'](numpy.dot(model['U'],x))

    data_generator = simulator_LNLEP( model, stimulus, 
                                      N_timebins=N_timebins, firing_rate=firing_rate)
    return accumulate_statistics( 
            data_generator=data_generator, N_timebins=N_timebins, average_me=average_me)