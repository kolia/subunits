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
    for i in numpy.arange(r.shape[0]):
        for j in numpy.arange(r.shape[1]):
            for k in numpy.arange(r.shape[2]):
                r[i,j,k] = shapes[k](centers_out[i][0]-centers_in[j][0], 
                                     centers_out[i][1]-centers_in[j][1])
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
    
def make_statistics(x,spikes):
    N_spikes         = numpy.sum(spikes,1)
    NRGC, N_timebins = spikes.shape
    d = {}
    d['mean'] = numpy.sum(x,axis=1)/N_timebins
    d['cov']  = numpy.cov(x)
    d['STA']  = [ numpy.sum(x*spikes[i,:],1) / N_spikes[i] 
                  for i in numpy.arange(NRGC) ]
    d['STC']  = \
        [ numpy.dot( x-d['STA'][i][:,numpy.newaxis], 
                    numpy.transpose((x-d['STA'][i][:,numpy.newaxis])*spikes[i,:])) \
        / N_spikes[i]               for i in numpy.arange(NRGC) ]
    return d

def simulator_LNLEP( model , stimulus ,
    N_timebins     = 10000        ,  # number of stimulus time samples
    firing_rate     = 0.1          ): # RGC firing rate
    """Simulate spiking data for a Linear-Nonlinear-Linear-Nonlinear-Poisson model.
    """
    while 1:
        X         = stimulus.generate( N_timebins=N_timebins, dimension=model['U'].shape[1] )
        b         = model['nonlinearity'](numpy.dot(model['U'],X))  # subunit activations
        intensity = numpy.exp(numpy.dot(model['V'],b))                                # RGC activations
        constant  = numpy.log( model['V'].shape[0] * N_timebins * firing_rate / numpy.sum(intensity) )
        intensity = intensity * numpy.exp(constant)        
        spikes    = R.poisson(intensity)
        yield {'stimulus':X , 'spikes':spikes}

def read_dataset( stimulus_pattern='stimulus_%d.mat', data_file='data.mat'):
    from scipy.io import loadmat
    data = loadmat(data_file)
    data = data['data']    
    spikes = data['spike_rate'][0][0]
    del data['spike_rate']
    data['rgc_ids']        = data['rgc_ids'][0][0][0]    
    data['cone_weights']   = data['cone_weights'][0][0]    
    data['cone_types']     = data['cone_types'][0][0].tolist()    
    data['cone_locations'] = data['cone_locations'][0][0]    
    data['rgc_locations']  = numpy.array([d[0][0] for d in data['rgc_locations'][0][0]])    
    data['rgc_types']      = dict((d[0][0],d[1][0].tolist()) 
                                   for d in filter( lambda d : len( d[0] )>0 , [d[0][0] 
                                   for d in data['cell_types'][0][0][0]] ))
    try:
        i = 0
        N_timebins = 0
        while 1:
            data['stimulus'] = loadmat(stimulus_pattern % i)['cone_input'].T
            data['spikes'] = spikes[N_timebins:N_timebins+data['stimulus'].shape[1]]
            N_timebins += data['stimulus'].shape[1]
            i += 1
            yield data
    except:
        raise StopIteration()        

def accumulate_statistics( 
    data_generator = None         ,  # a function which yields {'stimulus','spikes'} 
    N_timebins     = 100000       ,  # number of time samples requested
    average_me     = {}           ): # calculate STA, STC, stim. avg of this
    generate = data_generator()

    def get_chunk():
        chunk = generate.next()
        chunk['N_timebins'] = chunk['stimulus'].shape[1]
        chunk['N_spikes']   = numpy.sum( chunk['spikes'], 1)
        chunk['statistics'] = dict( (name,make_statistics(
                                    average_me[name](chunk['stimulus']),chunk['spikes'])) 
                                    for name in average_me.keys())
        return chunk

    result = get_chunk()        
    while result['N_timebins'] < N_timebins:
        ll = get_chunk()
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
        result['N_spikes']   = result['N_spikes']   + ll['N_spikes']
    return result

def run_LNLEP( model , stimulus ,
    sigma_stimulus = 1.           ,  # stimulus standard deviation
    firing_rate     = 0.1          ,  # RGC firing rate
    N_timebins     = 100000       ,  # number of time samples
    average_me     = {}           ): # calculate STA, STC, stim. avg 
#    picklable      = True         ): # and cov of these functions of the stimulus 

    average_me['stimulus'] = lambda x : x
    average_me['subunits'] = lambda x : model['nonlinearity'](numpy.dot(model['U'],x))

    data_generator = lambda : simulator_LNLEP( model , stimulus , firing_rate=firing_rate)
    return accumulate_statistics( 
            data_generator=data_generator, N_timebins=N_timebins, average_me=average_me)