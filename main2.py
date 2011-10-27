import numpy

import retina
reload(retina)

import kolia_base as kb
reload(kb)

import optimize
reload(optimize)

import kolia_theano
reload(kolia_theano)

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import LQLEP_wBarrier, LQLEP, UV12, UVs , \
                        linear_reparameterization


# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

data = retina.read_data()

cones             = data['cone_locations']
possible_subunits = cones

from kolia_base import load
stats = load('Linear_localization')

def n_largest( values , keep=10 ):
    sorted_values = numpy.abs(values)
    sorted_values.sort()
    cutoff = sorted_values[-keep]
    return numpy.nonzero( numpy.abs(values) >= cutoff )[0]
    
@memory.cache
def fit_U_stats( rgc_type='off midget', keep=15, stats=stats ):
    which_rgc = [i for i,ind in enumerate(data['rgc_ids'])
                   if  ind   in data['rgc_types'][rgc_type]]
    spikes = data['spikes'][which_rgc,:]
    sparse_index = [n_largest( sta, keep=keep ) for sta in stats['STA']]
    stats['rgc_type']  = rgc_type
    stats['rgc_index'] = which_rgc
    stats.update( retina.accumulate_statistics( 
        data_generator = retina.read_stimulus( spikes ) ,
        feature        = lambda x : x                   ,
        pipelines      = retina.fit_U                    ,
        sparse_index   = sparse_index                   ))
    return stats

rgc_type = 'off midget'
keep = {'off midget' :  20, 'on midget' :  20, 
        'off parasol': 100, 'on parasol': 100}

ustats = {}
ustats[rgc_type] = fit_U_stats( rgc_type=rgc_type, keep=keep[rgc_type] )
NRGC = len( ustats[rgc_type]['rgc_index'] )


def index( sequence = [] , f = lambda _: True ):
    """Return the index of the first item in seq where f(item) == True."""
    return next((i for i in xrange(len(sequence)) if f(sequence[i])), None)

def radial_piecewise_linear( nodes=[] , values=[] , default=0.):
    '''Returns a function which does linear interpolation between a sequence 
    of nodes. 
    nodes is an increasing sequence of n floats, 
    values is a list of n values at nodes.'''
    def f(dx,dy):
        x = numpy.sqrt(dx**2.+dy**2.)
        first = index( nodes , lambda node: node>x )
        if first is not None and first>0:
            return (values[first-1]*(nodes[first]-x)+values[first]*(x-nodes[first-1]))/ \
                   (nodes[first]-nodes[first-1])
        else: return default
    return f

nodes = numpy.array([ 0. ,  1.  ,  1.5,  2. ,  2.5 ,  3. ,  3.5 ,  4. , 5.])

value_matrix = numpy.eye(len(nodes))
shapes = [ radial_piecewise_linear(nodes,values) for values in value_matrix]

init_u = numpy.exp(-0.5*(nodes**2))
#init_u = numpy.ones(nodes.shape)
init_u = init_u/numpy.sqrt(numpy.sum(init_u**2.))

import sys
sys.setrecursionlimit(10000)
#import cPickle
#f = file('obj_tu.pyckle','wb')
#cPickle.dump(obj_tu, f, protocol=cPickle.HIGHEST_PROTOCOL)

def extract(d, keys):
    return dict((k, d[k]) for k in keys if k in d)

import sys

@memory.cache
def objective_u( u=init_u ):
    inputs  = UV12( **linear_reparameterization() )
    env     = [LQLEP_wBarrier( **LQLEP( **uv ) ) for uv in UVs(NRGC)(**inputs)]    
    outputs = { 'f'      :sum([d['LQLEP'  ] for d in env]),
                'barrier':sum([d['barrier'] for d in env]) }
    params = extract( inputs, ['u'])
    args   = extract( inputs, ['STAs','STCs','V2','V1','N','N_spikes','T'])
    print 'Compiling Objective_u...'
    sys.stdout.flush()
    return kolia_theano.Objective( params, {'u': u }, args, outputs, 
                                   differentiate=['f'], mode='FAST_RUN' )
obj_u   = objective_u()

iterations = [0]
def callback( objective , params ):
#    fval = objective.f(params)
    print ' Iter:', iterations[0] # ,' Obj: ' , fval
    if numpy.remainder( iterations[0] , 3 ) == 0:
        result = objective.unflat(params)
#        if result.has_key('u'):
#            if numpy.remainder( iterations[0] , 7 ) == 0: pylab.close('all')
#            pylab.figure(1, figsize=(10,12))
#            pylab.plot(nodes,result['u'])
#            pylab.title('u params')
##            p.savefig('/Users/kolia/Desktop/u.svg',format='svg')
#            pylab.savefig('/Users/kolia/Desktop/u.pdf',format='pdf')

iterations[0] = iterations[0] + 1

def objU_data( run , v1 , v2 , T ):
   data = {'STAs':numpy.vstack(run['sparse_STA']) ,
           'STCs':numpy.vstack([stc[numpy.newaxis,:] for stc in run['sparse_STC']]), 
           'V2':v2 , 'V1': v1 , 'N':NRGC , 'N_spikes':run['N_spikes'] , 'T': T}
   return obj_u.where(**data).with_callback(callback)

@memory.cache
def optimize_u( v1, init_u, v2 , T, gtol=1e-7 , maxiter=500):
   optimizer = optimize.optimizer( objU_data( ustats[rgc_type] , v1 , v2 , T ) )
   # debug_here()
   params = optimizer(init_params={'u': init_u },maxiter=maxiter,gtol=gtol)
   opt_u = obj_u.unflat(params)
   return opt_u['u']
