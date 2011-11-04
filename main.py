import numpy

import retina
reload(retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import kolia_base as kb
reload(kb)

import optimize
reload(optimize)

import kolia_theano
reload(kolia_theano)

import simulate_retina
reload(simulate_retina)
from simulate_retina import place_cells

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import LQLEP_wBarrier, LQLEP, thetaM, \
                        linear_reparameterization

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

data = retina.read_data()

cones             = data['cone_locations']
possible_subunits = cones

#V2 = 0.1
#def NL(x): return x + 0.5 * V2 * ( x ** 2 )

def rgc_type( i , data=data ):
    eyedee = data['rgc_ids'][i]
    for rgctype, inds in data['rgc_types'].items():
        if eyedee in inds: return rgctype
    return None

filters = retina.gaussian2D_weights( cones, possible_subunits, sigma=3. )

from kolia_base import save
save(data,'data_localization')


@memory.cache
def localization( filters ):
    return retina.accumulate_statistics( 
        data_generator = 
            retina.read_stimulus( data['spikes'],
                                 stimulus_pattern='cone_input_%d.mat' ) ,
        feature= lambda x : numpy.dot(filters,x))

#data['spikes'] = data['spikes'][:2,:]
stats = localization( filters )

import pylab
def hista(i , data=data, stats=stats, N=2):
    pylab.close('all')
    pylab.figure(1, figsize=(10,12))
    for j in range(N): 
        pylab.subplot(N,1,j+1)
        pylab.hist( stats['STA'][i+j] , bins=30)
        pylab.ylabel(rgc_type(i+j,data=data))

from kolia_base import save
save(stats,'Linear_localization')

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

from kolia_base import save
save(ustats,'ustats_0')

def extract(d, keys):
    return dict((k, d[k]) for k in keys if k in d)

dSTA  = numpy.concatenate(
        [STA[:,numpy.newaxis]-stats['mean'][:,numpy.newaxis] 
         for STA in stats['STA']], axis=1)

print 'calculated dSTA'
sys.stdout.flush()

Cin = stats['cov']/2

print 'calculated Cin'
sys.stdout.flush()

from scipy.linalg   import schur
@memory.cache
def ARD( dSTA , Cin , lam=0.0001 ):
    print 'Starting ARD of size ', Cin.shape,' with lambda=',lam
    sys.stdout.flush()
    D,Z = schur(Cin)
    print 'Schur decomposition completed'
    sys.stdout.flush()
    DD  = numpy.diag(D)
    keep= DD>1e-10
    P   =  (Z[:,keep] * numpy.sqrt(DD[keep])).T
    y   =  numpy.dot ( (Z[:,keep] * 1/numpy.sqrt(DD[keep])).T , dSTA ) / 2    
    iW = 1e-1
    for i in range(2):
        print 'Irlsing'
        sys.stdout.flush()
        V, iW = IRLS( y, P, x=0, disp_every=10, lam=lam, maxiter=5 , 
                      ftol=1e-5, nonzero=1e-1, iw=iW)
        save({'V':V,'iW':iW},'Localizing_lam%.0e'%lam)
    return V, iW
    
V, iW = ARD( dSTA , Cin , lam=0.009 )

print 'V'
kb.print_sparse_rows( V, precision=1e-1 )

keepers = numpy.array( [sum(abs(v))>3.e-1 for v in V] )
U        = filters[keepers,:]
V1       = V[keepers,:].T

inferred_locations = [possible_subunits[i] for i in numpy.nonzero(keepers)[0]]

import time

def single_objective_u( u=init_u):
    arg     = ['u','STA','STC','V2','v1','N_spike','T']
    result  = ['LQLEP_wPrior','dLQLEP_wPrior','barrier']
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **linear_reparameterization())))
    print 'Simplifying single objective_u...'
    sys.stdout.flush()
    t0 = time.time()
    params = extract(vardict,arg)
    args   = extract(vardict,result)
    outputs = { 'f':vardict['LQLEP'  ], 'barrier':vardict['barrier'] }
    kolia_theano.Objective( params, {'u': u }, args, outputs, 
                                   differentiate=['f'], mode='FAST_RUN' )
    inputs, outputs = kolia_theano.simplify( extract(vardict,arg), extract(vardict,result) )
    t1 = time.time()
    print 'done simplifying single objective_u in ', t1-t0, ' sec.'
    sys.stdout.flush()
    return inputs, outputs
single_objective = single_objective_u()

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

def objU_data( run , v1 , v2 , T , i ):
   data = { 'STA':run['sparse_STA'][i], 'STC':run['sparse_STC'][i], 
            'V2':v2 , 'V1': v1[i] , 'N_spike':run['N_spikes'][i] , 'T': T}
   return single_objective.where(**data).with_callback(callback)


@memory.cache
def optimize_u( v1, init_u, v2 , T, gtol=1e-7 , maxiter=500):
    objective = kb.Sum_objective( [objU_data( ustats[rgc_type] , v1 , v2 , T , i )
                                   for i in range(NRGC)] )
    optimizer = optimize.optimizer( objective )
    # debug_here()
    params = optimizer(init_params={'u': init_u },maxiter=maxiter,gtol=gtol)
    opt_u = single_objective.unflat(params)
    return opt_u['u']

V2 = 0.5
T  = place_cells( cones , inferred_locations , shapes )
opt_u  = optimize_u( V1, init_u, V2*numpy.ones(V1.shape[1]) , T , gtol=1e-6 , maxiter=500)