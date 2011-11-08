import numpy

import retina
reload(retina)

import IRLS
reload(IRLS)

import kolia_base as kb
reload(kb)

import optimize
reload(optimize)

import kolia_theano
reload(kolia_theano)

import retina
reload(retina)
from retina import place_cells

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import LQLEP_wBarrier, LQLEP, thetaM, \
                 linear_reparameterization, quadratic_V2_parameterization

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

from IPython.Debugger import Tracer; debug_here = Tracer()

data = retina.read_data()

cones             = data['cone_locations']
possible_subunits = cones

#V2 = 0.1
#def NL(x): return x + 0.5 * V2 * ( x ** 2 )

def get_rgc_type( i , data=data ):
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
        pylab.ylabel(get_rgc_type(i+j,data=data))

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
    return stats, sparse_index


rgc_type = 'off parasol'
keep = {'off midget' :  20, 'on midget' :  20, 
        'off parasol': 100, 'on parasol': 100}

ustats = {}
ustats[rgc_type], sparse_index = \
     fit_U_stats( rgc_type=rgc_type, keep=keep[rgc_type] )
NRGC = len( ustats[rgc_type]['rgc_index'] )


#def index( sequence = [] , f = lambda _: True ):
#    """Return the index of the first item in seq where f(item) == True."""
#    return next((i for i in xrange(len(sequence)) if f(sequence[i])), None)

def radial_piecewise_linear( nodes=[] , values=[] , default=0.):
    '''Returns a function which does linear interpolation between a sequence 
    of nodes. 
    nodes is an increasing sequence of n floats, 
    values is a list of n values at nodes.'''
    def f(dx,dy):
        x = numpy.sqrt(dx**2.+dy**2.)
        result = numpy.ones(x.shape) * default
        index  = result > default    # False
        for i,n in enumerate(nodes):
            index_i = x >= n
            if i>0: # len(nodes)-1:
                ind = ~index_i & index
                print numpy.sum(ind), ' distances between', nodes[i-1], ' and ', n 
                result[ ind ] = \
                      (values[i-1]*(n-x[ind])+values[i]*(x[ind]-nodes[i-1])) / (n-nodes[i-1])
            index = index_i
        return result
#        first = index( nodes , lambda node: node>x )
#        if first is not None and first>0:
#            return (values[first-1]*(nodes[first]-x)+values[first]*(x-nodes[first-1]))/ \
#                   (nodes[first]-nodes[first-1])
#        else: return default
    return f

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

del stats, fit_U_stats, localization

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
        V, iW = IRLS.IRLS( y, P, x=0, disp_every=2, lam=lam, maxiter=3 , 
                           ftol=1e-5, nonzero=1e-1, iw=iW)
        save({'V':V,'iW':iW},'Localizing_lam%.0e'%lam)
    return V, iW
    
V, iW = ARD( dSTA , Cin , lam=0.01 )

del dSTA, Cin, ARD

#print 'V'
#kb.print_sparse_rows( V, precision=1e-1 )

keepers = iW>1.e-1
#U       = filters[keepers,:]
V1      = V[keepers,:].T

inferred_locations = [possible_subunits[i] for i in numpy.nonzero(keepers)[0]]

import time

iterations = [0]
def callback( objective , params ):
#    fval = objective.f(params)
    result = objective.unflat(params)
    print ' Iter:', iterations[0]
    for name,x in result.items():
        if x.size>10:
            print name, 'min mean max: ',
            print numpy.min(x), numpy.mean(x), numpy.max(x)            
        else:
            print name, ': ',
            for p in x: print '%.2f' %p,
            print
    if numpy.remainder( iterations[0] , 5 ) == 0:
        pylab.clf()
        pylab.close('all')
        pylab.figure(1, figsize=(10,12))
        filename = '_'.join(sorted(result.keys()))+'_'+rgc_type
        save(result,filename)
        N = len(result.keys())
        i = 1
        for name,x in result.items():
            pylab.subplot(N,1,i)
            if x.size is nodes.size:
                pylab.plot(nodes,x)
                pylab.title(name+': '+', '.join([('%.2f'%y) for y in x]))
                pylab.xlabel('nodes: '+', '.join([('%.1f'%y) for y in nodes]))
            else:
                pylab.subplot(N,1,i)
                pylab.hist(x, bins=30)
            i += 1
        pylab.savefig('/Users/kolia/Desktop/%s.pdf'%filename, format='pdf')
#        p.savefig('/Users/kolia/Desktop/u.svg',format='svg')
    iterations[0] = iterations[0] + 1

def single_objective( param_templates ):
    arg     = set(['u','V2','STA','STC','v1','N_spike','T'])
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **linear_reparameterization())))
    print 'Simplifying single objective...'
    sys.stdout.flush()
    t0 = time.time()
    params = extract(vardict,param_templates.keys())
    args   = extract(vardict,arg-set(param_templates.keys()))
    outputs = { 'f':vardict['LQLEP'], 'barrier':vardict['barrier'] }
    obj = kolia_theano.Objective( params, param_templates, args, outputs,
                                  differentiate=['f'], mode='FAST_RUN' )
    t1 = time.time()
    print 'done simplifying single objective for', param_templates.keys(),
    print 'in ', t1-t0, ' sec.'
    sys.stdout.flush()
    return obj

def single_obj_uabc( param_templates ):
    arg     = set(['ub','b','uc','c','V2','STA','STC','v1','N_spike','T'])
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **quadratic_V2_parameterization())))
    print 'Simplifying single objective...'
    sys.stdout.flush()
    t0 = time.time()
    params = extract(vardict,param_templates.keys())
    args   = extract(vardict,arg-set(param_templates.keys()))
    outputs = { 'f':vardict['LQLEP_wPrior'], 'barrier':vardict['barrier'] }
    obj = kolia_theano.Objective( params, param_templates, args, outputs,
                                  differentiate=['f'], mode='FAST_RUN' )
    t1 = time.time()
    print 'done simplifying single objective for', param_templates.keys(),
    print 'in ', t1-t0, ' sec.'
    sys.stdout.flush()
    return obj


def objective( single_obj, run , knowns, v1 , T , index ):
    def objective_RGC_i( i ):
        data = { 'STA':run['sparse_STA'][i], 'STC':run['sparse_STC'][i], 
                 'v1': v1[i] , 'N_spike':float(run['N_spikes'][i]) , 
                 'T': T[:,index[i],:]}
        data.update(knowns)
        print '.',
        sys.stdout.flush()
        return single_obj.where(**data)
    objective = objective_RGC_i(0).with_callback(callback)
    sum_obj   = kb.Sum_objectives( [ objective_RGC_i(i) for i in range(NRGC)] , 
                                attributes=['f','df','barrier'])
    objective.f  = sum_obj.f
    objective.df = sum_obj.df
    objective.barrier = sum_obj.barrier
    return objective

#@memory.cache
def optimize_objective( single_obj, obj, init, gtol=1e-7 , maxiter=500 ):
    optimizer = optimize.optimizer( obj )
    params = optimizer( init_params=init, maxiter=maxiter, gtol=gtol )
    return single_obj.unflat( params )


nodes = numpy.array([0., 2., 3.5, 5., 10., 20., 40., 80., 150.])
#nodes = numpy.array([0., 2.3, 2.8, 3.3, 4., 5., 6., 7.5, 10., 14., 20., 40.])
#nodes = numpy.array([0., 3., 4., 5., 5.5, 6., 6.5, 7., 7.5, 8.5, 10., 12., 14., 20., 30., 50.])
#nodes = numpy.array([0., 3.2, 4.5, 5.5, 6.3, 6.9, 7.5, 8.3, 9., 10., 11., 12., 13., 14., 15., 20., 25., 35.])

value_matrix = numpy.eye(len(nodes))
shapes = [ radial_piecewise_linear(nodes,values) for values in value_matrix]

#init_u = numpy.array([3.05, 2.12, -2.14, 0.45, 0.56, -0.01, -0.18])
init_u = numpy.array([2.58, -0.7, -0.3, -0.2, 0.09, -0.03, 0., 0., 0.])
#init_u = numpy.exp(-0.5*(nodes**2))
##init_u = numpy.ones(nodes.shape)
#init_u = init_u/numpy.sqrt(numpy.sum(init_u**2.)) * 0.05

init_V2 = 0.5*numpy.ones(len(inferred_locations))

T  = place_cells( cones , inferred_locations , shapes )

import copy
@memory.cache
def ML( params, unknowns, objective_name=None, gtol=1e-7 , maxiter=150 ):
    print
    print 'FITTING ', unknowns
    sys.stdout.flush()
    knowns = copy.copy(params)
    for uk in unknowns: del knowns[uk]
    unknowns = extract(params,unknowns)
    if objective_name is 'ub':
        single_obj = single_obj_uabc( unknowns )
    else:
        single_obj = single_objective( unknowns )
    global_obj = objective( single_obj, ustats[rgc_type] , knowns , 
                            V1  , T , sparse_index )
    return optimize_objective( single_obj, global_obj, unknowns, 
                               gtol=gtol , maxiter=maxiter)

#params = {'u':init_u, 'ub':0.*init_u, 'V2':init_V2}
#params.update(ML(params, ['u','ub','V2'], 'ub'))
#params.update(ML(params, ['V2'], 'uabc', gtol=1e-7 , maxiter=30))
#for i in range(2):
#    params.update(ML(params, ['ua','ub'] , 'uabc', gtol=1e-7 , maxiter=9))
#    params.update(ML(params, ['V2'], 'uabc', gtol=1e-7 , maxiter=30))
#params.update(ML(params, ['ua','ub','V2'], 'uabc'))

params = {#'u':init_u, 
          'ub': init_u, 'b':numpy.ones_like(init_V2), 
          'uc':-0.01*init_u, 'c':numpy.ones_like(init_V2), 
          'V2':init_V2}
params.update(ML(params, ['ub','b','uc','c'],'ub'))
#params.update(ML(params, ['u'] , gtol=1e-1 , maxiter=9))
#params.update(ML(params, ['V2'], gtol=1e-1 , maxiter=30))
#params.update(ML(params, ['u'] , gtol=1e-1 , maxiter=8))
#params.update(ML(params, ['V2'], gtol=1e-1 , maxiter=30))
#params.update(ML(params, ['u'], gtol=1e-1 , maxiter=8))
