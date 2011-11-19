import numpy

import retina
reload(retina)

import IRLS
reload(IRLS)

import kolia_base as kb
reload(kb)

#import klbfgsb
#reload(klbfgsb)

import optimize
reload(optimize)

import kolia_theano
reload(kolia_theano)

import retina
reload(retina)

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import LQLEP_wBarrier, LQLEP, thetaM, \
                 linear_reparameterization

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

import ipdb
#from IPython.Debugger import Tracer; debug_here = Tracer()

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


from scipy.linalg   import schur
import sys
sys.setrecursionlimit(10000)


@memory.cache
def localization( filters , which ):
    stats = retina.accumulate_statistics( 
        data_generator = 
            retina.read_stimulus( data['spikes'][which],
                                 stimulus_pattern='cone_input_%d.mat' ,
                                 skip_pattern=(5,0) ) ,
        feature= lambda x : numpy.dot(filters,x))
    stats['dSTA']  = numpy.concatenate(
        [STA[:,numpy.newaxis]-stats['mean'][:,numpy.newaxis] 
         for STA in stats['STA']], axis=1)
    print 'calculated dSTA'
    sys.stdout.flush()
    stats['D'], stats['Z'] = schur(stats['cov'])
    del stats['cov']
    return stats

which = range(50)

rgc_type = 'off parasol'
#rgc_type = 'on midget'
keep = {'off midget' :  20, 'on midget' :  20, 
        'off parasol': 100, 'on parasol': 100}


#data['spikes'] = data['spikes'][:2,:]
stats = localization( filters, which )

del filters

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
def fit_U_stats( rgc_type='off midget', keep=15, stats=stats, skip_pattern=None, which=which ):
    which_rgc = [i for i,ind in enumerate(data['rgc_ids'])
                   if  ind   in data['rgc_types'][rgc_type] and i in which]
#                   and data['rgc_locations'][i][1]>170 ]
    print 'Fitting stats for',len(which_rgc),'RGCs'
    # Used to subtract mean and normalize by covariance
#    normalizer   = (stats['mean'],numpy.dot(stats['Z'],
#                                  numpy.dot(stats['D']**(-0.5),stats['Z'].T)))
    spikes       = data['spikes'][which_rgc,:]
    stats['cov'] = numpy.dot( stats['Z'],numpy.dot(stats['D'],stats['Z'].T) )
    diagcov      = numpy.diag( stats['cov'] )
    stats[ 'sparse_index'] = [n_largest( stats['STA'][i]/diagcov, keep=keep ) 
                              for i in which_rgc]
    stats['subunit_index'] = [n_largest( stats['STA'][i]/diagcov, keep=keep ) 
                              for i in which_rgc]
    stats['rgc_type']  = rgc_type
    stats['rgc_index'] = which_rgc
    stats.update( retina.accumulate_statistics( 
        data_generator = retina.read_stimulus( spikes , 
#                                               normalizer=normalizer,
                                               skip_pattern=skip_pattern) ,
        feature        = lambda x : x                   ,
        pipelines      = retina.fit_U                    ,
        sparse_index   = stats['sparse_index']          ))
    return stats

train_stats = {}
train_stats[rgc_type] = \
     fit_U_stats( rgc_type=rgc_type, keep=keep[rgc_type], skip_pattern=(5,0), which=which )
NRGC = len( train_stats[rgc_type]['rgc_index'] )

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

from kolia_base import save
save(train_stats,'train_stats')

def extract(d, keys):
    return dict((k, d[k]) for k in keys if k in d)


@memory.cache
def ARD( stats , lam=0.02 ):
    print 'Starting ARD of size ', stats['Z'].shape,' with lambda=',lam
    sys.stdout.flush()
    D,Z = stats['D']/2 , stats['Z']
    print 'Schur decomposition completed'
    sys.stdout.flush()
    DD  = numpy.diag(D)
    keep= DD>1e-10
    P   =  (Z[:,keep] * numpy.sqrt(DD[keep])).T
    y   =  numpy.dot ( (Z[:,keep] * 1/numpy.sqrt(DD[keep])).T , stats['dSTA'] ) / 2
    iW = 1e-1
    for i in range(1):
        print 'Irlsing'
        sys.stdout.flush()
        V, iW = IRLS.IRLS( y, P, x=0, disp_every=1, lam=lam, maxiter=2 , 
                           ftol=1e-5, nonzero=1e-1, iw=iW)
        save({'V':V,'iW':iW},'Localizing_lam%.0e'%lam)
    return V, iW
    
lam = 0.0
V, iW = ARD( stats , lam=lam )

sv1 = numpy.vstack([V[index,i] for i,index in 
                    enumerate(train_stats[rgc_type]['subunit_index'])])

print 'sv1.shape:',sv1.shape

#print 'V'
#kb.print_sparse_rows( V, precision=1e-1 )

keepers = iW > -numpy.Inf #iW>1.e-1
##U       = filters[keepers,:]
V1      = V[keepers,:].T

del V, stats, localization, ARD, iW

inferred_locations = [possible_subunits[i] for i in numpy.nonzero(keepers)[0]]

import time

maxiter = 1000

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
    if numpy.remainder( iterations[0] , 20 ) == 99:
        filename = '_'.join(sorted(result.keys()))+'_'+rgc_type+'_lam'+str(int(100*lam))+'_'+str(int(maxiter))+'iters'
        pylab.clf()
        pylab.close('all')
        pylab.figure(2, figsize=(12,10))
        a1_V2   = kb.values_to_color( result['V2'] , (0.8,0.,0.) )
        a_V2 = [(x,1.-x,0.,a) for x,_,_,a in a1_V2]
#        ipdb.set_trace()
        kb.plot_circles( sizes=1.5, offsets=inferred_locations, linewidth=0.,
                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
        pylab.savefig('/Users/kolia/Desktop/V2_%s.pdf'%filename, format='pdf')
        pylab.figure(1, figsize=(10,12))
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
    arg     = set(['u','V2','STA','STC','v1','N_spike','T','Cm1'])
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **linear_reparameterization())))
    print 'Simplifying single objective...'
    sys.stdout.flush()
    t0 = time.time()
    params = extract(vardict,param_templates.keys())
    args   = extract(vardict,arg-set(param_templates.keys()))
    outputs = { 'f':vardict['LQLEP_wPrior'], 'LL':vardict['LQLEP'],
                'barrier':vardict['barrier'] }
    obj = kolia_theano.Objective( params, param_templates, args, outputs,
                                  differentiate=['f'], mode='FAST_RUN' )
    t1 = time.time()
    print 'done simplifying single objective for', param_templates.keys(),
    print 'in ', t1-t0, ' sec.'
    sys.stdout.flush()
    return obj

from functools import partial

#def split_params( params , i ):
#    rt = copy.deepcopy( params )
#    for name,param in params.items():
#        if name is 'V1':
#            del rt['V1']
#            rt['v1'] = param[i]
#        if name is 'sparse_STA':
#            del rt['sparse_STA']
#            rt['STA'] = param[i]
#        if name is 'sparse_STC':
#            del rt['sparse_STC']
#            rt['STC'] = param[i]
#        if name is 'N_spikes':
#            del rt['N_spikes']
#            rt['N_spike'] = float(param[i])
#        if params.has_key('sparse_index'):
#            index = params['sparse_index'][i]
#            if name is 'T':
#                rt['T'] = param[:,index,:]
#            if name is 'cov':
#                del rt['cov']
#                rt['Cm1'] = numpy.linalg.inv( param[numpy.ix_(index,index)] )
##                rt['Cm1'] = numpy.diag(numpy.diag(numpy.linalg.inv(
##                                       C[numpy.ix_(index[i],index[i])])))
#    if rt.has_key('sparse_index'): del rt['sparse_index']
#    return  rt
 
#def fuse_params( into , params , i , index=None):
#    for name,param in params.items():
#        if name is 'v1':
#            into['V1'][i,:] = param
#        else:
#            into[name] += param
#    return into

import copy
def split_params( params , i , indices ):
    rt = copy.deepcopy( params )
    for name,param in params.items():
        if name is 'sparse_STA':
            del rt['sparse_STA']
            rt['STA'] = param[i]
        elif name is 'sparse_STC':
            del rt['sparse_STC']
            rt['STC'] = param[i]
        elif name is 'N_spikes':
            del rt['N_spikes']
            rt['N_spike'] = float(param[i])
        if name is 'V1':
            del rt['V1']
            rt['v1'] = param[i]
        if indices.has_key('sparse_index'):
            index = indices['sparse_index'][i]
            if name is 'sv1':
                del rt['sv1']
                rt['v1'] = param[i]
            elif name is 'cov':
                del rt['cov']
                rt['Cm1'] = numpy.linalg.inv( param[numpy.ix_(index,index)] )
            if indices.has_key('subunit_index'):
                sindex = indices['subunit_index'][i]
                if name is 'V2':
                    rt['V2'] = param[sindex]
                elif name is 'T':
                    rt['T'] = param[numpy.ix_(index,sindex,range(param.shape[2]))]
            else:
                if name is 'T':
                    rt['T'] = param[:,index,:]
#                rt['Cm1'] = numpy.diag(numpy.diag(numpy.linalg.inv(
#                                       C[numpy.ix_(index[i],index[i])])))
#    if rt.has_key( 'sparse_index'): del rt[ 'sparse_index']
#    if rt.has_key('subunit_index'): del rt['subunit_index']
    return  rt

def fuse_params( into , params , i , indices=None ):
    if indices is not None and indices.has_key('subunit_index'):
        for name,param in params.items():
            if name is 'v1':
                into['sv1'][i] = param
            elif name is 'V2':
                into['V2'][index] = param
            else:
                into[name] += param
    else:
        for name,param in params.items():
            if name is 'v1':
                into['V1'][i,:] = param
            else:
                into[name] += param
    return into

def _sum_objectives( objectives, global_objective, attribute, X ):
    x = global_objective.unflat(X)
    indices = global_objective.indices
    result  = getattr(objectives[0],attribute)(split_params(x,0,indices))
    if result.size>1:
        result = fuse_params( kb.zeros_like(x) , objectives[0].unflat( result ) , 
                              0 , indices )
        for i,obj in enumerate(objectives[1:]):
            param = obj.unflat( getattr(obj,attribute)(split_params(x,i,indices)) )
            result = fuse_params( result , param , i , indices )
        return global_objective.flat( result )
    else:
        return result + sum(getattr(obj,attribute)(split_params(x,i,indices))
                              for i,obj in enumerate(objectives[1:]))
#        for i,obj in enumerate(objectives)[1:]:
#            result += getattr(obj,attribute)(split_params(x,i))
#        return result
#    return Params.flat( result )

#    x = Params.unflat(X)
#    first = objectives[0].unflat( getattr(objectives[0],attribute)(split_params(x,0)) )
#    if first.size>1:
#        result = fuse_params( kb.zeros_like(x) , first , 0 )
#    else:
#        result = first
#    for i,obj in enumerate(objectives)[1:]:
#        param = obj.unflat( getattr(obj,attribute)(split_params(x,i)) )
#        ipdb.set_trace()
#        result = fuse_params( result , param , i )
#    return Params.flat( result )


def global_objective( params=None, unknowns=None , indices=None ):
    print
    print 'Preparing objective for unknowns: ', unknowns
    sys.stdout.flush()
    t0 = time.time()
    for name in unknowns.keys():
        if params.has_key(name): del params[name]
    single_obj = single_objective( split_params( unknowns, 0 , indices ) )
    objectives = [single_obj.where({},**split_params(params,i,indices)) 
                                   for i in range(NRGC)]
    arg     = set(['u','V2','sparse_STA','sparse_STC','V1','N_spikes','T','cov'])
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **linear_reparameterization())))
    symbolic_params = extract(vardict,unknowns.keys())
    args   = extract(vardict,arg-set(unknowns.keys()))
    objective = kolia_theano.Objective( symbolic_params, unknowns, args, {})
    objective = objective.where({'indices':indices},**params).with_callback(callback)
#    ipdb.set_trace()
    for fname in ['f','df','barrier']:
        setattr(objective,fname,partial(_sum_objectives, objectives, objective, fname))
        setattr(getattr(objective,fname),'__name__',fname)
    print '... done preparing objective in', time.time()-t0,'sec.'

    print
    print 'Testing global objective:'
    test_param = objective.flat(unknowns)
    print 'discrepancy when flattening and unflattening:', \
          [(n,numpy.sum((v-unknowns[n])**2.)) 
            for n,v in objective.unflat(test_param).items()]
    print 'f:', objective.f(test_param),objective.f(unknowns)
    print 'df:', objective.unflat(objective.df(test_param))
    print 'barrier:', objective.barrier(test_param),objective.barrier(unknowns)
    print

    return objective

#@memory.cache
def optimize_objective( obj, init, gtol=1e-7 , maxiter=500 , optimizer=optimize ):
    optimizer = optimizer.optimizer( obj )
    print 'Starting optimization:'
    sys.stdout.flush()
    t0 = time.time()
    params = optimizer( init_params=init, maxiter=maxiter, gtol=gtol )
    print '... done optimizing in', time.time()-t0,'sec.'
    return obj.unflat( params )


nodes = numpy.array([0., 2., 7., 12., 20., 40., 250.])
#nodes = numpy.array([0., 2.3, 2.8, 3.3, 4., 5., 6., 7.5, 10., 14., 20., 40.])
#nodes = numpy.array([0., 3., 4., 5., 5.5, 6., 6.5, 7., 7.5, 8.5, 10., 12., 14., 20., 30., 50.])
#nodes = numpy.array([0., 3.2, 4.5, 5.5, 6.3, 6.9, 7.5, 8.3, 9., 10., 11., 12., 13., 14., 15., 20., 25., 35.])

value_matrix = numpy.eye(len(nodes))
shapes = [ radial_piecewise_linear(nodes,values) for values in value_matrix]

#init_u = numpy.array([3.05, 2.12, -2.14, 0.45, 0.56, -0.01, -0.18])
init_u = numpy.array([0.4, 0.2, 0.05, 0.02, 0.01, 0.005, 0.])
#init_u = numpy.exp(-0.5*(nodes**2))
##init_u = numpy.ones(nodes.shape)
#init_u = init_u/numpy.sqrt(numpy.sum(init_u**2.)) * 0.05

init_V2 = 0.*numpy.ones(len(inferred_locations))


#params = {'u':init_u, 'ub':0.*init_u, 'V2':init_V2}
#params.update(ML(params, ['u','ub','V2'], 'ub'))
#params.update(ML(params, ['V2'], 'uabc', gtol=1e-7 , maxiter=30))
#for i in range(2):
#    params.update(ML(params, ['ua','ub'] , 'uabc', gtol=1e-7 , maxiter=9))
#    params.update(ML(params, ['V2'], 'uabc', gtol=1e-7 , maxiter=30))
#params.update(ML(params, ['ua','ub','V2'], 'uabc'))


def all_variables( run , **others ):
    result = extract( run, ['sparse_STA','sparse_STC','sparse_index',
                            'N_spikes','cov','subunit_index'] )
    result.update( others )
    return result

indices = extract( train_stats[rgc_type], ['sparse_index', 'subunit_index'] )

#params = all_variables( train_stats[rgc_type], u=init_u, V2=init_V2, V1=V1, 
#                        T=retina.place_cells( cones , inferred_locations , shapes ) )
#            , 'b':0.01*init_V2, 'ub':numpy.zeros_like(init_u) 

#unknowns     = {'u':init_u, 'V2':init_V2} 
#global_obj = global_objective( params = all_variables( 
#                        train_stats[rgc_type], sv1=sv1, 
#                        T=retina.place_cells( cones , inferred_locations , shapes ) ) ,
#                               unknowns = unknowns, indices = indices)
#global_obj = global_objective( params = all_variables( 
#                        train_stats[rgc_type], sv1=sv1, 
#                        T=retina.place_cells( cones , inferred_locations , shapes ) ) ,
#                               unknowns = unknowns)
                               
import gc
gc.collect()

#params = optimize_objective( global_obj, unknowns, gtol=1e-7 , maxiter=maxiter)

params = {'sv1':sv1, 'V2':init_V2}
global_obj = global_objective( params = all_variables( 
                        train_stats[rgc_type], sv1=sv1,
                        T=retina.place_cells( cones , inferred_locations , shapes ),
                        **{'u':init_u}) ,
                    unknowns = params, indices = indices)
params.update(optimize_objective( global_obj, params, gtol=1e-7 , maxiter=maxiter))


params.update({'u':init_u})
global_obj = global_objective( params = all_variables( 
                        train_stats[rgc_type], sv1=sv1,
                        T=retina.place_cells( cones , inferred_locations , shapes )) ,
                    unknowns = params, indices = indices)
opt0 = optimize_objective( global_obj, params, gtol=1e-7 , maxiter=maxiter)
                                  
#
#unknowns = ['u','V2','V1']
#global_obj = global_objective( params, unknowns )
#import gc
#gc.collect()
#params.update(optimize_objective( global_obj, extract(params,unknowns), 
#                                  gtol=1e-7 , maxiter=1000, optimizer=klbfgsb))

#params.update(ML(params, ['u'] , gtol=1e-1 , maxiter=9))
#params.update(ML(params, ['V2'], gtol=1e-1 , maxiter=30))
#params.update(ML(params, ['u'] , gtol=1e-1 , maxiter=8))
#params.update(ML(params, ['V2'], gtol=1e-1 , maxiter=30))
#params.update(ML(params, ['u'], gtol=1e-1 , maxiter=8))
