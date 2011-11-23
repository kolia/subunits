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
from   QuadPoiss import LQLEP_positiveV1, LQLEP_wBarrier, LQLEP, LNP, \
                        thetaM, linear_reparameterization, \
                        LQLEP_positive_u

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

from kolia_base import save
save(data,'data_localization')

from scipy.linalg   import schur
import sys
sys.setrecursionlimit(10000)


@memory.cache
def localization( filters ):
    stats = retina.accumulate_statistics( 
        data_generator = 
            retina.read_stimulus( data['spikes'],
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

#data['spikes'] = data['spikes'][:2,:]
stats = localization( retina.gaussian2D_weights( cones, possible_subunits, sigma=3. ) )

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
def fit_U_stats( rgc_type='off midget', stats=stats, skip_pattern=None ):
    keep = {'off midget' :  30, 'on midget' :  30, 
            'off parasol': 120, 'on parasol': 120}
    keep = keep[rgc_type]
    which_rgc = [i for i,ind in enumerate(data['rgc_ids'])
                   if  ind   in data['rgc_types'][rgc_type]] #and i in which]
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

#rgc_type = 'off parasol'
rgc_type = 'on midget'
#rgc_type = 'on parasol'
#rgc_type = 'off midget'

def train_stats(rgc_type):
    return fit_U_stats( rgc_type=rgc_type, skip_pattern=(5,0) )

def test_stats(rgc_type):
    return fit_U_stats( rgc_type=rgc_type, skip_pattern=(-5,0) )

NRGC = len( train_stats(rgc_type)['rgc_index'] )

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
                    enumerate(train_stats(rgc_type)['subunit_index'])])

print 'sv1.shape:',sv1.shape

#print 'V'
#kb.print_sparse_rows( V, precision=1e-1 )

keepers = iW > -numpy.Inf #iW>1.e-1
##U       = filters[keepers,:]
V1      = V[keepers,:].T

del V, stats, localization, ARD, iW

inferred_locations = [possible_subunits[i] for i in numpy.nonzero(keepers)[0]]

import time


def single_objective( param_templates , vardict , outputs ):
    arg     = set(['u','V2','STA','STC','v1','N_spike','T','Cm1','C'])
    print 'Simplifying single objective...'
    sys.stdout.flush()
    t0 = time.time()
    params = extract(vardict,param_templates.keys())
    args   = extract(vardict,arg-set(param_templates.keys()))
    obj = kolia_theano.Objective( params, param_templates, args, outputs,
                                  differentiate=['f'], mode='FAST_RUN' )
    t1 = time.time()
    print 'done simplifying single objective for', param_templates.keys(),
    print 'in ', t1-t0, ' sec.'
    sys.stdout.flush()
    return obj

from functools import partial

import copy
def split_params( params , i , indices ):
    rt = copy.deepcopy( params )
    for name,param in params.items():
        if name == 'sparse_STA':
            del rt['sparse_STA']
            rt['STA'] = param[i]
        elif name == 'sparse_STC':
            del rt['sparse_STC']
            rt['STC'] = param[i]
        elif name == 'N_spikes':
            del rt['N_spikes']
            rt['N_spike'] = float(param[i]/sum(params['N_spikes']))
        if name == 'V1':
            del rt['V1']
            rt['v1'] = param[i]
        if indices.has_key('sparse_index'):
            index = indices['sparse_index'][i]
            if name == 'sv1':
                del rt['sv1']
                rt['v1'] = param[i]
            elif name == 'cov':
                del rt['cov']
                rt['C'] = param[numpy.ix_(index,index)]
                rt['Cm1'] = numpy.linalg.inv( param[numpy.ix_(index,index)] )
            if indices.has_key('subunit_index'):
                sindex = indices['subunit_index'][i]
                if name == 'V2':
                    rt['V2'] = param[sindex]
                elif name == 'T':
                    rt['T'] = param[numpy.ix_(index,sindex,range(param.shape[2]))]
            else:
                if name == 'T':
                    rt['T'] = param[:,index,:]
    return  rt

def fuse_params( into , params , i , indices=None ):
    if indices is not None and indices.has_key('subunit_index'):
        index = indices['subunit_index'][i]
        for name,param in params.items():
            if name is 'v1':
                into['sv1'][i] = param
            elif name is 'V2':
                into['V2'][index] += param
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


def test_global_objective( objective, unknowns ):
    print
    print 'Testing global objective:'
    if isinstance( unknowns, type({}) ):
        test_param = objective.flat(unknowns)
        print 'discrepancy when flattening and unflattening:', \
              [(n,numpy.sum((v-unknowns[n])**2.)) 
                for n,v in objective.unflat(test_param).items()]
    else:
        test_param = objective.unflat( unknowns )
        print 'discrepancy when flattening and unflattening:', \
            numpy.sum((unknowns-objective.flat(test_param))**2.)
    print 'f:', objective.f(test_param),objective.f(unknowns)
#    print 'df:', objective.unflat(objective.df(test_param))
    if hasattr( objective , 'barrier' ):
        print 'barrier:', objective.barrier(test_param),objective.barrier(unknowns)
    print

def update_dict( d , key, resource , names ):
    for name in names:
        if resource.has_key(name): d.update({key:resource[name]})
#    return d

#@memory.cache
def optimize_objective( obj, init, gtol=1e-2 , maxiter=500 , optimizer=optimize ):
    optimizer = optimizer.optimizer( obj )
    return optimizer( init_params=init, maxiter=maxiter, gtol=gtol )


#nodes = numpy.array([0., 3., 6., 10., 20., 40., 100.])
#init_u = numpy.array([0.4, 0.2, 0.05, 0.02, 0.01, 0.005, 0.])

nodes = numpy.array([0., 2., 3., 5., 7., 10., 13., 16., 20., 25., 30., 40., 60., 100.])
init_u = numpy.array([0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.,0.,0.,0.,0.,0.])

#nodes = numpy.array([0., 2.3, 2.8, 3.3, 4., 5., 6., 7.5, 10., 14., 20., 40.])

value_matrix = numpy.eye(len(nodes))
shapes = [ radial_piecewise_linear(nodes,values) for values in value_matrix]

#init_u = numpy.array([3.05, 2.12, -2.14, 0.45, 0.56, -0.01, -0.18])
#init_u = numpy.exp(-0.5*(nodes**2))
##init_u = numpy.ones(nodes.shape)
#init_u = init_u/numpy.sqrt(numpy.sum(init_u**2.)) * 0.05

init_V2 = numpy.zeros(len(inferred_locations))

indices = extract( train_stats(rgc_type), ['sparse_index', 'subunit_index'] )
                               
import gc
gc.collect()

def default( d, defaults, fields=None ):
    for name,value in defaults.items():
        if (fields is None) or (name in fields):
            if not d.has_key(name): d[name] = value

def global_objective( unknowns, knowns, vardict, run ):
    print
    print 'Preparing objective for unknowns: ', unknowns
    sys.stdout.flush()
    t0 = time.time()
    variables = ['u','V2','sv1','sparse_STA','sparse_STC','V1','N_spikes','T','cov']
    run.update({'u':init_u, 'V2':init_V2, 'sv1':numpy.abs(sv1),
                'T':retina.place_cells( cones, cones, shapes )})
    default( knowns, run, variables ) 
    outputs = {}
    update_dict( outputs, 'f', vardict , 
            ['LNP', 'LQLEP_wPrior', 'LQLEP_positiveV1', 'LQLEP_positive_u'] )
    update_dict( outputs , 'LL'      , vardict , ['LNP', 'LQLEP'] )
    update_dict( outputs , 'barrier' , vardict , 
                ['barrier','barrier_positiveV1','barrier_positive_u'] )
    single_obj = single_objective( split_params( unknowns, 0, indices ), vardict, outputs)
    objectives = [single_obj.where({},**split_params(knowns,i,indices))
                                   for i in range(NRGC)]
    symbolic_params = extract(vardict,unknowns.keys())
    args   = extract(vardict,set(variables)-set(unknowns.keys()))
    global_obj = kolia_theano.Objective( symbolic_params, unknowns, args, {})
    global_obj = global_obj.where({'indices':indices}, **knowns)
#    ipdb.set_trace()
    for fname in ['f','df','barrier','LL']:
        if hasattr(objectives[0],fname):
            setattr(global_obj,fname,partial(_sum_objectives, objectives, global_obj, fname))
            setattr(getattr(global_obj,fname),'__name__',fname)
    test_global_objective( global_obj, unknowns )
    global_obj.description = ''
    print '... done preparing objective in', time.time()-t0,'sec.'
    return global_obj

maxiter = 3000

def display_params( result ):
    for name,x in result.items():
        if x.size == 1:
            print name,':',x,
        elif x.size>20:
            print name, 'min mean max: ',
            print numpy.min(x), numpy.mean(x), numpy.max(x)            
        else:
            scale = numpy.max(numpy.abs(x))
            print name, (': (scale %f)'%scale),
            for p in x: print '%.4f' %(p/scale),
            print
    print
    
def plot_params( result , filename ):
    save(result,filename)
    pylab.clf()
    pylab.close('all')
    if result.has_key('V2'):
        pylab.figure(2, figsize=(12,10))
        zeros = numpy.nonzero( result['V2'] == 0 )[0]
        a1_V2   = kb.values_to_color( result['V2'] , (0.8,0.,0.) )
        a_V2 = [[x,1.-x,0.,a] for x,_,_,a in a1_V2]
        for i in zeros: a_V2[i][3] = 0.1
        kb.plot_circles( sizes=1.5, offsets=inferred_locations, linewidth=0.001,
                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
        try:
            pylab.savefig('/Users/kolia/Desktop/V2_%s.pdf'%filename, format='pdf')
        except: pass                
    pylab.figure(1, figsize=(10,12))
    N = len([1 for v in result.values() if v.size>1])
    i = 1
    for name,x in result.items():
        pylab.subplot(N,1,i)
        if x.size is nodes.size:
            scale = numpy.max(numpy.abs(x))
            pylab.plot(nodes,x)
            pylab.title(name+': '+', '.join([('%.2f'%(y/scale)) for y in x])+' scale:'+('%.2f'%scale))
            pylab.xlabel('nodes: '+', '.join([('%.1f'%y) for y in nodes]))
            i += 1
        elif x.size>1:
            pylab.subplot(N,1,i)
            xx = x[ numpy.nonzero(x) ]
            pylab.title(name)
            pylab.hist(x, bins=50)
            i += 1
    pylab.savefig('/Users/kolia/Desktop/%s.pdf'%filename, format='pdf')


iterations = [0]
def callback( objective , params , force=False , other={} , objectives=[] ):
    result = objective.unflat(params)
    result.update(other)
    for o in objectives:
        result.update( {o.description:o.LL(params)} )
    display_params( result )
    if force or numpy.remainder( iterations[0] , 20 ) == 19:
        filename = objective.description+'_'+'_'.join(sorted(result.keys()))+'_'+ \
                  rgc_type+'_lam'+str(int(100*lam))+'_'+str(int(maxiter))+'iters'
        plot_params( result, filename )
    iterations[0] = iterations[0] + 1


@memory.cache
def test_LNP(rgc_type=rgc_type):
    vardict   = LNP( **thetaM( **linear_reparameterization()))
    unknowns  = {'sv1':sv1}
    u = numpy.zeros(len(nodes))
    u[0] = 1.
    train_LNP = global_objective( {'sv1':sv1}, {'u':u,'V2':init_V2}, vardict, 
                                   run=train_stats(rgc_type))
    train_LNP.with_callback(callback)
    train_LNP.description = 'LNP'
    params = optimize.optimizer( train_LNP )( init_params=unknowns, 
                                              maxiter=1000, gtol=1e-7 )
    return global_objective( {'sv1':sv1}, {'u':u,'V2':init_V2}, vardict, 
                              run=test_stats(rgc_type)).LL(params)

test_LNP_LL = test_LNP()
print
print 'LNP test LL:',test_LNP_LL
print

@memory.cache
def optimize_LQLEP(rgc_type, maxiter=maxiter,
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **linear_reparameterization())))):
    unknowns = {'sv1':sv1, 'V2':init_V2 , 'u':init_u+0.0000001}
    train_LQLEP = global_objective( unknowns, {}, vardict, run=train_stats(rgc_type))
    test_LQLEP  = global_objective( unknowns, {}, vardict, run= test_stats(rgc_type))
    test_LQLEP.description = 'Test LQLEP'
    train_LQLEP.with_callback(partial(callback,other={'Test LNP':test_LNP_LL},
                                               objectives=[test_LQLEP]))
#    train_LQLEP.with_callback(callback)
    train_LQLEP.description = 'LQLEP_positiveU'
    trained = optimize_objective( train_LQLEP, unknowns, gtol=1e-10 , maxiter=maxiter)
    print 'RGC type:', rgc_type
    test_global_objective( train_LQLEP, trained )
    train_LQLEP.callback( trained, force=True )
    train_LQLEP.callback( train_LQLEP.unflat( trained ), force=True )
    return trained
    
trained = optimize_LQLEP(rgc_type, maxiter=40,
         vardict = LQLEP_positive_u( **LQLEP_wBarrier( **LQLEP( **thetaM( **linear_reparameterization())))))
