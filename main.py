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
                        thetaM, linear_parameterization, \
                        LQLEP_positive_u, u2V2_parameterization, \
                        u2c_parameterization, u2cd_parameterization, \
                        Poisson_LL, RGC_LE, subunit_LQ

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

import ipdb

data = retina.read_data()

cones             = data['cone_locations']

def get_rgc_type( i , data=data ):
    eyedee = data['rgc_ids'][i]
    for rgctype, inds in data['rgc_types'].items():
        if eyedee in inds: return rgctype
    return None

#from kolia_base import save
#save(data,'data_localization')

from scipy.linalg   import schur
import sys
sys.setrecursionlimit(10000)

@memory.cache
def STA_stats( spikes = data['spikes'] ):
    stats = retina.accumulate_statistics( 
            retina.read_stimulus( data['spikes'],
                                 stimulus_pattern='cone_input_%d.mat' ,
                                 skip_pattern=(5,0) ) )
    print 'calculated STAs'
    sys.stdout.flush()
    return stats

def n_largest( values , keep=10 ):
    sorted_values = numpy.abs(values)
    sorted_values.sort()
    cutoff = sorted_values[-keep]
    return numpy.nonzero( numpy.abs(values) >= cutoff )[0]

keep = {'off midget' :  30, 'on midget' :  30, 
        'off parasol': 120, 'on parasol': 120}


def which_cells( rgc_type='off midget' ):
    return [i for i,ind in enumerate(data['rgc_ids'])
            if  ind   in data['rgc_types'][rgc_type]] #and i in which]
#                   and data['rgc_locations'][i][1]>170 ]

def which_spikes( rgc_type='off midget' ):
    return data['spikes'][which_cells(rgc_type),:]

def make_sparse_indices( rgc_type='off midget' , stats={} ):
    keepers = keep[rgc_type]
    which_rgc = which_cells( rgc_type )
    print 'Making sparse indices for',len(which_rgc),'RGCs. ',
    print 'stats.keys():',stats.keys()
    if stats.has_key('N_spikes'):
        stats['N_spikes'] = [stats['N_spikes'][i] for i in which_rgc]
    if stats.has_key('STA'):
        stats['STA'] = [stats['STA'][i] for i in which_rgc]
    if stats.has_key('cov'):
        std = numpy.sqrt( numpy.diag( stats['cov'] ) )
    if stats.has_key('STA') and stats.has_key('mean') and stats.has_key('cov'):
        stats[ 'sparse_index'] = [n_largest( numpy.abs(sta-stats['mean'])/std, keep=keepers )
                                  for sta in stats['STA']]
        stats['subunit_index'] = [n_largest( numpy.abs(sta-stats['mean'])/std, keep=keepers )
                                  for sta in stats['STA']]
        del stats['STA']
    stats['rgc_type']  = rgc_type
    stats['rgc_index'] = which_rgc
    stats['spikes']    = which_spikes( rgc_type )
    return stats

def make_sparse_stats( rgc_type='off midget', stats={}, skip_pattern=None ):
    print 'Fitting sparse stats for',len(stats['rgc_index']),'RGCs'
    stats.update( retina.accumulate_statistics( 
        data_generator = retina.read_stimulus( stats['spikes'] , 
                                               skip_pattern=skip_pattern) ,
        feature        = lambda x : x                   ,
        pipelines      = retina.fit_U                    ,
        sparse_index   = stats['sparse_index']          ))
    return stats

#@memory.cache
#def smooth_STA_stats( rgc_type = 'off parasol' ):
#    spikes = which_spikes( rgc_type )
#    return dSTA_stats( spikes, lambda x: 
#          numpy.dot( retina.gaussian2D_weights( cones, cones, sigma=3. ), x) )

@memory.cache
def linear_stats( rgc_type, skip_pattern=(5,0)):
    stats = STA_stats()
    stats = make_sparse_indices( rgc_type=rgc_type, stats=stats )
    return make_sparse_stats( rgc_type=rgc_type, stats=stats, skip_pattern=skip_pattern )

#@memory.cache
#def linear_stats( rgc_type, skip_pattern=(5,0)):
#    stats = stats_for_ARD( lambda x: numpy.dot( retina.gaussian2D_weights( cones, cones, sigma=3. ), x) )
#    return fit_U_stats( rgc_type=rgc_type, stats=stats, skip_pattern=skip_pattern )

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
#                print numpy.sum(ind), ' distances between', nodes[i-1], ' and ', n 
                result[ ind ] = \
                      (values[i-1]*(n-x[ind])+values[i]*(x[ind]-nodes[i-1])) / (n-nodes[i-1])
            index = index_i
        return result
    return f

from kolia_base import save, extract


@memory.cache
def ARD( stats , lam=0.00 ):
    stats['D'], stats['Z'] = schur(stats['cov'])
    print 'Starting ARD of size ', stats['Z'].shape,' with lambda=',lam
    sys.stdout.flush()
    D,Z = stats['D']/2 , stats['Z']
    print 'Schur decomposition completed'
    sys.stdout.flush()
    DD  = numpy.diag(D)
    keep= DD>1e-10
    P    =  (Z[:,keep] * numpy.sqrt(DD[keep])).T
    dSTA = numpy.concatenate(
        [STA[:,numpy.newaxis]-stats['mean'][:,numpy.newaxis] 
         for STA in stats['STA']], axis=1)
    y    =  numpy.dot ( (Z[:,keep] * 1/numpy.sqrt(DD[keep])).T , dSTA ) / 2
    iW = 1e-1
    for i in range(1):
        print 'Irlsing'
        sys.stdout.flush()
        V, iW = IRLS.IRLS( y, P, x=0, disp_every=1, lam=lam, maxiter=2 , 
                           ftol=1e-5, nonzero=1e-1, iw=iW)
        save({'V':V,'iW':iW},'Localizing_lam%.0e'%lam)
    return V, iW
    
import time


def single_objective( param_templates , vardict , outputs ):
    arg     = set(['u','V2','STA','STC','v1','N_spike','T','Cm1','C','c','uc'])
    print 'Simplifying single objective...'
    sys.stdout.flush()
    t0 = time.time()
    params = extract(vardict,param_templates.keys())
    args   = extract(vardict,arg-set(param_templates.keys()))
    differentiate = []
    if outputs.has_key('f'): differentiate += ['f']
    obj = kolia_theano.Objective( params, param_templates, args, outputs,
                                  differentiate=differentiate, mode='FAST_RUN' )
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
        elif name == 'all_spikes':
            del rt['all_spikes']
            rt['spikes'] = param[i]
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
                elif name == 'stimulus':
                    rt['stimulus'] = param[sindex,:]
                elif name == 'nonlinearity':
                    rt['V2'] = param[sindex]
                elif name == 'c':
                    rt['c'] = param[sindex]
                elif name == 'd':
                    rt['d'] = param[sindex]
                elif name == 'T':
                    rt['T'] = param[numpy.ix_(index,sindex,range(param.shape[2]))]
            else:
                if name == 'T':
                    rt['T'] = param[:,index,:]
    return  rt

def fuse_params( into , params , i , indices=None ):
    if isinstance(into, type({})) and len(params.keys()) == 1:
        key = params.keys()[0]
        if key == 'nonlinearity':
            into = numpy.zeros(indices['N_subunits'])
        elif key == 'rates':
            into = []
        elif key == 'theta':
            into = []
        elif key in ['f', 'LL', 'barrier']:
            into = 0
    if indices is not None and indices.has_key('subunit_index'):
        index = indices['subunit_index'][i]
        for name,param in params.items():
            if name is 'v1':
                into['sv1'][i] = param
            elif name == 'V2':
                into['V2'][index] += param
            elif name == 'nonlinearity':
                into[index] += param
            elif name == 'rates':
                into += [param]
            elif name == 'theta':
                p = numpy.zeros(indices['N_subunits'])
                p[index] = param
                into += [p]
            elif name == 'c':
                into['c'][index] += param
            elif name == 'd':
                into['d'][index] += param
            elif isinstance(into, type({})):
                into[name] += param
            else:
                into += param
    else:
        for name,param in params.items():
            if name == 'v1':
                into['V1'][i,:] = param
            elif isinstance(into, type({})):
                into[name] += param
            else:
                into += param
    return into

def _sum_objectives( objectives, global_objective, attribute, X ):
    x = global_objective.unflat(X)
    indices = global_objective.indices
    result  = kb.zeros_like(x)
    for i,obj in enumerate(objectives):
        flat = getattr(obj,attribute)(split_params(x,i,indices))
        try:
            param = obj.unflat(flat)
        except:
            param = {attribute:flat}
        result = fuse_params( result , param , i , indices )
    return kb.flat( result )

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
    if hasattr( objective , 'f' ):
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
    print 'Optimization initialized with', [(n,v.shape) for n,v in init.items()]
    optimizer = optimizer.optimizer( obj )
    return optimizer( init_params=init, maxiter=maxiter, gtol=gtol )

nodes = numpy.array([0., 2., 3., 5., 7., 10., 13., 16., 20., 25., 30., 40., 60., 100.])
init_u = numpy.array([0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.,0.,0.,0.,0.,0.])

value_matrix = numpy.eye(len(nodes))
shapes = [ radial_piecewise_linear(nodes,values) for values in value_matrix]

init_V2 = numpy.zeros(len(cones))

import gc
gc.collect()

def default( d, defaults, fields=None ):
    for name,value in defaults.items():
        if (fields is None) or (name in fields):
            if not d.has_key(name): d[name] = value

def init_sv1( rgc_type ):
    run = linear_stats(rgc_type,(5,0))
    return numpy.vstack([numpy.linalg.solve( 
            run['cov'][numpy.ix_(index,index)], sparse_STA ) 
            for sparse_STA,index in zip(run['sparse_STA'],run['subunit_index'])])

def global_objective( unknowns, knowns, vardict, run, indices ):
    print
    print 'Preparing objective for unknowns: ', [(n,v.shape) for n,v in unknowns.items()]
    sys.stdout.flush()
    variables = ['u','V2','sv1','sparse_STA','sparse_STC','V1','N_spikes','T','cov']
    run.update({'u':init_u, 'V2':init_V2, 'sv1':init_sv1(run['rgc_type']), 
                'T':retina.place_cells( cones, cones, shapes )})
    default( knowns, run, variables ) 
    outputs = {}
    update_dict( outputs, 'f', vardict , 
            ['LNP', 'LQLEP_wPrior', 'LQLEP_positiveV1', 'LQLEP_positive_u'] )
    update_dict( outputs , 'LL'      , vardict , ['LNP', 'LQLEP'] )
    update_dict( outputs , 'barrier' , vardict , 
                ['barrier','barrier_positiveV1','barrier_positive_u'] )
    update_dict( outputs , 'nonlinearity', vardict , ['nonlinearity'] )
    if unknowns.has_key('sv1'):
        print
        print 'sv1 shape', unknowns['sv1'].shape
        print
    NRGC = len( run['sparse_STA'] )
    return make_global_objective(unknowns,knowns,vardict,variables,outputs,indices,NRGC)

def make_global_objective(unknowns,knowns,vardict,variables,outputs,indices,NRGC):
    t0 = time.time()
    print 'unknowns:',unknowns.keys()
    print 'knowns:',  knowns.keys()
    single_obj = single_objective( split_params( unknowns, 0, indices ), vardict, outputs)
    objectives = [single_obj.where({},**split_params(knowns,i,indices))
                                   for i in range(NRGC)]
    symbolic_params = extract(vardict,unknowns.keys())
    args   = extract(vardict,set(variables)-set(unknowns.keys()))
    global_obj = kolia_theano.Objective( symbolic_params, unknowns, args, {})
    global_obj = global_obj.where({'indices':indices}, **knowns)
#    ipdb.set_trace()
    for fname in outputs.keys():
        if hasattr(objectives[0],fname):
            setattr(global_obj,fname,partial(_sum_objectives, objectives, global_obj, fname))
            setattr(getattr(global_obj,fname),'__name__',fname)
#    test_global_objective( global_obj, unknowns )
    global_obj.description = ''
    print '... done preparing objective in', time.time()-t0,'sec.'
    return global_obj


maxiter = 10000

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
    
from   matplotlib.ticker import *
def plot_params( result , filename ):
    import pylab
    pylab.clf()
    pylab.close('all')
    nonlinearity = None
    if result.has_key('V2'):
        nonlinearity = result['V2']
    if result.has_key('nonlinearity'):
        nonlinearity = result['nonlinearity']
    if nonlinearity is not None:
        pylab.figure(2, figsize=(12,10))
        zeros = numpy.nonzero( nonlinearity == 0 )[0]

#        a1_V2   = kb.values_to_color( nonlinearity , 'r' )
#        a_V2 = [[x,1.-x,0.,a] for x,_,_,a in a1_V2]
#        for i in zeros: a_V2[i][3] = 0.1
#        kb.plot_circles( sizes=1.5, offsets=cones, linewidth=0.001,
#                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
#        try:
#            pylab.savefig('/Users/kolia/Desktop/nonlinearity_%s.pdf'%filename, format='pdf')
#        except: pass

#        a1_V2   = kb.values_to_uniform_color( numpy.abs(nonlinearity) , 'r' )
#        a_V2 = [[x,0.,0.,x*x] for x,_,_,a in a1_V2]
#        for i in zeros:  a_V2[i] = [0.,0.,0.,0.1]
#        kb.plot_circles( sizes=1.5, offsets=cones, linewidth=0.001,
#                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
#        try:
#            pylab.savefig('/Users/kolia/Desktop/nonlinearity_abs_unif_%s.pdf'%filename, format='pdf')
#        except: pass

        a1_V2   = kb.values_to_uniform_color( nonlinearity , 'r' )
        a_V2 = [[x,1.-x,0.,a] for x,_,_,a in a1_V2]
        for i in zeros: a_V2[i][3] = 0.1
        kb.plot_circles( sizes=1.5, offsets=cones, linewidth=0.001,
                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
        try:
            pylab.savefig('/Users/kolia/Desktop/nonlinearity_unif_%s.pdf'%filename, format='pdf')
        except: pass

#    if result.has_key('c'):
#        a1_V2   = kb.values_to_color( result['c'] , 'r' )
#        a_V2 = [[x,1.-x,0.,a] for x,_,_,a in a1_V2]
#        for i in zeros: a_V2[i][3] = 0.1
#        kb.plot_circles( sizes=1.5, offsets=cones, linewidth=0.001,
#                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
#        try:
#            pylab.savefig('/Users/kolia/Desktop/c_%s.pdf'%filename, format='pdf')
#        except: pass

        a1_V2   = kb.values_to_uniform_color( result['c'] , 'r' )
        a_V2 = [[x,1.-x,0.,a] for x,_,_,a in a1_V2]
        for i in zeros: a_V2[i][3] = 0.1
        kb.plot_circles( sizes=1.5, offsets=cones, linewidth=0.001,
                         facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))
        try:
            pylab.savefig('/Users/kolia/Desktop/c_unif_%s.pdf'%filename, format='pdf')
        except: pass


    pylab.figure(1, figsize=(10,12))
    N = len([1 for v in result.values() if v.size>1 and v.ndim<2])
    if N>0:
        i = 1
        for name,x in result.items():
            if x.ndim<2:
                ax = pylab.subplot(N,1,i)
                ax.yaxis.set_major_locator( LinearLocator(numticks=2) )
                if x.size is nodes.size:
                    print 'Plotting', name
#                    scale = numpy.max(numpy.abs(x))
                    pylab.plot(nodes,x,'b-',nodes,x,'ro')
                    pylab.ylabel(name,size='medium')
#                    pylab.title(name+': '+', '.join([('%.2f'%(y/scale)) for y in x])+ \
#                                ' scale:'+('%.2f'%scale))
#                    pylab.xlabel('nodes: '+', '.join([('%.1f'%y) for y in nodes]))
                    i += 1
                elif x.size>1:
                    print 'Plotting', name
                    xx = x[ numpy.nonzero(x) ]
                    pylab.ylabel(name,size='medium')
                    pylab.hist(x, bins=80)
                    i += 1
        pylab.savefig('/Users/kolia/Desktop/%s.pdf'%filename, format='pdf')


iterations = [0]
def callback( objective , params , force=False , other={} , objectives=[] , plot_every=42):
    result = objective.unflat(params)
#    ipdb.set_trace()
    result.update(other)
    for o in objectives:
        result.update( {o.description:o.LL(params)} )
    for name,value in result.items():
        if hasattr(value, '__call__'):
            result[name] = value(params)
    try:
        display_params( result )
    except:
        pass
    if force or numpy.remainder( iterations[0] , plot_every ) == plot_every-1:
        filename = objective.description+('iter%d'%iterations[0])
#        filename = objective.description+'_'+'_'.join(sorted(result.keys()))+'_'+ \
#                  rgc_type+'_lam'+str(int(100*lam))+'_'+str(int(maxiter))+'iters'
        save(result,filename)
        plot_params( result, filename )
    iterations[0] = iterations[0] + 1

def LNP_model( sv1, params=None ):
    u = numpy.zeros(len(nodes))
    u[0] = 1.
    model = {'sv1':sv1, 'u':u,'V2':init_V2, 'c':init_V2, 'uc':numpy.zeros_like(nodes)}
    if params is not None:
        model.update(params)
    return model

def _test_LNP( rgc_type='off parasol' ):
    vardict   = LNP( **thetaM( **linear_reparameterization()))
    init_LNP  = LNP_model( init_sv1(rgc_type) )
    indices = extract( linear_stats( rgc_type, (5,0) ), ['sparse_index', 'subunit_index'] )
    indices['N_subunits'] = len(cones)
    unknown = extract(init_LNP,['sv1'])
    train_LNP = global_objective( unknown, extract(init_LNP,['u','V2']), 
                                  vardict, run=linear_stats( rgc_type, (5,0) ),
                                  indices=indices)
    train_LNP.with_callback(callback)
    train_LNP.description = 'LNP'
    sv1 = optimize.optimizer( train_LNP )( init_params=unknown, maxiter=5000, gtol=1e-7 )
    model = LNP_model( train_LNP.unflat( sv1 )['sv1'] )
    model['LL'] = global_objective( unknown, extract(init_LNP,['u','V2']), 
                             vardict, run=linear_stats( rgc_type, (-5,0)), 
                             indices=indices).LL(sv1)
    save(model,'LNP_'+rgc_type)
    return model

@memory.cache
def test_LNP(rgc_type='off parasol'):
    return _test_LNP(rgc_type=rgc_type)

@memory.cache
def optimize_LQLEP( rgc_type, filename=None, maxiter=maxiter, indices=None, description='',
    unknowns=['sv1','V2','u','uc','c','ud','d'],
    vardict = LQLEP_wBarrier( **LQLEP( **thetaM( **u2c_parameterization())))):
#    unknowns = {'sv1':sv1, 'V2':init_V2 , 'u':old['u'], 'uc':old['u'], 'c':init_V2}
    defaults = extract( { 'sv1':init_sv1( rgc_type ), 'V2':init_V2 , 'u':init_u, 
                          'uc':numpy.zeros_like(init_u), 'c':init_V2,
                          'ud':0.001*numpy.ones_like(init_u), 'd':0.0001+init_V2},
                        list( set(unknowns).intersection( set(vardict.keys()) ) ) + ['sv1'])
    if filename is not None:
        print 'Re-optimizing',filename
        unknowns = kb.load(filename)
        for name in unknowns.keys():
            if not defaults.has_key(name): del unknowns[name]
        default(unknowns,defaults)
    else:
        unknowns = defaults
#    if rgc_type[:3] == 'off':
#        unknowns['u'] = -0.01*numpy.abs(unknowns['u'])
    if vardict.has_key('barrier_positiveV1'):
        unknowns['sv1'] = numpy.abs(unknowns['sv1'])
    else:
        unknowns['sv1'] = unknowns['sv1']*0.01
    train_LQLEP = global_objective( unknowns, {}, vardict, run=linear_stats(rgc_type,( 5,0)), indices=indices)
    test_LQLEP  = global_objective( unknowns, {}, vardict, run=linear_stats(rgc_type,(-5,0)), indices=indices)
    test_LQLEP.description = 'Test_LQLEP'
    train_LQLEP.with_callback(partial(callback,other=
                 {'Test_LNP':test_LNP(rgc_type)['LL'], 'nonlinearity':test_LQLEP.nonlinearity},
                  objectives=[test_LQLEP]))
    train_LQLEP.description = description+rgc_type
    unknowns['V2'] = unknowns['V2']*0.001
    trained = optimize_objective( train_LQLEP, unknowns, gtol=1e-10 , maxiter=maxiter)
    print 'RGC type:', rgc_type
    test_global_objective( train_LQLEP, trained )
    train_LQLEP.callback( trained, force=True )
    train_LQLEP.callback( train_LQLEP.unflat( trained ), force=True )
    return trained

def forward_LQLEP( stimulus, all_spikes, model, indices, vardict=u2c_parameterization()):
    print
    print 'Preparing forward LQLEP model.'
    sys.stdout.flush()
    vardict = Poisson_LL( **RGC_LE( **subunit_LQ( **vardict)))
    variables = ['stimulus','all_spikes','u','V2','sv1','T','c','uc']
    knowns = model
    knowns.update({'T':retina.place_cells( cones, cones, shapes )})
    unknowns = {'stimulus':stimulus, 'all_spikes':all_spikes}
    outputs  = {'LL': vardict['loglikelihood'], 'rates': vardict['rgc_out']} 
#                'theta': vardict['theta']}
    NRGC = len( all_spikes )
    return make_global_objective(unknowns,knowns,vardict,variables,outputs,indices,NRGC)


def load_model( filename=None, rgctype='off parasol' ):
#    filename += rgctype
    print 'Loading model', filename
    indices = extract( linear_stats(rgctype,(5,0)), ['sparse_index', 'subunit_index'] )
    indices['N_subunits'] = len(cones)
    spikes = which_spikes( rgctype )
    data_generator = retina.read_stimulus( spikes,
                                 stimulus_pattern='cone_input_%d.mat' ,
                                 skip_pattern=(-5,0) )
    stimdata = data_generator.next()
    print 'stimdata', stimdata.keys()    
    model = kb.load(filename)
    for n,v in model.items():
        if isinstance(v,type({})):
            model.update(v)
    return forward_LQLEP( stimdata['stimulus'], stimdata['spikes'], model, indices)


import scipy
@memory.cache
def exact_LL( filename=None, rgctype='off parasol' ):
    print 'Calculating exact LL for', filename
    forward = load_model( filename, rgctype ).LL
    return sum( map( forward, retina.read_stimulus( which_spikes( rgctype ),
                                     stimulus_pattern='cone_input_%d.mat' ,
                                     skip_pattern=(-5,0) ) ) )

@memory.cache
def exact_normalized_LLs( filename=None, rgctype='off parasol' ):
    def normalizer( spikes=None, **other ):
        lnfact  = numpy.sum( scipy.special.gammaln( numpy.vstack( spikes['spikes'] + 1 ) ) )
        N_total = numpy.sum( numpy.vstack( spikes['spikes'] ) )
        return numpy.array([lnfact, N_total])
    lnfact, N_total = sum( map( normalizer, 
                           retina.read_stimulus( which_spikes( rgctype ),
                                                stimulus_pattern='cone_input_%d.mat' ,
                                                skip_pattern=(-5,0) ) ) )
    LQLEP_LL = exact_LL( filename, rgctype )
    LNP_LL   = exact_LL( 'LNP_'+rgctype, rgctype )
    print 'LL', (LQLEP_LL+lnfact)/N_total,'LNPLL',(LNP_LL+lnfact)/N_total
    return {'LQLEP_LL':float((LQLEP_LL+lnfact)/N_total), 'LNP_LL':float((LNP_LL+lnfact)/N_total),
            'log_factorial':lnfact/N_total, 'N_total':N_total,
            'source':''''LQLEP_LL':float((LQLEP_LL+lnfact)/N_total), 
                        'LNP_LL':float((LNP_LL+lnfact)/N_total),
                        'log_factorial':lnfact/N_total, 'N_total':N_total'''}

@memory.cache
def simulated_STAC( filename=None, rgctype='off parasol' ):
    print 'Calculating STAC for spikes generated with model', filename
    forward = load_model( filename, rgctype )
    def spike_generator( d ):
        return [numpy.random.poisson(r) for r in forward.rates(d)]
    stats = STA_stats()
    stats = make_sparse_indices( rgctype, stats )
    stats.update( retina.accumulate_statistics( 
        data_generator = retina.simulate_data( spike_generator,
                         retina.read_stimulus( which_spikes( rgctype) , 
                                               skip_pattern=(-5,0))) ,
        pipelines      = retina.fit_U                    ,
        sparse_index   = stats['sparse_index']          ))
    return stats

def plot_cone_values( values ):
    a1_V2   = kb.values_to_color( values , 'r' )
    a_V2 = [[x,1.-x,0.,a] for x,_,_,a in a1_V2]
    for i in zeros: a_V2[i][3] = 0.1
    kb.plot_circles( sizes=1.5, offsets=cones, linewidth=0.001,
                     facecolors=a_V2, edgecolors=(0.,0.,0.,0.3))


types = ['off parasol', 'on parasol', 'on midget', 'off midget']
#types.reverse()

for rgctype in types:
    print
    print 'Calculating linear_stats for', rgctype
    ls  = linear_stats(rgctype,(5,0))
    moot = test_LNP( rgctype )

#for rgctype in types:
#    print
#    print
#    print
#    print rgctype
#    print
#    print
#    infile = 're2STD_Uc2_'+rgctype
#
#    indices = extract( linear_stats(rgctype,(5,0)), ['sparse_index', 'subunit_index'] )
#    indices['N_subunits'] = len(cones)
#    retrain = optimize_LQLEP(rgctype, filename=infile, maxiter=maxiter, indices=indices,
#             description='posV1_c',
#             vardict = LQLEP_positiveV1( **LQLEP_wBarrier( **LQLEP(
#                                         **thetaM( **u2c_parameterization())))))

for rgctype in types:
    print
    print
    print
    print rgctype
    print
    print
    infile = 're2STD_Uc2_'+rgctype

    indices = extract( linear_stats(rgctype,(5,0)), ['sparse_index', 'subunit_index'] )
    indices['N_subunits'] = len(cones)
    retrain = optimize_LQLEP(rgctype, filename=infile, indices=indices,
             description='posV1_c', maxiter=42*10+2, 
             vardict = LQLEP_positiveV1( **LQLEP_wBarrier( **LQLEP(
                                         **thetaM( **u2c_parameterization())))))

#for rgctype in types:
#    print
#    print 'Calculating simulated_STAC for', rgctype
#    filename = 're2STD_Uc2_'
#    save( simulated_STAC( filename, rgctype), filename+rgctype+'_STAC' )


#for rgctype in types:
#    infile = 're2STD_usmooth_'
#    e = exact_LL( filename=infile, rgctype=rgctype )
#    save( e, infile+'_LL' )