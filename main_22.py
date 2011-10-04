import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, UVs , eig_barrier , eig_pos_barrier, eigsM, invM
 
import kolia_base as kb 
reload(kb) 
 
import kolia_theano
reload(kolia_theano)
 
import simulate_retina
reload(simulate_retina)
 
import optimize
reload(optimize)

import FALMS
reload(FALMS)
 
import IRLS
reload(IRLS)
from   IRLS import IRLS
 
import numpy        as np
from scipy.linalg   import schur
import numpy.random as Rand
  
import pylab as p

from IPython.Debugger import Tracer; debug_here = Tracer()
 
# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)
 
 
############################
# Setting up simulated data
 
#N_cells=[40,20,12]
#N_cells=[30,15,9]
#N_cells=[20,10,6]
N_cells=[20,13,7]

V2 = 0.5
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
 
# Quantities of interest
overcompleteness = 10
N_filters = N_cells[1]*overcompleteness
filters = np.concatenate(
    [simulate_retina.weights(sigma=n, shape=(N_filters,N_cells[0]))
    for n in [10.]] )

# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
def simulator( v2, N_filters, nonlinearity, N_cells , sigma_spatial , N_timebins ):
    retina = simulate_retina.LNLNP_ring_model( nonlinearity = nonlinearity , 
                                               N_cells = N_cells , 
                                               sigma_spatial = sigma_spatial )
#    stimulus = simulate_retina.white_gaussian_stimulus( dimension  = N_cells[0] , 
#                                                         sigma = 1. )
    stimulus = simulate_retina.Stimulus( simulate_retina.white_gaussian )
    return simulate_retina.run_LNLNP( retina , stimulus = stimulus , 
                   N_timebins = N_timebins ,
                   average_me = {'features':lambda x: NL(np.dot(filters,x))} )
simulate = memory.cache(simulator)
R = simulate( V2, N_filters, nonlinearity=NL, N_cells=N_cells , sigma_spatial=[10.,3.],
              N_timebins = 10000 )
 
#testR = simulate( V2, nonlinearity=NL, N_cells=N_cells , sigma_spatial=[20.,3.],
#                  N_timebins = 90000 )
 
             
total = float( np.sum([Nspikes for Nspikes in R['N_spikes']]) )
 
Nspikes = R['N_spikes']/total

dSTA  = np.concatenate(
        [STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis]
        for STA in R['statistics']['features']['STA']], axis=1)
 
D,Z = schur(R['statistics']['features']['cov']/2)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA ) / 2
 
irls = memory.cache(IRLS)
V, iW = irls( y, P, x=0, disp_every=1000, lam=0.02, maxiter=1000000 , 
              ftol=1e-5, nonzero=1e-1)
print 'V'
kb.print_sparse_rows( V, precision=1e-1 )
 
keepers = np.array( [sum(abs(v))>1e-1 for v in V] )
 
Ncones   = R['model']['N_cells'][0]
Nsub     = R['model']['N_cells'][1]
NRGC     = R['model']['N_cells'][2]
#N_spikes = R['N_spikes']
#STA      = R['statistics']['stimulus']['STA']
#STC      = R['statistics']['stimulus']['STC']
Nproj    = np.sum(keepers)
U        = filters[keepers,:]
V1       = V[keepers,:].T
#V2       = V2 * np.ones(Nproj)

centers = np.arange(0,N_cells[0],N_cells[0]/N_cells[1])
indices = np.fromfunction( lambda i,j: j , R['model']['U'].shape)
 
iterations = [0]
def callback( objective , params ):
#    fval = objective.f(params)
    print ' Iter:', iterations[0] # ,' Obj: ' , fval
    if np.remainder( iterations[0] , 5 ) == 0:
        result = objective.unflat(params)
        if np.remainder( iterations[0] , 6 ) == 0: p.close('all')
        if result.has_key('V1'):
            p.figure(1, figsize=(10,12))
            kb.plot_filters(result['V1'])
            p.title('V1 filters')
            p.savefig('/Users/kolia/Desktop/V1_initrue.svg',format='svg')
            p.savefig('/Users/kolia/Desktop/V1_initrue.pdf',format='pdf')
        if result.has_key('U'):
            p.figure(2, figsize=(10,12))
            kb.plot_filters(result['U'][:10,:])
            p.title('1st ten U filters')
            p.savefig('/Users/kolia/Desktop/U_initrue.svg',format='svg')
            p.savefig('/Users/kolia/Desktop/U_initrue.pdf',format='pdf')
    iterations[0] = iterations[0] + 1
#    if not np.isfinite( fval ):
#        eigsM = objective.eigsM( params )
#        invM  = objective.invM( params )
#        debug_here()
 
def FALMopt( objective , u, v2 , mus=np.array([30.,3.,0.3])): #mus=np.array([0.1,1.,10.])): #
    lambdas = np.zeros(v2.size)
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) , 
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':v2 , 'U': u , 'N':NRGC , 'N_spikes':R['N_spikes'] ,
            'Ncones':Ncones , 'centers':centers , 'indices':indices , 'lam':lam , 'lambdas':lambdas }
    obj_V1 = objective.where(**data).with_callback(callback)
    optimize_L2 = FALMS.add_L2_term( obj_V1 , gtol=1e-3 )
    init = (0.001 + 0.01 * Rand.rand(NRGC,u.shape[0])).flatten()
    r = {}
    def falms_callback(falmer):  print '   L0: ', falmer[1][falmer[1]<>0].shape[0]
    for mu in mus:
        print
        print
        print 'mu: ', mu
        reg_L1 = FALMS.L1_L2( mu )
        params = FALMS.falms(init,optimize_L2,reg_L1,mu=0.01,ftol=2e-7,callback=falms_callback)
        r[mu] = params
        print [ (mu,par['M'][par['M']<>0].shape[0]) for mu,par in sorted(r.items())]
    return r

 
#true = {'U' : R['U'] , 'V1': R['V'] }
 
##def objective_U():
##    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
##    targets = kolia_theano.reparameterize(targets,UVs(NRGC))
##    return kolia_theano.Objective( init_params={'U': U }, differentiate=['f'], 
##                                        callback=callback, **targets )
 
 
def objective_U():
    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
    targets = kolia_theano.reparameterize(targets,UVs(NRGC))
    list_targets= kolia_theano.reparameterize({'eigsM':eigsM, 'invM':invM },UVs(NRGC),
                                               reducer=lambda r,x: r + [x], zero=[])
    targets.update( list_targets )
     
    return    kolia_theano.Objective( init_params={'U': U }, differentiate=['f'], **targets )    
#obj_U   = objective_U()
 
@memory.cache
def optimize_U( objective, v1,init_U, v2  ):
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) ,
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':v2 , 'V1': v1 , 'N':NRGC , 'N_spikes':R['N_spikes'] }
     
    optimizer = optimize.optimizer( objective.where(**data).with_callback(callback) )
    params = optimizer(init_params={'U': init_U },maxiter=2000,gtol=1.1e-6)
    opt_U = objective.unflat(params)
    return opt_U['U']
 
 
def objective_V1( v1 = R['model']['V'] ):
    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
    targets = kb.reparameterize(targets,UVs(NRGC))
    return    kolia_theano.Objective( init_params={'V1': v1 }, differentiate=['f'], 
                                       mode='FAST_COMPILE' , **targets )
 
iterations[0] = 0
lam     = 20.
@memory.cache
def optimize_V1( objective, u,v2, lambdas=None , init=None , gtol=1e-4 , maxiter=100):
    if lambdas is None:
        lambdas = np.zeros(v2.size)
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) , 
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':v2 , 'U': u , 'N':NRGC , 'N_spikes':R['N_spikes'] ,
            'Ncones':Ncones , 'centers':centers , 'indices':indices , 'lam':lam , 'lambdas':lambdas }
    optimizer = optimize.optimizer( objective.where(**data).with_callback(callback) )
    if init is None:
        init_params = {'V1': 0.001 + 0.01 * Rand.rand(NRGC,u.shape[0]) }
    else:
        init_params = {'V1': init }
    params = optimizer(init_params=init_params,maxiter=maxiter,gtol=gtol)
    opt_V1 = objective.unflat(params)
    return opt_V1['V1']
 
 
def objective_UV1():
    targets = { 'f':quadratic_Poisson, 'barrier':eig_pos_barrier }
    targets = kolia_theano.reparameterize(targets,UVs(NRGC))    
    return kolia_theano.Objective( init_params={'V1': R['V'] , 'U': R['U'] }, differentiate=['f'], 
                                    mode='FAST_RUN' , **targets )
#    return kolia_theano.Objective( init_params={'V1': V1 , 'U': U }, differentiate=['f'], **targets )

iterations[0] = 0
lam     = 20.
@memory.cache
def optimize_UV1( objective, v1,u,v2 ):
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) ,
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':v2 , 'N':NRGC , 'N_spikes':R['N_spikes'] , 
            'Ncones':Ncones , 'centers':centers , 'indices':indices , 'lam':lam }
    optimizer = optimize.optimizer( objective.where(**data).with_callback(callback) )
    params = optimizer(init_params={'V1': v1 , 'U': u },maxiter=10000,gtol=1.1e-7)
    opt_U = objective.unflat(params)
    return opt_U
 
#print 'Optimizing U,V1'
##opt_UV1 = optimize_UV1( np.wabs(V1)+0.0001, U, V2 )
#opt_UV1 = optimize_UV1( 0.001 + 0.001 * Rand.rand(*V1.shape), U, V2 )
##opt_UV1 = optimize_UV1( opt_UV1['V1'], opt_UV1['U'], V2 )
##opt_UV1 = optimize_UV1( opt_V1, opt_U, V2 )

#obj_UV1  = objective_UV1()
#opt_UV1 = optimize_UV1( R['V'], R['U'], V2*np.ones(Nsub) )

obj_V1 = objective_V1( v1=np.ones((NRGC,filters.shape[0])) )

###########
iterations[0] = -1

#FALMopt( obj_V1 , filters , V2 * np.ones(filters.shape[0] ) )

opt_V1_lam0 = optimize_V1( obj_V1, filters , V2 * np.ones(filters.shape[0] ))

opt_V1 = optimize_V1( obj_V1, filters , V2 * np.ones(filters.shape[0] ), gtol=1e-4 ,
                      lambdas= 0.01 * np.ones(filters.shape[0]), maxiter=700)

############

#opt_V1 = optimize_V1( obj_V1, filters , V2 * np.ones(filters.shape[0] ))
#
#eps=0.03
#iterations[0] = -1
#opt_V1 = optimize_V1( obj_V1, filters , V2 * np.ones(filters.shape[0] ), gtol=1e-3 ,
#                      lambdas=0.01/(np.sqrt(np.sum(opt_V1**2.,axis=0)+eps)),
#                      init=opt_V1**2., maxiter=80)
#
#eps=0.01
#iterations[0] = 0
#opt_V1 = optimize_V1( obj_V1, filters , V2 * np.ones(filters.shape[0] ), gtol=1e-3 ,
#                      lambdas=0.01/(np.sqrt(np.sum(opt_V1**2.,axis=0)+eps)),
#                      init=opt_V1, maxiter=80)


#opt_V1 = optimize_V1( obj_V1, R['U'], V2 * np.ones(Nsub) )

#obj_UV1  = objective_UV1()
#opt_UV1 = optimize_UV1( obj_UV1, opt_V1, R['U'], V2* np.ones(Nsub) )
 
#opt_U = U
#for i in range(5):
#    print 'Optimizing V1'
#    opt_V1 = optimize_V1( opt_U )
#    save(opt_V1,'opt_V1_%d'%(i))
#    print 'Optimizing U'
#    opt_U  = optimize_U( opt_V1, opt_U )
#    save(opt_U,'opt_U_%d'%(i))
#
#testdata = {'STAs':np.vstack(testR['statistics']['stimulus']['STA']) ,
#            'STCs':np.vstack([stc[np.newaxis,:] for stc in testR['statistics']['stimulus']['STC']]), 
#            'V2':V2 , 'N':NRGC , 'N_spikes':testR['N_spikes'] }
#test_objective = obj_UV1.where(**testdata)
#
#print 'Unregularized ARD + alternating full model ll:', test_objective.f( {'V1': opt_UV1['V1'] , 'U': opt_UV1['V1U'] } )
#
#glm = optimize_V1( np.eye(Ncones), V2=np.zeros(V2.shape) )
#testdata['U'] = np.eye(Ncones)
#print 'Unregularized GLM (STA) ll:', obj_V1.where(**testdata).f( glm.flatten() )