import kolia_base as kb 
reload(kb) 

import simulate_retina as retina
reload(retina)

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, UVs , eig_barrier, LNLEP
import pylab as p

import kolia_theano
reload(kolia_theano)

import optimize
reload(optimize)
 
import numpy.random as Rand
import copy 
import numpy        as np 

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

from IPython.Debugger import Tracer; debug_here = Tracer()
 
iterations = [0]
def callback( objective , params ):
#    fval = objective.f(params)
    print ' Iter:', iterations[0] # ,' Obj: ' , fval
    iterations[0] = iterations[0] + 1


def LL_V1( v1 ):
    targets = { 'f':quadratic_Poisson, 'LL':LNLEP , 'barrier':eig_barrier }
    targets = kb.reparameterize(targets,UVs(N_cells[2]))
    return    kolia_theano.Objective( init_params={'V1': v1 }, differentiate=['f'], 
                                       mode='FAST_COMPILE' , **targets )

def LL_with_U( objective , u , v2 ):
    data = {'STAs':np.vstack(true_run['statistics']['stimulus']['STA']) , 
            'STCs':np.vstack([stc[np.newaxis,:] 
                              for stc in true_run['statistics']['stimulus']['STC']]), 
            'V2':v2 , 'U': u , 'N':true_run['model']['N_cells'][2] ,
            'N_spikes':true_run['N_spikes'] , 'Ncones':true_run['model']['N_cells'][0] }
    return objective.where(**data).with_callback(callback)

def optimize_V1( objective, true_run, u , v2, gtol=1e-4 , maxiter=500):
    optimizer = optimize.optimizer( objective )
    init_params = {'V1': 0.001 + 0.01 * Rand.rand(*true_run['model']['V'].shape) }
    params = optimizer(init_params=init_params,maxiter=maxiter,gtol=gtol)
    return objective.unflat(params)
                                      
def phaseout( objective, true_run , stimulus , V2 , phase ):
    iterations[0] = -1
    model = copy.deepcopy( true_run['model'] )
    model['U'] = retina.ring_weights( shape=true_run['model']['U'].shape, 
                                      offset_out=phase)
    phase_objective = LL_with_U( objective , model['U'] , V2 )
    true_objective  = LL_with_U( objective , true_run['model']['U'] , V2 )
    params      = optimize_V1( phase_objective , true_run , model['U'], V2 )
    opt_params  = optimize_V1( true_objective  , true_run , true_run['model']['V'], V2 )
    model['V']  = params['V1']
    dLopt = true_objective.LL(true_run['model']['V']) - true_objective.LL(opt_params)
    dLL   = phase_objective.LL(params) - true_objective.LL(opt_params)
    seed        = Rand.get_state()
    true_100    = retina.run_LNLNP( true_model , stimulus = stimulus , 
                                    keep=frozenset(['intensity']), N_timebins = 100000 )
    Rand.set_state(seed)
    this_100 = retina.run_LNLNP( model , stimulus = stimulus , 
                                 keep=frozenset(['intensity']), N_timebins = 100000 )
    intensities = np.vstack([true_100['intensity'].flatten(),this_100['intensity'].flatten()])
    order       = np.argsort( intensities[0,:] )
    intensities = intensities[:,order[::20]]
    p.close('all')
    p.figure(1, figsize=(10,12))
    ax0 = p.subplot(3,1,1)
    p.plot( true_100['model']['V'].T )
    #ax0.xaxis.set_label_text('subunits')
    p.title('V filters')
    ax1 = p.subplot(3,1,2)
    p.plot( true_100['model']['U'].T )
    #ax1.xaxis.set_label_text('cones')
    p.title('True U filters ,  sigma = %.0f' % (sigma_spatial[0]))
    ax2 = p.subplot(3,1,3)
    p.semilogy( intensities.T )
    ax2.xaxis.set_label_text('samples reordered by intensity of true model')
    ax2.yaxis.set_label_text('intensities in spikes / bin')
    p.title('Intensities of true vs. out-of-phase model  (dLL=%.1e bits/spike, dLopt=%.1e) ' % \
             (dLL/np.log(2.), dLopt/np.log(2.)) )
    #p.subplot(2,1,2)
    #p.plot( intensities.T )
    p.savefig('/Users/kolia/Desktop/out-of-phase_sigma%d_phase%.1f.pdf' % \
             (true_run['model']['sigma_spatial'][0], phase),format='pdf')

N_cells       = [ 20 , 13 , 7 ]
V2            =  0.5 * np.ones(N_cells[1])
def NL(x): return x + 0.5 * (V2 * ( x ** 2 ).T).T

for sigma_spatial in [[ 20., 3. ] , [ 10., 3. ] , [ 5., 3. ] , [ 2., 3. ] , [1.,3.]]:
    true_model = retina.LNLNP_ring_model( nonlinearity = NL , N_cells = N_cells , 
                                           sigma_spatial = sigma_spatial )
    
    stimulus = retina.Stimulus( retina.white_gaussian )
    true_run = retina.run_LNLNP( true_model , stimulus = stimulus , 
                             keep=frozenset(['intensity']), N_timebins = 1000000 )
    
    obj_V1 = LL_V1( true_model['V'] )
    
    phaseout( obj_V1, true_run , stimulus , V2 , 0.5 )