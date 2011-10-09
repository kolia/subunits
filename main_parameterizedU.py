import kolia_base as kb
reload(kb)

import kolia_theano
reload(kolia_theano)

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, UVs , eig_barrier , linear_reparameterization, eigsM, invM, logdetIM, log_detIM

import simulate_retina
reload(simulate_retina)
from simulate_retina import *

import optimize
reload(optimize)

import IRLS
reload(IRLS)
from   IRLS import IRLS

import numpy        as np
from scipy.linalg   import schur

import pylab

from IPython.Debugger import Tracer; debug_here = Tracer()

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

############################
# Setting up simulated data

V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

# Quantities of interest
S = 10.

possible_subunits = hexagonal_2Dgrid( spacing=1. , field_size_x=S , field_size_y=S )
overcompleteness  = 6
subunits          = possible_subunits[::overcompleteness]

model = LNLEP_gaussian2D_model(
    cones    = hexagonal_2Dgrid( spacing=1. , field_size_x=S , field_size_y=S ) ,
    subunits = subunits,
    RGCs     = hexagonal_2Dgrid( spacing=2. , field_size_x=S , field_size_y=S ) ,
    nonlinearity   = NL           ,  # subunit nonlinearity
    sigma_spatial  = [1.4, 2.]     , V2=V2 )
 
filters = gaussian2D_weights( model['cones'] , possible_subunits , 
                             sigma=model['sigma_spatial'][0] )

NRGC = len(model['RGCs'])
print 'N cones   : ', len(model['cones'   ])
print 'N filters :  ', len(possible_subunits)
print 'N subunits: ', len(model['subunits'])
print 'N RGCs    : ', NRGC

pylab.close('all')
fig = pylab.figure(1)
fig.frameon = False
fig.figurePatch.set_alpha(0.0)
ax  = pylab.subplot(1,3,1)
#ax  = fig.add_axes((0,0,1,1))
kb.plot_circles( sizes=0.1, offsets=model['cones'],
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
pylab.title('Cones')
#pylab.subplot(1,3,2)
#ax = kb.plot_circles( sizes=3*model['sigma_spatial'][0], offsets=possible_subunits,
#                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
#pylab.title('Filters')
ax = pylab.subplot(1,3,2)
kb.plot_circles( sizes=3*model['sigma_spatial'][0], offsets=model['subunits'],
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
pylab.title('Subunits')
ax = pylab.subplot(1,3,3)
kb.plot_circles( sizes=3*model['sigma_spatial'][1], offsets=model['RGCs'],
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
pylab.title('RGCs')
pylab.savefig('/Users/kolia/Desktop/retina.pdf',format='pdf')


# Generate stimulus , spikes , and (STA,STC,mean,cov) of quantities of interest
def simulator( model , N_timebins , candidate_subunits ):
    stimulus = simulate_retina.Stimulus( simulate_retina.white_gaussian )
    bigU     = gaussian2D_weights( model['cones'], candidate_subunits , 
                                   sigma=model['sigma_spatial'][0] )
    return simulate_retina.run_LNLEP( model , stimulus = stimulus , 
                   N_timebins = N_timebins ,
                   average_me = {'features':lambda x: 
                       model['nonlinearity'](np.dot(bigU,x))} )
simulate = memory.cache(simulator)
R = simulate( model , 1000000 , possible_subunits )
 

total = float( np.sum([Nspikes for Nspikes in R['N_spikes']]) )

Nspikes = R['N_spikes']/total

dSTA  = np.concatenate(
        [STA[:,np.newaxis]-R['statistics']['features']['mean'][:,np.newaxis]
        for STA in R['statistics']['features']['STA']], axis=1)

Cin = R['statistics']['features']['cov']/2

D,Z = schur(Cin)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA ) / 2

irls = memory.cache(IRLS)
V, iW = irls( y, P, x=0, disp_every=1000, lam=0.2, maxiter=1000000 , 
              ftol=1e-5, nonzero=1e-1)
print 'V'
kb.print_sparse_rows( V, precision=1e-1 )

keepers = np.array( [sum(abs(v))>3.e-1 for v in V] )
U        = filters[keepers,:]
V1       = V[keepers,:].T

inferred_locations = [possible_subunits[i] for i in np.nonzero(keepers)[0]]
sum_V1   = kb.values_to_alpha( np.sum( V1 , axis=0  ) , (0.5,0.,0.) )

pylab.figure(2)
kb.plot_circles( sizes=model['sigma_spatial'][0]*2., offsets=inferred_locations,
                 facecolors=sum_V1, edgecolors=(0.,0.,0.,0.))
kb.plot_circles( sizes=model['sigma_spatial'][0]*2., offsets=model['subunits'],
                 facecolors=(0.,0.,0.,0.), edgecolors=(0.,0.,0.,0.2))
pylab.title('True and inferred subunit locations')
pylab.savefig('/Users/kolia/Desktop/subunits&inferred.pdf',format='pdf')

pylab.close('all')




def index( sequence = [] , f = lambda _: True ):
    """Return the index of the first item in seq where f(item) == True."""
    return next((i for i in xrange(len(sequence)) if f(sequence[i])), None)

def radial_piecewise_linear( nodes=[] , values=[] , default=0.):
    '''Returns a function which does linear interpolation between a sequence of nodes. 
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

#nodes        = numpy.arange(0.,4.,0.05)
nodes = numpy.array([ 0. ,  1.  ,  1.5,  2. ,  2.5 ,  3. ,  3.5 ,  4.5])

value_matrix = numpy.eye(len(nodes))
shapes = [ radial_piecewise_linear(nodes,values) for values in value_matrix]

init_u = numpy.exp(-0.5*(nodes**2))
#init_u = numpy.ones(nodes.shape)
init_u = init_u/numpy.sqrt(numpy.sum(init_u**2.))

def objective_u( u=init_u ):
    targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }
    targets = kb.reparameterize(targets,UVs(NRGC))
    list_targets= kb.reparameterize({'eigsM':eigsM, 'invM':invM, 'logdetIM':logdetIM, 'log_detIM':log_detIM },UVs(NRGC),
                                               reducer=lambda r,x: r + [x], zero=[])
    targets.update( list_targets )
    targets = kb.reparameterize(targets,linear_reparameterization)
    return    kolia_theano.Objective( init_params={'u': u }, differentiate=['f'],
                                       mode='FAST_COMPILE' , **targets )
obj_u   = objective_u()

iterations = [0]
def callback( objective , params ):
#    fval = objective.f(params)
    print ' Iter:', iterations[0] # ,' Obj: ' , fval
    if np.remainder( iterations[0] , 3 ) == 0:
        result = objective.unflat(params)
        if np.remainder( iterations[0] , 7 ) == 0: pylab.close('all')
        pylab.figure(1, figsize=(10,12))
        pylab.plot(nodes,result['u'])
        pylab.title('u params')
#        p.savefig('/Users/kolia/Desktop/u.svg',format='svg')
        pylab.savefig('/Users/kolia/Desktop/u.pdf',format='pdf')
    iterations[0] = iterations[0] + 1

@memory.cache
def optimize_u( v1, init_u, v2 , T, gtol=1e-7 , maxiter=500):
    data = {'STAs':np.vstack(R['statistics']['stimulus']['STA']) ,
            'STCs':np.vstack([stc[np.newaxis,:] for stc in R['statistics']['stimulus']['STC']]), 
            'V2':v2 , 'V1': v1 , 'N':NRGC , 'N_spikes':R['N_spikes'] , 'T': T}
    obj = obj_u.where(**data).with_callback(callback)
    optimizer = optimize.optimizer( obj )
    # debug_here()
    params = optimizer(init_params={'u': init_u },maxiter=maxiter,gtol=gtol)
    opt_u = obj_u.unflat(params)
    return opt_u['u']


iterations[0] = -1

T = place_cells( model['cones'] , inferred_locations , shapes )
opt_u = optimize_u( V1, init_u, V2*np.ones(V1.shape[1]) , T=T , gtol=1e-6)
