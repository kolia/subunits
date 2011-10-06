import kolia_base as kb
reload(kb)

import kolia_theano
reload(kolia_theano)

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
 
V2 = 0.5
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
    sigma_spatial  = [1., 3.]     , V2=V2 )

filters = gaussian2D_weights( model['cones'] , possible_subunits , 
                             sigma=model['sigma_spatial'][0] )

print 'N cones   : ', len(model['cones'   ])
print 'N filters :  ', len(possible_subunits)
print 'N subunits: ', len(model['subunits'])
print 'N RGCs    : ', len(model['RGCs'    ])

pylab.close('all')
pylab.figure(1)
ax = pylab.subplot(1,4,1)
kb.plot_circles( sizes=0.1, offsets=model['cones'],
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
pylab.title('Cones')
pylab.subplot(1,4,2)
ax = kb.plot_circles( sizes=3*model['sigma_spatial'][0], offsets=possible_subunits,
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
pylab.title('Filters')
ax = pylab.subplot(1,4,3)
kb.plot_circles( sizes=3*model['sigma_spatial'][0], offsets=model['subunits'],
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3))
pylab.title('Subunits')
ax = pylab.subplot(1,4,4)
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

#lam2 = 200.
#prior = np.fromfunction( lambda i,j: (i-j>0)*(i-j<6) , Cin.shape)
##prior = np.dot( filters , filters.T )
##prior = (prior - np.diag(np.diag(prior)))
##prior = prior / np.sqrt( np.sum( prior**2. ) )
#Cin = Cin + lam2 * prior

D,Z = schur(Cin)
DD  = np.diag(D)
keep= DD>1e-10
P   =  (Z[:,keep] * np.sqrt(DD[keep])).T
y   =  np.dot ( (Z[:,keep] * 1/np.sqrt(DD[keep])).T , dSTA ) / 2

irls = memory.cache(IRLS)
V, iW = irls( y, P, x=0, disp_every=1000, lam=0.1, maxiter=1000000 , 
              ftol=1e-5, nonzero=1e-1)
print 'V'
kb.print_sparse_rows( V, precision=1e-1 )

keepers = np.array( [sum(abs(v))>3e-1 for v in V] )
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
