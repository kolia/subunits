import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, UVs , eig_barrier

import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)

import optimize
reload(optimize)

import numpy as np
import numpy.random as R
import pylab as p

from matplotlib.ticker import *

from IPython.Debugger import Tracer; debug_here = Tracer()

sigma = 1.

V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = simulate_data.LNLNP(NL=NL,N=24)
Nsub, Ncone =  U.shape
NRGC, Nsub  =  V1.shape


def callback( objective , params ):
    print 'Objective: ' , objective.f(params) , '  barrier: ', objective.barrier(params)


true = {'U' : U , 'V1': V1 }
data = {'STAs':np.vstack(STA) , 'STCs':np.vstack([stc[np.newaxis,:] for stc in STC]), 
        'V2':V2*np.ones(Nsub) , 'N':NRGC }

targets = { 'f':quadratic_Poisson, 'barrier':eig_barrier }

targets = kolia_theano.reparameterize(targets,UVs(NRGC))

objective = kolia_theano.Objective( init_params=true, differentiate=['f'], 
                          callback=callback, **targets )

optimizer = optimize.optimizer( objective.where(**data) )

trupar = true
for i in range(2):
    trupar = optimizer(init_params=trupar)
trupar = objective.unflat(trupar)


init_params = {'U' : 0.0001+0.05*R.random(size=U.shape ) ,
               'V1': 0.0001+0.05*R.random(size=V1.shape) }

params = init_params
for i in range(10):
    params = optimizer(init_params=params)
params = objective.unflat(params)


def plot_U(params,trupar=None):
    p.figure(3)
    U = params['U']
    if trupar is not None: trupar=trupar['U']
    Nsub = U.shape[0]
    for i in range(Nsub):        
        theta = U[i]
        x  = np.arange(theta.size)
        ax = p.subplot(Nsub,1,i+1)
        if trupar is not None:
            p.plot(x,theta,'b',x,trupar[i],'rs')
        else:
            p.plot(x,theta,'b')
        ax.yaxis.set_major_locator( MaxNLocator(nbins=2) )
        if i == 0:  p.title('Inferred U')
        p.show()
    p.xlabel('Cone space')


optU = params['U']
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', np.sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', np.sum(optU*optU)
print

plot_lU(params,trupar=trupar)