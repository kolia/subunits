from functools import partial

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import positive_quadratic_Poisson, UVs , eig_positive_barrier, \
                        eig_barrier, ldet, eigs, positive

import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)

import optimize
reload(optimize)

#from numpy  import arange, sum, dot, min, identity, eye, reshape, concatenate
import numpy as np
from numpy.linalg import norm, slogdet, eig, svd
from scipy.linalg import orth
import numpy.random as R
import pylab as p
from copy import deepcopy

from matplotlib.ticker import *

from IPython.Debugger import Tracer; debug_here = Tracer()

sigma = 1.

V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = simulate_data.LNLNP(NL=NL,N=24)
Nsub, Ncone =  U.shape
NRGC, Nsub  =  V1.shape

A = np.concatenate([N*np.reshape(sta,(1,len(sta))) for N,sta in zip(N_spikes,STA)])
B = np.concatenate([N*stc for N,sta,stc in zip(N_spikes,STA,STC)])
C = np.concatenate((A,B))
YU,S,VI = svd(C)

def subspace(A,B):
    u,s,v = svd(np.dot(np.transpose(orth(A)), orth(B)))
    return np.arccos(min(s.min(), 1.0))

def projection(uu,X):
    X = orth(X)
    return np.dot(X,np.dot(uu,X))



def callback( term , params ):
    print 'Objective: ' , term.f(params) , '  barrier: ', term.barrier(params) 
    #, '  logdet: ', term.ldet(params),  '  eigs: ', term.eigs(params), '  positive: ',term.positive(params)

true = {'U' : U , 'V1': V1 }
data = {'STAs':np.vstack(STA) , 'STCs':np.vstack([stc[np.newaxis,:] for stc in STC]), 
        'V2':V2*np.ones(Nsub) , 'N':NRGC }

targets = { 'f':positive_quadratic_Poisson, 'positive':positive,
            'barrier':eig_positive_barrier, 'ldet':ldet, 'eigs':eigs }

targets = kolia_theano.reparameterize(targets,UVs(NRGC))

term = kolia_theano.term( init_params=true, differentiate=['f'], 
                          callback=callback, **targets )

optimizer = optimize.optimizer( term.where(**data) )

trupar = true
for i in range(2):
    trupar = optimizer(init_params=trupar)
trupar = term.unflat(trupar)


init_params = {'U' :0.001+0.01*R.random(size=U.shape) , 
               'V1':0.001+0.01*R.random(size=V1.shape)}

params = init_params
for i in range(2):
    params = optimizer(init_params=params)
params = term.unflat(params)


def plot_U(params):
    p.figure(3)
    Nsub = len(params)
    for i in range(Nsub):
        di = params[i]
        ax = p.subplot(Nsub,1,i+1)
        theta = di['U']
        p.plot(np.arange(theta.size),theta,'b')
        ax.yaxis.set_major_locator( MaxNLocator(nbins=2) )
        if i == 0:  p.title('Inferred thetas')
        p.show()
    p.xlabel('Cone space')


def plot_matrix(m):
    p.figure(4)
    Nsub = m.shape[0]
    for i in range(Nsub):
        di = m[i,:]
        ax = p.subplot(Nsub,1,i+1)
        p.plot(np.arange(di.size),di,'b')
        ax.yaxis.set_major_locator( MaxNLocator(nbins=2) )
        if i == 0:  p.title('First few SVD components')
        p.show()
    p.xlabel('Cone space')


optU = params['U']
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', np.sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', np.sum(optU*optU)
print


def show(string,p):
    print 'log-likelihood of %s = %f   barrier = %f    ldet = %f     minw = %f' \
        % ( string , objective.f(p,data), objective.barrier(p,data) , 
           objective.ldet(p,data) , np.min(objective.eigs(p,data)) )
#    print 'bar ' , objective.bar(p,data)

#show('init params' ,init_params)
#show('true params' ,true       )
#show('opt params'  ,params     )
##show('opt of true' ,trupar     )
##print 'improvement of opt of true = ', 
##    objective.f(params,data) - objective.f(trupar,data)

#p.figure(2)
#objective.plot(params,U)