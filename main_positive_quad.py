import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, eig_barrier, ldet, eigs

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


data = {}
init_params = {}


Nproj = Nsub
T    = VI[0:Nproj,:]

#iii = NRGC
#data  = [{ 'STA':np.dot(T,STA[i]) , \
#           'STC':np.dot(np.dot(T,STC[i]),np.transpose(T)) } for i in range(iii)]
#init_params = [{'theta':data[i]['STA'] * 0.1 , \
#                'M':data[i]['STC'] * 0.1} for i in range(iii)]

iii = NRGC
data  = [{ 'STA': STA[i] , \
           'STC': STC[i] ,
           'V2' : V2*np.ones(Nsub) } for i in range(iii)]
init_params = [{'U' : U , \
                'V1': V1[i,:].flatten() } for i in range(iii)]


def callback( term , params ):
    print 'Objective: ' , term.f(params) , '  barrier: ', term.barrier(params)


f = quadtaric_Poisson(**UV())

term = kolia_theano.term(init_params=init_params[0],differentiate=['f'],callback=callback,
                         f=quadtaric_Poisson(**UV()),
                         barrier=eig_barrier(**UV()),
                         ldet=ldet(**UV()), eigs=eigs(**UV()))

optimizers = [ optimize.optimizer( term.where(**dat) ) for dat in data ]

true   = [{'U' : U , \
                'V1': V1[i,:].flatten() } for i in range(iii)]

params = init_params
for i in range(2):
    params = [opt(init_params=par) for opt,par in zip(optimizers,params)]
params = [term.unflat(par) for par in params]


def plot_U(params):
    p.figure(3)
    Nsub = len(params)
    for i in range(Nsub):
        di = params[i]
        ax = p.subplot(Nsub,1,i+1)
#        theta = np.dot(di['theta'],T)
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


optU = [param['U'] for param in params]
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', np.sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', [np.sum(optu*optu) for optu in optU]
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