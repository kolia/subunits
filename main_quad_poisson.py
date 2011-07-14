import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)

import optimize
reload(optimize)

import numpy as np
from numpy.linalg import svd
from scipy.linalg import orth
import numpy.random as R
import pylab as p

from matplotlib.ticker import *

from IPython.Debugger import Tracer; debug_here = Tracer()

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, eig_barrier


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

iii = NRGC
data  = [{ 'STA':np.dot(T,STA[i]) , \
           'STC':np.dot(np.dot(T,STC[i]),np.transpose(T)) } for i in range(iii)]
init_params = [{'theta':data[i]['STA'] * 0.1 , \
                'M':data[i]['STC'] * 0.1} for i in range(iii)]

#iii = NRGC
#data  = [{ 'STA':STA[i] , \
#           'STC':STC[i] } for i in range(iii)]
#init_params = [{'theta':data[i]['STA'] * 0.1 , \
#                'M':data[i]['STC'] * 0.1} for i in range(iii)]


def callback( objective , params ):
    print 'Objective: ' , objective.f(params) , '  barrier: ', objective.barrier(params)

objective = kolia_theano.Objective(init_params=init_params[0],differentiate=['f'],callback=callback,
                          f=quadratic_Poisson, barrier=eig_barrier)

optimizers = [ optimize.optimizer( objective.where(**dat) ) for dat in data ]

true   = [{ 'theta' : np.dot( U.T , V1[i,:] ) , \
            'M' : 0.1*np.dot( U.T * V1[i,:] , U ) } for i in range(iii)]

params = init_params
for i in range(2):
    params = [opt(init_params=par) for opt,par in zip(optimizers,params)]
params = [objective.unflat(par) for par in params]


def plot_thetas(params):
    p.figure(3)
    Nsub = len(params)
    for i in range(Nsub):
        di = params[i]
        ax = p.subplot(Nsub,1,i+1)
#        theta = np.dot(di['theta'],T)
        theta = di['theta']
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


optU = [param['theta'] for param in params]
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', np.sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', [np.sum(optu*optu) for optu in optU]
print

exparams = [{'theta': np.dot(q['theta'],T)  ,  \
             'M':np.dot(np.dot(np.transpose(T),q['M']),T)} for q in params]

exA = np.concatenate([np.concatenate([np.array([q['theta']]),q['M']]) for q in exparams])

exUa,exSa,exVa = svd(exA)

p.figure(5)
nnn = U.shape[1]
for i in range(Nsub):
    ax = p.subplot(Nsub,1,i+1)
    p.plot(np.arange(nnn),U[i,:].flatten(),'b',np.arange(nnn), 
           projection(np.transpose(U[i,:]),np.transpose(exVa[0:9,:])).flatten(),'rs')
    if i==0: p.title('Subunit RFs and projection onto svd(theta,M)')
    p.show()
p.xlabel('Cone space')


p.figure(6)
nnn = U.shape[1]
for i in range(Nsub):
    ax = p.subplot(Nsub,1,i+1)
    p.plot(np.arange(nnn),U[i,:].flatten(),'b',np.arange(nnn), 
           projection(np.transpose(U[i,:]),np.transpose(VI[0:Nsub,:])).flatten(),'rs')
    if i==0: p.title('Subunit RFs and projection onto svd(STA,STC)')
    p.show()
p.xlabel('Cone space')
