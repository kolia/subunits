import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, eig_det_barrier, det_barrier, eig_barrier

import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)
from simulate_data import LNLNP

import optimize
reload(optimize)

#from numpy  import arange, sum, dot, min, identity, eye, reshape, concatenate
import numpy as np
from numpy.linalg import norm, slogdet, eig, svd
from scipy.linalg import orth
import numpy.random as R
import pylab as p
from copy import deepcopy

from IPython.Debugger import Tracer; debug_here = Tracer()


sigma  = 1.

quad = lambda x : 0.2*(x+1.)**2

(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=sigma,NL=quad,N=27)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape


iii = NRGC
data  = [{ 'STA':STA[i] , \
           'STC':STC[i] } for i in range(iii)]
init_params = [{'theta':data[i]['STA'] * 0.1 , \
                'M':data[i]['STC'] * 0.1} for i in range(iii)]


term = kolia_theano.term(init_params=init_params[0],differentiate=['f'],
                          f=quadratic_Poisson, barrier=eig_barrier)

class Objective: pass
objective = Objective()

def f(params,args):
    return np.sum([term.f(params[i*N*(N+1):(i+1)*N*(N+1)],args[i]) for i in range(iii)],0)

def barrier(params,args):
    return np.sum([term.barrier(params[i*N*(N+1):(i+1)*N*(N+1)],args[i]) for i in range(iii)],0)

def df(params,args):
    return np.concatenate([term.df(params[i*N*(N+1):(i+1)*N*(N+1)],args[i]) for i in range(iii)])

objective.f       = f
objective.df      = df
objective.barrier = barrier

def callback_one(ip,d): pass

optimize  = optimize.optimizer( objective , callback=callback_one )


#true   = [{ 'theta' : np.dot( U.T , V1[i,:] ) , 'M' : 0.1*np.dot( U.T * V1[i,:] , U ) } for i in range(iii)]
#
#for t in true:
#    w,v = eig( np.eye(t['M'].shape[0]) - t['M'] )
#    print 'eig true M' , w.real
#
#trupar = true
#for i in range(5):
#    trupar = optimize(init_params=trupar,args=data)
#    callback_one(trupar,data)
#trupar = objective.inflate(trupar)

params = np.concatenate( [term.flatten(ip) for ip in init_params] )
for i in range(8):
    params = optimize(init_params=params,args=data)
#    callback_one(params,data)
#params = objective.inflate(params)


#optU = [param['theta'] for param in params]
#print
#print 'stimulus sigma  :  ', sigma
#print 'true    ||subunit RF||^2  : ', np.sum(U*U,axis=1)
#print 'optimal ||subunit RF||^2  : ', [np.sum(optu*optu) for optu in optU]
#print
#
#def show(string,p):
#    print 'log-likelihood of %s = %f   barrier = %f    ldet = %f     minw = %f' \
#        % ( string , objective.f(p,data), objective.barrier(p,data) , 
#           objective.ldet(p,data) , np.min(objective.eigs(p,data)) )
##    print 'bar ' , objective.bar(p,data)
#
#show('init params' ,init_params)
#show('true params' ,true       )
#show('opt params'  ,params     )
#show('opt of true' ,trupar     )
#print 'improvement of opt of true    = ', objective.f(params,data) - objective.f(trupar,data)
#
##p.figure(2)
##objective.plot(params,U)