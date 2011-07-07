import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, eig_barrier, ldet, eigs

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
from copy import copy

from IPython.Debugger import Tracer; debug_here = Tracer()


sigma  = 1.

quad = lambda x : 0.2*(x+1.)**2

(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=sigma,NL=quad,N=27)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

A = np.concatenate([N*np.reshape(sta,(1,len(sta))) for N,sta in zip(N_spikes,STA)])
#B = np.concatenate([N*(stc + np.outer(sta,sta)) for N,sta,stc in zip(N_spikes,STA,STC)])
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


#init_params = 0.1*( ones(len(UU)) + 2.*R.randn(len(UU)) )
#init_params = UU


#Nproj = Nsub
#T    = VI[0:Nproj,:]
#STA  = [np.dot(T,sta) for sta in STA]
#STC  = [np.dot(np.dot(T,stc),np.transpose(T)) for stc in STC]
#
#iii = NRGC
#data  = [{ 'STA':np.dot(T,STA[i]) , \
#           'STC':np.dot(np.dot(T,STC[i]),np.transpose(T)) } for i in range(iii)]
#init_params = [{'theta':data[i]['STA'] * 0.1 , \
#                'M':data[i]['STC'] * 0.1} for i in range(iii)]


iii = NRGC
data  = [{ 'STA':STA[i] , \
           'STC':STC[i] } for i in range(iii)]
init_params = [{'theta':data[i]['STA'] * 0.1 , \
                'M':data[i]['STC'] * 0.1} for i in range(iii)]


term = kolia_theano.term(init_params=init_params[0],differentiate=['f'],
                          f=quadratic_Poisson, barrier=eig_barrier, ldet=ldet, eigs=eigs)

#terms = [deepcopy(term) for i in range(iii)]
terms = [copy(term) for i in range(iii)]

objective = kolia_theano.sum_objective(terms)


## Check derivative
#print
#print 'Checking derivatives:'
#pp = objective.flatten(init_params)
#df = objective.df(pp,data)
#dh = 0.00000001
#ee = eye(len(pp))*dh
#for i in arange(len(pp)):
#    print (objective.f(pp+ee[:,i],data)-objective.f(pp,data))/dh,df[i]


def callback_one(ip,d):
    print 'Objective: ' , objective.f(ip,d) , '  barrier: ', objective.barrier(ip,d)
#    pp = objective.inflate(ip)
#    print
#    print 'CALLBACK:'
#    for term,ppp,dd in zip(terms,pp,d):
#        M = ppp['M']
#        N = M.shape[0]
#        s , lldet = slogdet(np.identity(N)-M)
#        df = term.df(ppp,dd)
#        dM = term.inflate(df)['M']
#        ds , dldet = slogdet(np.identity(N)-M+0.001*dM)
##        w,v = eig( np.identity(N) - M )
##        print 'eig M' , w.real
##        print [term.f(ppp,dd)]
#        print 'Iteration s, ldet I-M: %d , %f     %d , %f     norm theta %f    norm M %f   barr %d' % \
#              (s , lldet , ds, dldet, norm(ppp['theta']), norm(M), term.barrier(ppp,dd))
#        print

optimize  = optimize.optimizer( objective , callback=callback_one )

true   = [{ 'theta' : np.dot( U.T , V1[i,:] ) , 'M' : 0.1*np.dot( U.T * V1[i,:] , U ) } for i in range(iii)]

#for t in true:
#    w,v = eig( np.eye(t['M'].shape[0]) - t['M'] )
#    print 'eig true M' , w.real

trupar = true
for i in range(5):
    trupar = optimize(init_params=trupar,args=data)
#    callback_one(trupar,data)
trupar = objective.inflate(trupar)

params = init_params
for i in range(5):
    params = optimize(init_params=params,args=data)
#    callback_one(params,data)
params = objective.inflate(params)


optU = [param['theta'] for param in params]
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

show('init params' ,init_params)
show('true params' ,true       )
show('opt params'  ,params     )
show('opt of true' ,trupar     )
print 'improvement of opt of true    = ', objective.f(params,data) - objective.f(trupar,data)

#p.figure(2)
#objective.plot(params,U)