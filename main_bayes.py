# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:07:10 2011

@author: - kolia
"""


import bayes_LNLExP  ; reload(bayes_LNLExP)
import simulate_data ; reload(simulate_data)
from   simulate_data import LNLNP
import optimize      ; reload(optimize)

from bayes_LNLExP import *
from numpy  import reshape,eye,sin,exp,ones,argmax,zeros_like,mod,concatenate
from numpy.linalg import norm, slogdet, svd
import pylab as p


from numpy import arange, sum
import numpy.random as R

alpha = 10.
#prior = lambda U : -0.001*Th.sum( (U - Th.concatenate([U[:,1:],U[:,0:1]],axis=1)) ** 2. ) \
#                   - 0.05 * ( Th.sum( L2mr(U,alpha/2+1)**2 ) + alpha * L2(U) )

#prior = lambda U : - 0.0005 * Th.sum( L2mr(U,1.)**2 )
prior = lambda U : 0.

#prior = lambda U : - 0.001 * ( Th.sum( L2mr(U,alpha/2+1)**2 ) + alpha * L2(U) )
#prior = lambda U : - 0.005 * ( 3.*Th.sum( L2mr(U,1.5)**2 ) + 2.*sL1(   U,0.01) \
#                            + 0.5*L2c(U)                   +    rL1(-5*U,0.1 ) \
#  + 0.2 * sL1 (U - Th.concatenate([U[:,1:],U[:,0:1]],axis=1),0.01) )
#  + 0.1 * Th.sum( (U - Th.concatenate([U[:,1:],U[:,0:1]],axis=1)) ** 2. ) \

#prior = lambda U : 0

sigma  = 0.3
model  = sin_model(sigma)
#model  = quad_model(sigma,0.2,2.)
#model  = exp_model(sigma)
#model  = lin_model(sigma)
baye   = posterior(model,prior)
baye.optimize = optimizer(baye)


data, U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=model.sigma,NL=model.NL,N=27)
Nsub, Ncones =  U.shape
NRGC, Nsub   =  V1.shape
(N_spikes,STA,STC) = data

# Check analytical equations if small problem
if STAB[0].shape[0]<5:
    print 'Theano.STAB', baye.STAB(U,data[1][0],data[2][0])
    print 'Empiri.STAB', STAB[0]
    print
    print 'Theano.bbar', baye.bbar(U)
    print 'Empiri.bbar', bbar
    print
    print 'Theano.Cb  ', baye.Cb(U)
    print 'Empiri.Cb  ', Cb

UU = U.flatten()

#init_params = 0.0001 * R.randn(len(UU))

A = concatenate([N*reshape(sta,(1,len(sta))) for N,sta in zip(N_spikes,STA)])
#B = np.concatenate([N*(stc + np.outer(sta,sta)) for N,sta,stc in zip(N_spikes,STA,STC)])
B = concatenate([N*stc for N,sta,stc in zip(N_spikes,STA,STC)])
C = concatenate((A,B))
YU,S,VI = svd(C)

init_params = 0.0001 * R.randn(U.size)
flat_svdU = VI[0:Nsub,:].flatten()
init_params = concatenate( [flat_svdU , init_params[len(flat_svdU):]] )

#init_params = zeros_like(U)

#permute = R.permutation( init_params.shape[1] )
##permute = arange(init_params.shape[1])
#for i,j in enumerate( argmax(U,axis=1) ):
#    init_params[i][permute[j]] = 1.
 
#for i in arange( init_params.shape[0] ):
#    init_params[i][mod(i+20,N)] = 1.

#init_params = 0.1*( ones(len(UU)) + 2.*R.randn(len(UU)) )
#init_params = UU


print
print 'U  : ', U
print
print 'V1 : ' , V1
print
print baye.callback(UU,data)
print
print 'U.shape    = ', U.shape
print 'N_cones    = ', N
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print
print 'true params :'
baye.callback(U,data)
print
print 'initial params :'
baye.callback(init_params,data)
baye.barrier(init_params,data)
s,ld = slogdet(baye.Cb(U))
print ' slogdet=', s, exp(ld)
print

trupar = UU
for i in arange(1):
    trupar = baye.optimize(init_params=trupar,args=data)

UUUUUU = concatenate( (U[:,-1:],U[:,:-1]) , 1)
shipar = UUUUUU.flatten()
for i in arange(1):
    shipar = baye.optimize(init_params=shipar,args=data)


params = init_params.flatten()
for i in arange(1):
    params = baye.optimize(init_params=params,args=data)


optU = reshape(params,(Nsub,Ncones))
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', sum(optU*optU,axis=1)
print

print 'log-likelihood of init params = ', baye.f(init_params,data)
print 'log-likelihood of true params = ', baye.f(UU,data)
print 'log-likelihood of opt  params = ', baye.f(params,data)
print 'log-likelihood of opt of true = ', baye.f(trupar,data)
print 'log-likelihood of shift  true = ', baye.f(shipar,data)
print 'disadvantage of shift true    = ', baye.f(shipar,data) - baye.f(trupar,data)
print 'improvement of opt of true    = ', baye.f(params,data) - baye.f(trupar,data)

p.figure(2)
baye.plot(params,U)