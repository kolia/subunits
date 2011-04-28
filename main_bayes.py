# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:07:10 2011

@author: - kolia
"""


import bayes_LNLExP  ; reload(bayes_LNLExP)
import simulate_data ; reload(simulate_data)
import optimize      ; reload(optimize)

from bayes_LNLExP import *
from numpy  import reshape,concatenate,eye,sin,exp,ones
from numpy.linalg import norm, slogdet
import pylab as p


from numpy import arange, sum
import numpy.random   as R

sigma  = 1.
#model  = sin_model(sigma)
model  = quad_model(sigma,0.5)
#model  = exp_model(sigma)
#model  = lin_model(sigma)
baye   = posterior(model)

data, U, V1 =  simulate_data.LNLNP(sigma=model.sigma,NL=model.NL,N=6)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

UU = U.flatten()

#init_params = 0.0001 * R.randn(len(UU))
init_params = 0.1*( ones(len(UU)) + 3.*R.randn(len(UU)) )
#init_params = UU

## Check derivative
#print
#print 'Checking derivatives:'
#df = baye.df(init_params,data)
#dh = 0.00000001
#ee = eye(len(init_params))*dh
#for i in arange(len(init_params)):
#    print (baye.f(init_params+ee[:,i],data)-baye.f(init_params,data))/dh,df[i]


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

params = init_params
for i in arange(5):
    params = baye.MAP(params,[data])

optU = reshape(params,(Nsub,N))
print
print 'stimulus sigma  :  ',sigma
print 'true    ||subunit RF||^2  : ', sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', sum(optU*optU,axis=1)
print


print 'log-likelihood of init params = ', baye.f(init_params,data)
print 'log-likelihood of opt  params = ', baye.f(params,data)
print 'log-likelihood of true params = ', baye.f(UU,data)

#print
#print 'init params:'
#print  baye.params(init_params,data[0])[index]
#print 'true params:'
#print  baye.params(true_params,data[0])[index]
#print 'opt  params:'
#print  baye.params(params ,data[0])[index]

p.figure(2)
baye.plot(params,U)