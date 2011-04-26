# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:07:10 2011

@author: - kolia
"""


import bayes_LNLExP  ; reload(bayes_LNLExP)
import simulate_data ; reload(simulate_data)
import optimize      ; reload(optimize)

from bayes_LNLExP import posterior, sin_model, exp_model, lin_model
from numpy  import reshape,concatenate,eye,sin,exp,ones
from numpy.linalg import norm, slogdet
import pylab as p


from numpy import arange, sum
import numpy.random   as R

sigma  = 1.1
model  = sin_model(sigma)
#model  = exp_model(sigma)
#model  = lin_model(sigma)
baye   = posterior(model)

data, U, V1 =  simulate_data.LNLNP(sigma=model.sigma,NL=model.NL,N=12)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

UU = U.flatten()
data = [data]


## Check derivative
#print
#print 'Checking derivatives:'
#df = baye.df(UU,data)
#dh = 0.00000001
#ee = eye(len(UU))*dh
#for i in arange(len(UU)):
#    print (baye.f(UU+ee[:,i],data[0])-baye.f(UU,data[0]))/dh , df[i]


print
print 'U  : ', U
print
print 'V1 : ' , V1
print
print baye.callback(UU,data[0])
print
print 'U.shape    = ', U.shape
print 'N_cones    = ', N
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print
print 'true params :'
baye.callback(U,data[0])
#init_params = 0.0001 * R.randn(len(UU))
init_params = 0.1*( ones(len(UU)) + 2.*R.randn(len(UU)) )
#init_params = UU
print
print 'initial params :'
baye.callback(init_params,data[0])
baye.barrier(init_params,data[0])
s,ld = slogdet(baye.Cb(U))
print ' slogdet=', s, exp(ld)
print

params = init_params
for i in arange(3):
    params = baye.MAP(params,data)

optU = reshape(params,(Nsub,N))
print
print 'stimulus sigma  :  ',sigma
print 'true    ||subunit RF||^2  : ', sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', sum(optU*optU,axis=1)
print


print 'log-likelihood of init params = ', baye.f(init_params,data[0])
print 'log-likelihood of opt  params = ', baye.f(params,data[0])
print 'log-likelihood of true params = ', baye.f(UU,data[0])

#print
#print 'init params:'
#print  baye.params(init_params,data[0])[index]
#print 'true params:'
#print  baye.params(true_params,data[0])[index]
#print 'opt  params:'
#print  baye.params(params ,data[0])[index]

p.figure(2)
baye.plot(params,U)