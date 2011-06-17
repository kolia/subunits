import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, barrier

import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)
from simulate_data import LNLNP

import optimize
reload(optimize)

from numpy  import reshape,eye,sin,exp,ones,argmax,zeros_like,mod, \
            arange, sum, concatenate, dot, mean, min, max
from numpy.linalg import norm, slogdet
import numpy.random as R
import pylab as p

sigma  = 0.3

quad = lambda x : 0.5*(x+1.)**2

(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=sigma,NL=quad,N=27)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

iii = 0

data  = { 'STA':STA[iii] , 'STC':STC[iii] }

init_params = {}
init_params['theta'] = STA[iii] * 0.1
init_params['M'    ] = STC[iii] * 0.00001

#init_params = 0.1*( ones(len(UU)) + 2.*R.randn(len(UU)) )
#init_params = UU

objective = kolia_theano.objective(quadratic_Poisson,init_params=init_params,barrier=barrier)

## Check derivative
#print
#print 'Checking derivatives:'
#pp = objective.flatten(init_params)
#df = objective.df(pp,data)
#dh = 0.00000001
#ee = eye(len(pp))*dh
#for i in arange(len(pp)):
#    print (objective.f(pp+ee[:,i],data)-objective.f(pp,data))/dh,df[i]

def callback(ip,d):
    pp = objective.inflate(ip)
    s , ldet = slogdet(pp['M'])
    df = objective.df(pp,d)
    print [objective.f(ip-0.00000001*dp*df,d) for dp in [-1.,0.,1.,2.]]
    print 'Iteration s, ldet M: %d , %f' % (s,ldet)

optimize  = optimize.optimizer( objective , callback=callback )

true   = { 'theta' : dot( U.T , V1[iii,:] ) , 'M' : dot( U.T * V1[iii,:] , U ) }
trupar = true
for i in arange(2):
    trupar = optimize(init_params=trupar,args=data)
    callback(trupar,data)
trupar = objective.inflate(trupar)

params = init_params
for i in arange(2):
    params = optimize(init_params=params,args=data)
    callback(params,data)
params = objective.inflate(params)


optU = params['theta']
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', sum(optU*optU)
print

print 'log-likelihood of init params = ', objective.f(init_params,data)
print 'log-likelihood of true params = ', objective.f(true,data)
print 'log-likelihood of opt  params = ', objective.f(params,data)
print 'log-likelihood of opt of true = ', objective.f(trupar,data)
print 'improvement of opt of true    = ', objective.f(params,data) - objective.f(trupar,data)

#p.figure(2)
#objective.plot(params,U)