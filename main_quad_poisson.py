import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, barrier, quad_term, data_match, ldet, eigs, bar

import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)
from simulate_data import LNLNP

import optimize
reload(optimize)

from numpy  import arange, sum, dot, mean, min, max, identity, eye
from numpy.linalg import norm, slogdet, eig
import numpy.random as R
import pylab as p

sigma  = 1.

quad = lambda x : 0.2*(x+1.)**2

(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=sigma,NL=quad,N=27)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

iii = 0

data  = { 'STA':STA[iii] , 'STC':STC[iii] }

init_params = {}
init_params['theta'] = STA[iii] * 0.1
init_params['M'    ] = STC[iii] * 0.1

#init_params = 0.1*( ones(len(UU)) + 2.*R.randn(len(UU)) )
#init_params = UU


objective = kolia_theano.objective(quadratic_Poisson,init_params=init_params,barrier=barrier,data_match=data_match,quad_term=quad_term,ldet=ldet,eigs=eigs,bar=bar)

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
    M = pp['M']
    N = M.shape[0]
    s , ldet = slogdet(identity(N)-M)
    df = objective.df(pp,d)
    dM = objective.inflate(df)['M']
    ds , dldet = slogdet(identity(N)-M+0.001*dM)
    w,v = eig( identity(N) - M )
    print 'eig M' , w.real
    print [objective.f(ip,data),objective.ldet(ip,data),objective.quad_term(ip,data),objective.data_match(ip,data)]
    print 'Iteration s, ldet I-M: %d , %f     %d , %f     norm theta %f    norm M %f   barr %d' % \
          (s , ldet , ds, dldet, norm(pp['theta']), norm(M), objective.barrier(ip,data))
    print

optimize  = optimize.optimizer( objective , callback=callback )

true   = { 'theta' : dot( U.T , V1[iii,:] ) , 'M' : dot( U.T * V1[iii,:] , U ) }

w,v = eig( true['M'] )
print 'eig true M' , w.real

trupar = true
for i in arange(5):
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

def show(string,p):
    print 'log-likelihood of %s = %f   barrier = %f    ldet = %f     minw = %f' \
        % ( string , objective.f(p,data), objective.barrier(p,data) , objective.ldet(p,data) , min(objective.eigs(p,data)) )
#    print 'bar ' , objective.bar(p,data)

show('init params' ,init_params)
show('true params' ,true       )
show('opt params'  ,params     )
show('opt of true' ,trupar     )
print 'improvement of opt of true    = ', objective.f(params,data) - objective.f(trupar,data)

#p.figure(2)
#objective.plot(params,U)