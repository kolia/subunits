import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, barrier, eigs, ldet

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
from copy import deepcopy

from IPython.Debugger import Tracer; debug_here = Tracer()


sigma  = 1.

quad = lambda x : 0.2*(x+1.)**2

(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=sigma,NL=quad,N=27)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

data = {}
init_params = {}


#init_params = 0.1*( ones(len(UU)) + 2.*R.randn(len(UU)) )
#init_params = UU

iii = 2
data  = [{ 'STA':STA[i] , 'STC':STC[i] } for i in range(iii)]
init_params = [{'theta':STA[i] * 0.1 , 'M':STC[iii] * 0.1} for i in range(iii)]

term = kolia_theano.term(init_params=init_params[0],differentiate=['f'],
                          f=quadratic_Poisson, barrier=barrier, ldet=ldet, eigs=eigs)

terms = [deepcopy(term) for i in range(iii)]

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
    pp = objective.inflate(ip)
    print
    print 'CALLBACK:'
    for term,ppp,dd in zip(terms,pp,d):
        M = ppp['M']
        N = M.shape[0]
        s , lldet = slogdet(identity(N)-M)
        df = term.df(ppp,dd)
        dM = term.inflate(df)['M']
        ds , dldet = slogdet(identity(N)-M+0.001*dM)
        w,v = eig( identity(N) - M )
        print 'eig M' , w.real
        print [term.f(ppp,dd)]
        print 'Iteration s, ldet I-M: %d , %f     %d , %f     norm theta %f    norm M %f   barr %d' % \
              (s , lldet , ds, dldet, norm(ppp['theta']), norm(M), term.barrier(ppp,dd))
        print

optimize  = optimize.optimizer( objective , callback=callback_one )

true   = [{ 'theta' : dot( U.T , V1[i,:] ) , 'M' : 0.1*dot( U.T * V1[i,:] , U ) } for i in range(iii)]

for t in true:
    w,v = eig( eye(t['M'].shape[0]) - t['M'] )
    print 'eig true M' , w.real

trupar = true
for i in arange(5):
    trupar = optimize(init_params=trupar,args=data)
    callback_one(trupar,data)
trupar = objective.inflate(trupar)

params = init_params
for i in arange(2):
    params = optimize(init_params=params,args=data)
    callback_one(params,data)
params = objective.inflate(params)


optU = [param['theta'] for param in params]
print
print 'stimulus sigma  :  ', sigma
print 'true    ||subunit RF||^2  : ', sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', [sum(optu*optu) for optu in optU]
print

def show(string,p):
    print 'log-likelihood of %s = %f   barrier = %f    ldet = %f     minw = %f' \
        % ( string , objective.f(p,data), objective.barrier(p,data) , 
           objective.ldet(p,data) , min(objective.eigs(p,data)) )
#    print 'bar ' , objective.bar(p,data)

show('init params' ,init_params)
show('true params' ,true       )
show('opt params'  ,params     )
show('opt of true' ,trupar     )
print 'improvement of opt of true    = ', objective.f(params,data) - objective.f(trupar,data)

#p.figure(2)
#objective.plot(params,U)