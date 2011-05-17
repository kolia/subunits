import bayes_LNLExP  ; reload(bayes_LNLExP)
import simulate_data ; reload(simulate_data)
from   simulate_data import LNLNP
import optimize      ; reload(optimize)

from bayes_LNLExP import *
from numpy  import reshape,concatenate,eye,ones,zeros,zeros_like,abs,mod
from numpy.linalg import norm, slogdet
import pylab as p


from numpy import arange, sum
import numpy.random   as R


#prior = lambda U : - 0.001 * ( Th.sum( L2mr(U,alpha/2+1)**2 ) + alpha * L2(U) )
prior = lambda U : - 0.03  * ( Th.sum( L2mr(U,1.5)**2 ) + 1.*sL1(U,0.01) )
#       -0.001*Th.sum( (U - Th.concatenate([U[:,1:],U[:,0:1]],axis=1)) ** 2. )

sigma  = 0.5
model  = sin_model(sigma)
#model  = quad_model(sigma,0.2,2.)
#model  = exp_model(sigma)
#model  = lin_model(sigma)
baye   = posterior(model,prior)

data, U, V1, bbar, Cb , STAB = LNLNP(T=10000,sigma=model.sigma,NL=model.NL,N=27)
Nsub, N     =  U.shape
NRGC, Nsub  =  V1.shape

UU = U.flatten()

M = 4*N
filters = zeros((M,N))
lls     = zeros(M)
for i in arange(M):
    params      = zeros(N)
    params[mod(i-1,N)] = 0.2*R.rand()
    params[mod(i  ,N)] = 0.8
    params[mod(i+1,N)] = 0.2*R.rand()
    for j in arange(3):
        params    = baye.MAP(params,[data])
    filters[i][:] = params
    lls[i] = baye.f(params,data)

p.figure(2)
for i in arange(minimum(N,9)):
    p.subplot(minimum(N,9)*100+10+i)
    if i>0:
        p.plot(arange(N),filters[i],'b')
    else:
        p.plot(arange(N),sum(abs(filters),axis=0),'r')
p.show()

print
print 'stimulus sigma  :  ',sigma
print 'true    ||subunit RF||^2  : ', sum(U*U,axis=1)
print 'optimal ||subunit RF||^2  : ', sum(filters*filters,axis=1)
