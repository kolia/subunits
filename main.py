
import LQuadLExP
reload(LQuadLExP)

from LQuadLExP import posterior, posterior_dUV1, posterior_dU, \
       posterior_dV2, posterior_dV1, posterior_dV2V1, regularize_norm_smooth
from numpy  import add,reshape,concatenate,log,eye,isnan,ones_like,Inf,max,min
from numpy.linalg import inv, det, norm, eigvals

import simulate_data
reload(simulate_data)

import optimize
reload(optimize)

from numpy import dot, ones , arange, sum, array
import numpy.random   as R
import numpy.linalg   as L

# [ norm(U)-1  ,   smooth(U)  ,  L2(U)  ,  L2(V)  ]
#lam = array([0.1,10.,0.01,0.000001])
lam = array([0.00001,0.001,0.01,0.000001])
#lam = array([1.,0.,0.00001])


V2 = 0.3
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
(N_spikes,STA,STC), U, V1 = simulate_data.LNLNP(NL=NL,N=24)
Nsub, N    =  U.shape
NRGC, Nsub = V1.shape
V2 = V2 * ones((Nsub,))


#baye   = posterior_dU(N,Nsub,NRGC)
#data   = [ (V2, V1, N_spikes , STA , STC) ]
#index  = 0
#true_params = U.flatten()

#baye   = posterior_dV2(N,Nsub,NRGC)
#data   = [ ( U, V1, N_spikes , STA , STC ) ]
#index  = 1
#true_params = V2

#baye   = posterior_dV1(N,Nsub,NRGC)
#data   = [ ( U, V2, N_spikes , STA , STC) ]
#index  = 2
#true_params = V1.flatten()

baye   = posterior(N,Nsub,NRGC,regularize_norm_smooth)
data   = [ (lam, N_spikes , STA , STC) ]
index  = slice(3)
true_params = concatenate(( U.flatten() , V2 , V1.flatten() ))

#baye   = posterior_dUV1(N,Nsub,NRGC,regularize_norm_smooth)
#data   = [ (V2, lam, N_spikes , STA , STC) ]
#index  = slice(3)
#true_params = concatenate(( U.flatten() , V1.flatten() ))

#baye   = posterior_dV2V1(N,Nsub,NRGC)
#data   = [ (U, N_spikes , STA , STC) ]
#index  = slice(3)
#true_params = concatenate(( V2 , V1.flatten() ))


## Check derivative
#print
#print 'Checking derivatives:'
#df = baye.df(true_params,data[0])
#UU = U.flatten()
#dh = 0.00000001
#ee = eye(len(true_params))*dh
#for i in arange(len(true_params)):
#    print (baye.f(true_params+ee[:,i],data[0])-baye.f(true_params,data[0]))/dh , df[i]


print
print 'U  : ', U
print '||U||^2  : ', sum(U*U)
print
print 'V1 : ' , V1
print
print baye.callback(true_params,data)
print
print 'U.shape    = ', U.shape
print 'V2.shape   = ', V2.shape
print 'V1.shape   = ', V1.shape
print 'N_cones    = ', N
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print 'N_spikes   = ', N_spikes
print 'norm( Mk )  = '       , [ norm(eye(N)-baye.M(U,V2,V1[i,:])) \
                                        for i in range(NRGC)]

print
print 'true params :'
baye.callback(true_params,data)
#params = true_params * ( 1  + 1. * (R.randn(len(true_params))-0.5))
init_params = 0.0001 * R.randn(len(true_params))
print
print 'initial params :'
baye.callback(init_params,data)
print

params = init_params
for i in arange(20):
#    (V2, lam, N_spikes , STA , STC) = data[0]
#    data   = [ (V2, lam*(0.1*i+0.1), N_spikes , STA , STC) ]
    params = baye.MAP(params,data)

#params = baye.MAP(init_params,data)


print 'log-likelihood of init params = ', baye.f(init_params,data[0])
print 'log-likelihood of opt  params = ', baye.f(params,data[0])
print 'log-likelihood of true params = ', baye.f(true_params,data[0])

print 'init params:'
print  baye.params(init_params,data[0])[index]
print 'true params:'
print  baye.params(true_params,data[0])[index]
print 'opt  params:'
print  baye.params(params ,data[0])[index]
print
print
print 'true U*V1:' , dot(baye.params(true_params,data[0])[0].T,baye.params(true_params,data[0])[2].T)
print 'opt U*V1:' , dot(baye.params(params,data[0])[0].T,baye.params(params,data[0])[2].T)

baye.plot(params,true_params,data[0])