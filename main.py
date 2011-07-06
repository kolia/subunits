
import LQuadLExP
reload(LQuadLExP)

from LQuadLExP import posterior, posterior_dUV1, posterior_dU, \
       posterior_dV2, posterior_dV1, posterior_dV2V1
from numpy  import add,reshape,concatenate,log,eye,outer,transpose,zeros
from numpy.linalg import inv, det, norm, eigvals, svd

import pylab as p

import simulate_data
reload(simulate_data)

import optimize
reload(optimize)

from numpy import dot, ones , arange, sum, array
import numpy.random   as R
import numpy.linalg   as L


V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = simulate_data.LNLNP(NL=NL,N=24)
Nsub, Ncones =  U.shape
NRGC, Nsub   = V1.shape
Nproj = Nsub
V2 = V2 * ones((Nproj,))


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

#baye   = posterior(N,Nsub,NRGC)
#data   = [(N_spikes , STA , STC)]
#index  = slice(3)
#true_params = concatenate(( U.flatten() , V2 , V1.flatten() ))

baye   = posterior_dUV1(Ncones,Nproj,NRGC)
data   = [ (V2, N_spikes , STA , STC) ]
index  = slice(3)
true_params = concatenate(( U.flatten() , V1.flatten() ))

#baye   = posterior_dV2V1(N,Nsub,NRGC)
#data   = [ (U, N_spikes , STA , STC) ]
#index  = slice(3)
#true_params = concatenate(( V2 , V1.flatten() ))


## #Check derivative
## print
## print 'Checking derivatives:'
## df = baye.df(true_params,data[0])
## UU = U.flatten()
## dh = 0.00000001
## ee = eye(len(true_params))*dh
## for i in arange(len(true_params)):
##    print (baye.f(true_params+ee[:,i],data[0])-baye.f(true_params,data[0]))/dh , df[i]


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
print 'N_cones    = ', Ncones
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print 'N_spikes   = ', N_spikes
print 'norm( Mk )  = '       , [ norm(eye(Ncones)-baye.M(U,V2[0:Nsub],V1[i,:])) \
                                        for i in range(NRGC)]

print
print 'true params :'
baye.callback(true_params,data)

A = concatenate([N*reshape(sta,(1,len(sta))) for N,sta in zip(N_spikes,STA)])
#B = concatenate([N*(stc + outer(sta,sta)) for N,sta,stc in zip(N_spikes,STA,STC)])
B = concatenate([N*stc for N,sta,stc in zip(N_spikes,STA,STC)])
C = concatenate((A,B))
YU,S,VI = svd(C)
T       = VI[0:Nproj,:]

## baye_proj = posterior_dUV1(Ncones,Nproj,NRGC)
## data_proj = data
## flat_svdU = T.flatten()
baye_proj = posterior_dUV1(Nproj,Nproj,NRGC)
STA       = [dot(T,sta) for sta in STA]
STC       = [dot(dot(T,stc),transpose(T)) for stc in STC]
data_proj = [ (V2*ones(Nproj), N_spikes , STA , STC) ]
flat_svdU  = eye(Nproj).flatten()

init_params_proj = 0.0001 * R.randn(flat_svdU.size+Nproj*NRGC)
##order = range(flat_svdU.size)
##R.shuffle(order)
##print 'Permutation: ' , order
##flat_svdU = flat_svdU[ order ]
init_params_proj = concatenate( [flat_svdU , init_params_proj[flat_svdU.size:]] )
print
print 'initial params :'
baye.callback(init_params_proj,data_proj)
print

params_proj = init_params_proj
params_proj = baye_proj.optimize(params_proj,data_proj[0])

def rexpand(x):
    return concatenate([dot(reshape(x[0:Nproj*Nproj],(Nproj,Nproj)),T).flatten(),x[Nproj*Nproj:]])
#    return concatenate([dot(transpose(T),reshape(x[0:Nproj*Nsub],(Nproj,Nsub))).flatten(),x[Nproj*Nsub:]])

#init_params = init_params_proj
#params      = params_proj
init_params = rexpand( init_params_proj )
params      = rexpand(      params_proj )

true_params = concatenate( [zeros(params.size-true_params.size) , true_params] )

p.figure(2)
baye.plot(params,true_params,data[0])

print 'log-likelihood of init params = ', baye.f(init_params,data[0])
print 'log-likelihood of opt  params = ', baye.f(params,data[0])
print 'log-likelihood of true params = ', baye.f(true_params,data[0])

#print 'init params:'
#print  baye.params(init_params,data[0])[index]
#print 'true params:'
#print  baye.params(true_params,data[0])[index]
#print 'opt  params:'
#print  baye.params(params ,data[0])[index]
#print
#print
#print 'true U*V1:' , dot(baye.params(true_params,data[0])[0].T,baye.params(true_params,data[0])[2].T)
#print 'opt U*V1:' , dot(baye.params(params,data[0])[0].T,baye.params(params,data[0])[2].T)
