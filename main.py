
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
import numpy.random   as Rand
import numpy.linalg   as L

#execfile('main_group_LASSO.py')
import main_group_LASSO
reload(main_group_LASSO)
from main_group_LASSO import R, V, V2, iW, filters

print 'V = ',V

Ncones   = R['N_cells'][0]
Nsub     = R['N_cells'][1]
NRGC     = R['N_cells'][2]
N_spikes = R['N_spikes']
STA      = R['statistics']['stimulus']['STA']
STC      = R['statistics']['stimulus']['STC']
U        = R['U']
V1       = R['V']
nonzero  = 1e-1

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

data   = [ V2, N_spikes , STA , STC ]
baye   = posterior_dUV1(data,Nsub)
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
print baye.callback(true_params)
print
print 'U.shape    = ', U.shape
print 'V1.shape   = ', V1.shape
print 'N_cones    = ', Ncones
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print 'N_spikes   = ', N_spikes
print 'norm( Mk )  = '       , [ norm(eye(Ncones)-baye.M(U,V2*ones(Nsub),V1[i,:])) \
                                        for i in range(NRGC)]

print
print 'true params :'
baye.callback(true_params)

Nproj    = sum(iW>nonzero)
data     = [ V2*ones(Nproj), N_spikes , STA , STC ]
baye_ARD = posterior_dUV1(data,Nproj)

init_params = concatenate([ filters[iW>nonzero,:].flatten() , V[iW>nonzero,:].T.flatten() ])
print
print 'initial params :'
baye.callback(init_params)
print

params = baye_ARD.optimize(init_params,maxiter=1)

p.figure(2)
baye.plot(baye_ARD.params(params),baye.params(true_params))

#print 'log-likelihood of init params = ', baye.f(init_params,data[0])
#print 'log-likelihood of opt  params = ', baye.f(params,data[0])
#print 'log-likelihood of true params = ', baye.f(true_params,data[0])
#
##print 'init params:'
##print  baye.params(init_params,data[0])[index]
##print 'true params:'
##print  baye.params(true_params,data[0])[index]
##print 'opt  params:'
##print  baye.params(params ,data[0])[index]
##print
##print
##print 'true U*V1:' , dot(baye.params(true_params,data[0])[0].T,baye.params(true_params,data[0])[2].T)
##print 'opt U*V1:' , dot(baye.params(params,data[0])[0].T,baye.params(params,data[0])[2].T)
