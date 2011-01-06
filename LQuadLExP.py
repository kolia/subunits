"""
Linear-Quadratic-Linear-Exponential-Poisson model fitter.
@author: kolia
"""
from numpy  import sum,add,sqrt,ones,reshape,concatenate
from theano import function
import theano.tensor  as Th
import numpy.random   as R
import scipy.optimize as Opt

class posterior:
    '''Joint optimization in U, V2 and V1'''
    def _init_(self,N,Nsub,NRGC,prior=1):
        self.N     = N
        self.Nsub  = Nsub
        self.NRGC  = NRGC
        U   = Th.dmatrix()                   # SYMBOLIC variables       #
        V1  = Th.dvector()                                              #
        V2  = Th.dvector()                                              #
        STA = Th.dvector()                                              #
        STC = Th.dmatrix()                                              #
        theta = Th.dot( U.T , V1 )                                      #
        UV1U  = Th.dot( U , theta )                                     #
        posterior  = -0.5 * Th.sum( V1*V2 * U*U ) \
                     -0.5 * Th.sum( UV1U * UV1U * V1*V2 ) \
                     + Th.dot( theta.T , STA ) \
                     + Th.sum( Th.dot( U.T , V1*V2*U ) * (STC + STA.T*STA) )
        dpost_dU  = Th.grad( cost           = posterior ,               #
                             wrt            = U         )               #
        dpost_dV1 = Th.grad( cost           = posterior ,               #
                             wrt            = V1        )               #
        dpost_dV2 = Th.grad( cost           = posterior ,               #
                             wrt            = V2        )               #
        self.posterior  = function( [U,V2,V1,STA,STC],  posterior)      #
        self.dpost_dU   = function( [U,V2,V1,STA,STC], dpost_dU  )      #
        self.dpost_dV1  = function( [U,V2,V1,STA,STC], dpost_dV1 )      #
        self.dpost_dV2  = function( [U,V2,V1,STA,STC], dpost_dV2 )      #


    def callback(self,params):
        print sqrt( sum( params ** 2 ) )
        
    def U(self,params):
        return reshape( params , (self.Nsub,self.N) )

    def V1(self,params):
        return reshape( params , (self.NRGC,self.Nsub) )

    def params(self,params,data):
        return (self.U(params[0:self.N*self.Nsub]),
                params[self.N*self.Nsub:self.N*self.Nsub+self.Nsub],
                self.V1(params[self.N*self.Nsub+self.Nsub:]))

    def data(self,params,data): return data

    def sum_RGC(self,op,f,(U,V2,V1),(N_spikes,STA,STC)):
        return reduce( op ,  [ n * f(U,V2,V1[i,:],sta,stc) \
                                for i,(n,sta,stc) in enumerate(zip(N_spikes,STA,STC))])

    def     f (self, params, data):
        return -self.sum_RGC( add, self.posterior, self.params(params,data), self.data(params,data))

    def df_dU (self, params, data):
        return -self.sum_RGC( add, self.dpost_dU , self.params(params,data), self.data(params,data))

    def df_dV1(self, params, data):
        def concat(a,b): return concatenate((a,b))
        return -self.sum_RGC( concat, self.dpost_dV1, self.params(params,data), self.data(params,data))

    def df_dV2(self, params, data):
        return -self.sum_RGC( add, self.dpost_dV2, self.params(params,data), self.data(params,data))

    def df(self, params, data):
        return concatenate((self.df_dU(params,data),
                            self.df_dV2(params,data),
                            self.df_dV1(params,data)))

    def MAP(self,data,x0):
        return Opt.fmin_ncg(self.f,x0,fprime=self.df,avextol=1.1e-3,
                            args=data,callback=self.callback)

class posterior_single(posterior):
    def data(self,params,(X,Y,N_spikes,STA,STC)):
        return (N_spikes,STA,STC)    

class posterior_dU(posterior_single):
    '''Optimization wrt U only'''
    def params(self,params,(V2,V1,N_spikes,STA,STC)):
        return ( self.U(params), V2, V1)
                    
class posterior_dV2(posterior_single):
    '''Optimization wrt V2 only'''
    def params(self,params,(U,V1,N_spikes,STA,STC)):
        return ( U, params, V1)

class posterior_quad_dV1(posterior_single):
    '''Optimization wrt V1 only'''
    def params(self,params,(U,V2,N_spikes,STA,STC)):
        return ( U, V2, self.V1(params))

from simulate_data import LNLNP

data, U, V = LNLNP()
Nsub, N    = U.shape
NRGC, Nsub = V.shape
baye       = posterior(N,Nsub,NRGC)

V2 = ones((Nsub,))

N_spikes,STA,STC=data
print baye.f(U,V2,V.T,N_spikes,STA,STC)

## Check derivative
#df = baye.df(U,N_spikes,STA,STC)
#UU = U.flatten()
#dh = 0.00000001
#ee = eye(100)*dh
#for i in arange(100):
#    print (baye.f(UU+ee[:,i],N_spikes,STA,STC)-baye.f(UU,N_spikes,STA,STC))/dh , df[i]

x0  = U + R.randn(5,10)
#x0  = array( STA ).T

x  = baye.MAP(data,x0.flatten())
x  = reshape( x , (-1,10) )
