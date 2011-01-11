"""
Linear-Quadratic-Linear-Exponential-Poisson model fitter.
@author: kolia
"""
from numpy  import sum,add,sqrt,dot,ones,reshape,concatenate
from theano import function
import theano.tensor  as Th
import scipy.optimize as Opt

class posterior:
    '''Joint optimization in U, V2 and V1'''
    def __init__(self,N,Nsub,NRGC,prior=1):
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
        posterior  = -0.5 * Th.sum( V2 * U.T*U.T ) \
                     -0.5 * Th.sum( UV1U * UV1U *V2 ) \
                     -0.5 * Th.sum( theta * theta ) \
                     + Th.dot( theta.T , STA ) \
                     + Th.sum( Th.dot( V2*U.T , U ) \
                     * (STC + STA.T*STA) )
        dpost_dU  = Th.grad( cost           = posterior ,               #
                             wrt            = U         )               #
        dpost_dV1 = Th.grad( cost           = posterior ,               #
                             wrt            = V1        )               #
        dpost_dV2 = Th.grad( cost           = posterior ,               #
                             wrt            = V2        )               #
#        self.posterior  = function( [U,V2,V1,STA,STC],  x1)      #
        self.posterior  = function( [U,V2,V1,STA,STC],  posterior)      #
        self.dpost_dU   = function( [U,V2,V1,STA,STC], dpost_dU  )      #
        self.dpost_dV1  = function( [U,V2,V1,STA,STC], dpost_dV1 )      #
        self.dpost_dV2  = function( [U,V2,V1,STA,STC], dpost_dV2 )      #


    def callback(self,params): return
#        print sqrt( sum( params ** 2 ) )
        
    def U(self,params):
        return reshape( params , (self.Nsub,self.N) )

    def V1(self,params):
        return reshape( params , (self.NRGC,self.Nsub) )

    def params(self,params,data):
        return (self.U(params[0:self.N*self.Nsub]),
                params[self.N*self.Nsub:self.N*self.Nsub+self.Nsub],
                self.V1(params[self.N*self.Nsub+self.Nsub:]))

    def data(self,params,data): return data

    def sum_RGC(self,op,g,(U,V2,V1),(N_spikes,STA,STC)):
        return reduce( op ,  [ n * g(U,V2,V1[i,:],sta,stc) \
                                for i,(n,sta,stc) in \
                                enumerate(zip(N_spikes,STA,STC))])

#    def sum_RGC(self,op,g,params,data):
#        return reduce( op ,  [ n * g(U,V2,V1[i,:],sta,stc) \
#                                for i,(n,sta,stc) in \
#                                enumerate(zip(N_spikes,STA,STC))])

    def     f (self, params, data):
        return -self.sum_RGC( add, self.posterior, 
                              self.params(params,data),
                              self.data  (params,data))

    def df_dU (self, params, data):
        return -self.sum_RGC( add, self.dpost_dU ,
                              self.params(params,data),
                              self.data  (params,data))

    def df_dV1(self, params, data):
        def concat(a,b): return concatenate((a,b))
        return -self.sum_RGC( concat, self.dpost_dV1,
                              self.params(params,data),
                              self.data  (params,data))

    def df_dV2(self, params, data):
        return -self.sum_RGC( add, self.dpost_dV2,
                              self.params(params,data),
                              self.data  (params,data))

    def df(self, params, data):
        return concatenate((self.df_dU(params,data).flatten(),
                            self.df_dV2(params,data),
                            self.df_dV1(params,data)))

    def MAP(self,params,data):
        return Opt.fmin_ncg(self.f,params,fprime=self.df,avextol=1.1e-5,
                            maxiter=100,args=data,callback=self.callback)


class posterior_single(posterior):
    def data(self,params,data):
        X,Y,N_spikes,STA,STC = data
        return (N_spikes,STA,STC)


class posterior_dU(posterior_single):
    '''Optimization wrt U only'''
    def params(self,params,data):
        V2,V1,N_spikes,STA,STC = data
        return ( self.U(params), V2, V1)
        
    def df(self,params,data):
        return self.df_dU(params,data).flatten()

                    
class posterior_dV2(posterior_single):
    '''Optimization wrt V2 only'''
    def params(self,params,(U,V1,N_spikes,STA,STC)):
        return ( U, params, V1)

    def df(self,params,data):
        return self.df_dV2(params,data)
    


class posterior_dV1(posterior_single):
    '''Optimization wrt V1 only'''
    def params(self,params,(U,V2,N_spikes,STA,STC)):
        return ( U, V2, self.V1(params))

    def df(self,params,data):
        return self.df_dV1(params,data)


from simulate_data import LNLNP
import numpy.random   as R
import numpy.linalg   as L

V2 = 0.2
def NL(x): return x + V2 * x ** 2
(N_spikes,STA,STC), U, V1 = LNLNP(NL=NL)
Nsub, N    =  U.shape
NRGC, Nsub = V1.shape
V2 = V2 * ones((Nsub,))

print 'U.shape    = ', U.shape
print 'V1.shape   = ', V1.shape
print 'N_cones    = ', N
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print 'N_spikes   = ', N_spikes
print '|| U.T V2 U || = ', L.norm(dot(U.T*V2,U))

#baye   = posterior_dU(N,Nsub,NRGC)
#data   = [ (V2, V1, N_spikes , STA , STC) ]
#true_params = U.flatten()

#baye   = posterior_dV2(N,Nsub,NRGC)
#data   = [ ( U, V1, N_spikes , STA , STC ) ]
#param_shape = lambda X: reshape( X, V2.shape)
#true_params = V2

baye   = posterior_dV1(N,Nsub,NRGC)
data   = [ ( U, V2, N_spikes , STA , STC) ]
true_params = V1.flatten()

#baye   = posterior(N,Nsub,NRGC)
#data   = [ (N_spikes , STA , STC) ]
#true_params = concatenate(( U.flatten() , V2 , V1.flatten() ))

print 'log-likelihood of true params = ', baye.f(true_params,data[0])
params = true_params + 0.1 * R.randn(len(true_params))
opt_params = baye.MAP(params,data)

print 'true params = ', baye.params(true_params,data[0])[2]
print 'opt  params = ', baye.params(opt_params ,data[0])[2]