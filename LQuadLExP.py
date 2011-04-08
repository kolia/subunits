"""
Linear-Quadratic-Linear-Exponential-Poisson model fitter.
@author: kolia
"""
from numpy  import add, reshape, concatenate, eye, isnan, ones_like, \
                   Inf, arange, max, min
from numpy.linalg import inv, det
from theano import function
import theano.tensor  as Th
from optimize import fmin_barrier_bfgs

import pylab as p

class posterior:
    def __init__(self,N,Nsub,NRGC,prior=1):
        self.N     = N
        self.Nsub  = Nsub
        self.NRGC  = NRGC
        self.mindet= 0.2
        U    = Th.dmatrix()                   # SYMBOLIC variables     #
        V1   = Th.dvector()                                            #
        V2   = Th.dvector()                                            #
        STA  = Th.dvector()                                            #
        STC  = Th.dmatrix()                                            #
        theta= Th.dot( U.T , V1 )                                      #
        M    = Th.dot( V1 * U.T , (V2 * U.T).T )                       #
        detM = Th.dscalar()
        invM = Th.dmatrix()
        invMtheta = Th.as_tensor_variable(Th.dot(invM,theta),ndim=2)
        
        post = (  Th.log(detM) \
#                - 0.01 / (detM-self.mindet) \
                - Th.sum(invMtheta*theta) \
                + 2. * Th.sum( theta * STA ) \
                + Th.sum( M * (STC + Th.outer(STA,STA)) )) / 2. \
                - 1. * Th.sum( (Th.sum( U*U , axis=1 ) - ones_like(V1)) ** 2. )
        dpost_dM  = ( invM + invMtheta * invMtheta.T \
#                    + 0.01 * detM * invM / ((detM-self.mindet)**2) \
                    ) / 2.

        def dpost1(dX):
            return Th.grad( cost = post                   , wrt = dX , \
                            consider_constant=[invM,detM,STC,STA] )

        def dpost2(dX):
            return - Th.grad( cost = Th.sum( dpost_dM * M ) , wrt = dX , 
                            consider_constant=[dpost_dM,STA,STC,invM,invMtheta])

        def dpost(dX):
            return Th.grad( cost = post                   , wrt = dX , \
                            consider_constant=[invM,detM,STC,STA] ) \
                 - Th.grad( cost = Th.sum( dpost_dM * M ) , wrt = dX , 
                            consider_constant=[dpost_dM,STA,STC,invM,invMtheta])

        self.M          = function( [U,V2,V1]                  ,  M       ) #
        self.posterior  = function( [U,V2,V1,invM,detM,STA,STC],  post    ) #
        self.dpost_dU   = function( [U,V2,V1,invM,detM,STA,STC], dpost(U) ) #
        self.dpost_dV1  = function( [U,V2,V1,invM,detM,STA,STC], dpost(V1)) #
        self.dpost_dV2  = function( [U,V2,V1,invM,detM,STA,STC], dpost(V2)) #

        self.dpost_dM   = function( [U,V2,V1,invM,detM,STA,STC], dpost_dM) #        


#    def barrier(self,params,data):
#        (U,V2,V1) = self.params(params,data)
#        IM = eye(self.N)-self.M(U,V2,V1[i,:])
#        return det(IM) < self.mindet

    def barrier(self,params,data):
        (U,V2,V1) = self.params(params,data)
        for i in arange(V1.shape[0]):
            IM = eye(self.N)-self.M(U,V2,V1[i,:])
            if det(IM) < self.mindet:
                return True
        return False

    def callback(self,params,data):
        return
        
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
        result = None
        for i,(n,sta,stc) in enumerate(zip(N_spikes,STA,STC)):
            IM = eye(self.N)-self.M(U,V2,V1[i,:])
            term = n * g(U,V2,V1[i,:], inv(IM), det(IM), sta, stc)
            if any(isnan(term.flatten())):
                print 'oups'
                term = None
#                raise ArithmeticError('nan')
            if result is not None and term is not None:
                result = op( result , term )
            else:
                if term is not None:
                    result = term
        if result is None:
            return Inf
        else:
            return -result


    def     f (self, params, data):
        return self.sum_RGC( add, self.posterior, 
                              self.params(params,data),
                              self.data  (params,data))

    def df_dU (self, params, data):
        return self.sum_RGC( add, self.dpost_dU ,
                              self.params(params,data),
                              self.data  (params,data))

    def df_dV1(self, params, data):
        def concat(a,b): return concatenate((a,b))
        return self.sum_RGC( concat, self.dpost_dV1,
                              self.params(params,data),
                              self.data  (params,data))

    def df_dV2(self, params, data):
        return self.sum_RGC( add, self.dpost_dV2,
                              self.params(params,data),
                              self.data  (params,data))

    def df(self, params, data):
        return concatenate((self.df_dU(params,data).flatten(),
                            self.df_dV2(params,data),
                            self.df_dV1(params,data)))

    def MAP(self,params,data):
        def cb(para): return self.callback(para,data)
#        return Opt.fmin_ncg(self.f,params,fprime=self.df,avextol=1.1e-5,
#                            maxiter=10000,args=data,
#                            callback=cb)
        return fmin_barrier_bfgs(self.f,params,fprime=self.df,
                                 gtol=1.1e-2,maxiter=10000,args=data,
                                 callback=cb,barrier=self.barrier)


    def plot(self,params,true_params,data):
        (cU,cV2,cV1) = self.params(params,data)
        (tU,tV2,tV1) = self.params(true_params,data)
        (N,n) = cU.shape
        p.figure(1)
        for i in arange(N):
            p.subplot(N*100+10+i)
            cUi = cU[i,]
            m = min(cUi)
            M = max(cUi)
            if -m>M:
                cUi = cUi * max(tU[i,]) / m
            else:
                cUi = cUi * max(tU[i,]) / M
            p.plot(arange(n),cUi,'b',arange(n),tU[i,],'rs')
            p.show()
        

#    def Cbm1(self,U):
#        """Memoize Cbm1 = inv(Cb) for efficiency."""
#        if not all(self.memo_U == U):
#            self.memo_U = U
##            print self.Cb(U)
#            Cbm1 = inv(self.Cb(U))
#            self.memo_Cbm1 = 0.5 * (Cbm1 + Cbm1.T)
##            print amax(self.memo_Cbm1-self.memo_Cbm1.T)
#        return self.memo_Cbm1



class posterior_single(posterior):
    def data(self,params,data): return data[-3:]


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


class posterior_dUV1(posterior_single):
    '''Optimization wrt U and V1 only'''
    def params(self,params,(V2,N_spikes,STA,STC)):
        return (self.U(params[0:self.N*self.Nsub]),V2,
                self.V1(params[self.N*self.Nsub:]))

    def df(self,params,data):
        return concatenate((self.df_dU(params,data).flatten(),
                            self.df_dV1(params,data)))


class posterior_dV2V1(posterior_single):
    '''Optimization wrt U and V1 only'''
    def params(self,params,(U,N_spikes,STA,STC)):
        return (U,params[0:self.Nsub],
                self.V1(params[self.Nsub:]))

    def df(self,params,data):
        return concatenate((self.df_dV2(params,data),
                            self.df_dV1(params,data)))