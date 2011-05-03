"""
Bayesian Linear-Nonlinear-Linear-Exponential-Poisson model fitter.
@author: kolia
"""
from numpy import sum,add,all,reshape,iscomplex,log,exp,arange,minimum,sin
from numpy.linalg import inv,slogdet
from theano import function
import theano.tensor  as Th
#import scipy.optimize as Opt
from optimize import fmin_barrier_bfgs
import pylab as p

class quad_model:
    """b(s)  =  a * ( U s + c ) **2  for s with variance sigma**2."""
    def __init__(self,sigma,a,c):
        self.a     = a
        self.c     = c
        self.sigma = sigma
    def NL(self,x): return self.a*(x+self.c)**2
    def theano(self, U, STA, STC):
        STAB = self.a*(Th.sum(Th.dot(U,STC)*U,axis=1) + Th.dot(U,STA) **2 + self.c**2)
        bbar = self.a*(self.sigma**2* Th.sum( U*U , axis=1 ) + self.c**2)
        UU   = self.sigma**2* Th.dot(U,U.T)
        Cb   = self.a**2* (2.* UU**2 +  4.*self.c**2 * UU)
        regular  = 0.01* Th.sum(U*U)
        return (STAB,bbar,Cb,regular)

class sin_model:
    """b(s)  =  sin( U s )  for s with variance sigma**2."""
    def __init__(self,sigma): self.sigma = sigma
    def NL(self,x): return sin(x)
    def theano(self, U, STA, STC):
        STAB = Th.exp(-0.5*Th.sum(Th.dot(U,STC)*U,axis=1)) * Th.sin(Th.dot(U,STA))
        bbar = Th.zeros_like(STAB)
        eU   = Th.exp( -0.5 * self.sigma**2* Th.sum( U * U , axis=1 ) )
        Cb   = 0.5 * (Th.sinh(0.5* self.sigma**2* Th.dot(U,U.T))*eU).T*eU
        regular  = 0.0000001* Th.sum(U*U)
        return (STAB,bbar,Cb,regular)
        
class exp_model:
    """b(s)  =  exp( U s )  for s with variance sigma**2."""
    def __init__(self,sigma): self.sigma = sigma
    def NL(self,x): return exp(x)
    def theano(self, U, STA, STC):
        bbar = Th.exp(0.5* self.sigma**2* Th.sum( U * U , axis=1 ))
        STAB = Th.exp(0.5* Th.sum(Th.dot(U,STC)*U,axis=1) + Th.dot(U,STA))
        Cb   = (Th.exp(0.5* self.sigma**2* Th.dot(U,U.T))*bbar).T*bbar
        regular  = 0.01* Th.sum(U*U)
        return (STAB,bbar,Cb,regular)

class lin_model:
    """b(s)  =  U s  for s with variance sigma**2."""
    def __init__(self,sigma): self.sigma = sigma
    def NL(self,x): return x
    def theano(self, U, STA, STC):
        STAB = Th.dot( U , STA )
        bbar = Th.zeros_like(STAB)
        Cb   = self.sigma**2* Th.dot(U,U.T)
        regular  = 0 #0.0000000001*Th.sum(U*U)
        return (STAB,bbar,Cb,regular)


class posterior:
    def __init__(self,model,prior=1):
        self.memo_U    = None
        self.memo_Cbm1 = None
        self.prior     = prior
        self.mindet    = 1.e-60
        
        U   = Th.dmatrix()                   # SYMBOLIC variables       #
        STA = Th.dvector()                                              #
        STC = Th.dmatrix()                                              #
        (STAB,bbar,Cb,regular) = model.theano(U, STA, STC)              #
        Cbm1       = Th.dmatrix()                                       #
        STABC      = Th.dot(Cbm1,(STAB-bbar))                           #
        posterior  = Th.sum(STABC*(STAB-bbar)) - regular                #
        dposterior = 2*posterior - Th.sum( Th.outer(STABC,STABC) * Cb ) - regular #
        dposterior = Th.grad( cost              = dposterior,           #
                              wrt               = U         ,           #
                              consider_constant = [STABC]   )           #

        self.Cb         = function( [U]             ,  Cb       )
        # log-posterior term  (STAB - bbar).T inv(Cb) (STAB - bbar)
        self.posterior  = function( [U,STA,STC,Cbm1],  posterior)
        # gradient of log-posterior term
        self.dposterior = function( [U,STA,STC,Cbm1], dposterior)

    def barrier(self,U, (N_spikes,STA,STC)):
        U = reshape(U,(-1,len(STA[0])))
        s,ld = slogdet(self.Cb(U))
        if iscomplex(s) or s<1 or (ld < log(self.mindet)):
#            print ' Barrier True  ;  slogdet=', s, exp(ld)
            return True
#        print ' Barrier False ;  slogdet=', s, exp(ld)
        return False

    def Cbm1(self,U):
        """Memoize Cbm1 = inv(Cb) for efficiency."""
        if not all(self.memo_U == U):
            self.memo_U = U
#            print self.Cb(U)
            Cbm1 = inv(self.Cb(U))
            self.memo_Cbm1 = 0.5 * (Cbm1 + Cbm1.T)
#            print amax(self.memo_Cbm1-self.memo_Cbm1.T)
        return self.memo_Cbm1

    def sum_RGCs(self, f, U, (N_spikes,STA,STC)):
        U = reshape(U,(-1,len(STA[0])))
        Cbm1 = self.Cbm1(U)
        return reduce( add ,  [ n**2/(n+self.prior) * f(U,sta,stc,Cbm1).flatten() \
                               for n,sta,stc in zip(N_spikes,STA,STC)])

    def callback(self,U, (N_spikes,STA,STC)):
        U = reshape(U,(-1,len(STA[0])))
        s,ld = slogdet(self.Cb(U))
        print '||U||^2 : ', sum( U*U ), 'slogdet : ', s, exp(ld)

    def plot(self,UU,U):
        (N,n) = U.shape
        UU = reshape(UU,(N,n))  
        for i in arange(minimum(N,9)):
            p.subplot(minimum(N,9)*100+10+i)
            UUi = UU[i,]
            m = min(UUi)
            M = max(UUi)
            if -m>M:
                UUi = -UUi
            if -m>M:
                UUi = UUi * max(U[i,]) / m
            else:
                UUi = UUi * max(U[i,]) / M
            p.plot(arange(n),UUi,'b',arange(n),U[i,],'rs')
            p.show()

    def  f(self, U, data): return -self.sum_RGCs( self.posterior , U, data)
    def df(self, U, data): return -self.sum_RGCs( self.dposterior, U, data)
    def MAP(self,x0,data): 
        def cb(para): return self.callback(para,data[0])
        return fmin_barrier_bfgs(self.f,x0.flatten(),fprime=self.df,
                                 gtol=1.1e-6,maxiter=5000,args=data,
                                 callback=cb,barrier=self.barrier)