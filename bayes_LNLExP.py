"""
Bayesian Linear-Nonlinear-Linear-Exponential-Poisson model fitter.
@author: kolia
"""
from numpy  import sum,add,all,reshape
from numpy.linalg import inv
from theano import function
import theano.tensor  as Th
import scipy.optimize as Opt


def sin_model(U, STA, STC):
    """b(s)  =  sin( U s )"""
    STAB = Th.exp(-0.5*Th.sum(Th.dot(U,STC)*U,axis=1)) * Th.sin(Th.dot(U,STA))
    bbar = Th.zeros_like(STAB)
    eU   = Th.exp( -0.5 * Th.sum( U * U , axis=1 ) )
    Cb   = 0.5 * (Th.sinh(Th.dot(U,U.T))*eU).T*eU
    regular  = 0.0000000001*Th.sum( Th.cosh(Th.sum(U*U,axis=1)) )
    return (STAB,bbar,Cb,regular)


def exp_model(U, STA, STC):
    """b(s)  =  exp( U s )"""
    bbar = Th.exp(0.5* Th.sum( U * U , axis=1 ))
    STAB = Th.exp(0.5* Th.sum(Th.dot(U,STC)*U,axis=1) + Th.dot(U,STA))
    Cb   = (Th.exp(0.5* Th.dot(U,U.T))*bbar).T*bbar
    regular  = 0.0000000001*Th.sum( Th.cosh(Th.sum(U*U,axis=1)) )
    return (STAB,bbar,Cb,regular)


def lin_model(U, STA, STC):
    """b(s)  =  U s"""
    STAB = Th.dot( U , STA )
    bbar = Th.zeros_like(STAB)
    Cb   = Th.dot(U,U.T)
    regular  = 0 #0.0000000001*Th.sum(U*U)
    return (STAB,bbar,Cb,regular)


class posterior:
    def __init__(self,model,prior=1):
        self.memo_U    = None
        self.memo_Cbm1 = None
        self.prior     = prior
        
        U   = Th.dmatrix()                   # SYMBOLIC variables       #
        STA = Th.dvector()                                              #
        STC = Th.dmatrix()                                              #
        (STAB,bbar,Cb,regular) = model(U, STA, STC)                     #
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

    def callback(self,U):
        print sum( U*U, axis=1)

    def  f(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.posterior , U, (N_spikes,STA,STC))
    def df(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.dposterior, U, (N_spikes,STA,STC))
    def MAP(self,data,x0): return Opt.fmin_ncg(self.f,x0.flatten(),fprime=self.df,avextol=1.1e-3,args=data,callback=self.callback)

from simulate_data import LNLNP

data, U, V = LNLNP()
baye    = posterior(lin_model,prior=1.)
#baye    = posterior_b(exp_model,prior=1.)

N_spikes,STA,STC=data
print baye.f(U,N_spikes,STA,STC)

## Check derivative
#df = baye.df(U,N_spikes,STA,STC)
#UU = U.flatten()
#dh = 0.00000001
#ee = eye(100)*dh
#for i in arange(100):
#    print (baye.f(UU+ee[:,i],N_spikes,STA,STC)-baye.f(UU,N_spikes,STA,STC))/dh , df[i]

from numpy.random import randn
x0  = U + randn(5,10)
#x0  = array( STA ).T

x  = baye.MAP(data,x0.flatten())
x  = reshape( x , (-1,10) )

