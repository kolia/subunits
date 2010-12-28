from numpy  import dot,sin,exp,std,eye,sum,add,all,newaxis
from numpy  import arange,transpose,fromfunction,reshape    ,amax
from numpy.linalg import inv
from theano import function
import theano.tensor  as Th
import numpy.random   as R
import scipy.linalg   as L
import scipy.optimize as Opt

def adaptive_Linear(W,X):
    """Return W*X normalized to have standard deviation 0.5."""
    WX = dot(W,X)
    return WX * 0.5 / std(WX)

def simulate_LNLNP(
    N             = 10               ,  # number of cones, subunits & RGCs
    Sigma         = 1.               ,  # subunit & RGC connection width
    firing_rate   = 0.1              ,  # RGC firing rate
    T             = 100000           ): # number of time samples
    """Simulate spiking data for an LNLNP model.
    Then use this data to calculate the STAs and STCs.
    """
	
    spatial  = fromfunction( lambda x: exp(-0.5 * ((x-N/2)/Sigma)**2) , (N,) )
    U        = L.circulant( spatial )   # connections btw cones & subunits
    V        = U                        # connections btw subunits & RGCs

    X        = R.randn(N,T)                 # cone activations
    b        = sin(adaptive_Linear(U,X))    # subunit activations
    Y        = exp(adaptive_Linear(V,b))    # RGC activations
    Y        = Y * N * T * firing_rate / sum(Y)

    spikes   = R.poisson(Y)
    N_spikes = sum(spikes,1)
	
    STA = [ sum(X*spikes[i,:],1) / N_spikes[i] for i in arange(N) ]

    STC = [ dot( X-STA[i][:,newaxis] , transpose((X-STA[i][:,newaxis])*Y[i,:])) \
                 / N_spikes[i] - eye(N)                for i in arange(N) ]

    return ((N_spikes,STA,STC),U)


def sin_model(U, STA, STC):
    """b(s)  =  sin( U s )"""
    STAB = Th.exp(-0.5*Th.sum(Th.dot(U,STC)*U,axis=1)) * Th.sin(Th.dot(U,STA))
    bbar = Th.zeros_like(STAB)	
    eU   = Th.exp( -0.5 * Th.sum( U * U , axis=1 ) )
    Cb   = 0.5 * (Th.sinh(0.5*Th.dot(U,U.T))*eU).T*eU
    return (STAB,bbar,Cb)


class posterior:
    def __init__(self,model,prior=1):
        self.memo_U    = None
        self.memo_Cbm1 = None
        self.prior     = prior
        
        U   = Th.dmatrix()                   # SYMBOLIC variables       #
        STA = Th.dvector()                                              #
        STC = Th.dmatrix()                                              #
        (STAB,bbar,Cb) = model(U, STA, STC)                             #
        Cbm1       = Th.dmatrix()                                       #
        Cbm1       = 0.5*(Cbm1+Cbm1.T)                                  #
        STABC      = Th.dot(Cbm1,(STAB-bbar))                           #
        posterior  = Th.sum(STABC*(STAB-bbar))                          #
        dposterior = 2*posterior - Th.sum( Th.outer(STABC,STABC) * Cb ) #
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

    def  f(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.posterior , U, (N_spikes,STA,STC))
    def df(self, U, N_spikes,STA,STC): return self.sum_RGCs( self.dposterior, U, (N_spikes,STA,STC))
    def MAP(self,data,x0):     return Opt.fmin_bfgs(self.f,x0.flatten(),self.df,data,full_output=True,disp=True)

data, U = simulate_LNLNP()
baye    = posterior(sin_model,prior=0)

Cbm1    = baye.Cbm1(U)
N_spikes,STA,STC=data
aaa = baye.posterior(U,STA[0],STC[0],Cbm1)
print aaa
bbb = baye.dposterior(U,STA[0],STC[0],Cbm1)
print bbb

nU      = baye.MAP(data,U)