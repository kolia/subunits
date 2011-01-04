from numpy  import dot,sin,exp,std,eye,sum,add,all,newaxis,  array,sqrt
from numpy  import arange,transpose,fromfunction,reshape
from numpy.linalg import inv
from theano import function
import theano.tensor  as Th
import numpy.random   as R
import scipy.linalg   as L
import scipy.optimize as Opt

def adaptive_Linear(W,X):
    """Return W*X normalized to have standard deviation 1."""
    WX = dot(W,X)
    return WX / std(WX)

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

    U = U[0:10:2,:]
    V = V[0:10:2,0:10:2]
    NRGC = V.shape[0]

    X        = R.randn(N,T)                 # cone activations
    b        = sin(adaptive_Linear(U,X))    # subunit activations
    Y        = exp(adaptive_Linear(V,b))    # RGC activations
    Y        = Y * NRGC * T * firing_rate / sum(Y)

    spikes   = R.poisson(Y)
    N_spikes = sum(spikes,1)
	
    STA = [ sum(X*spikes[i,:],1) / N_spikes[i] for i in arange(NRGC) ]

    STC = [ dot( X-STA[i][:,newaxis] , transpose((X-STA[i][:,newaxis])*Y[i,:])) \
                 / N_spikes[i] - eye(N)                for i in arange(NRGC) ]

    return ((N_spikes,STA,STC),U)


def sin_model(U, STA, STC):
    """b(s)  =  sin( U s )"""
    STAB = Th.exp(-0.5*Th.sum(Th.dot(U,STC)*U,axis=1)) * Th.sin(Th.dot(U,STA))
    bbar = Th.zeros_like(STAB)	
    eU   = Th.exp( -0.5 * Th.sum( U * U , axis=1 ) )
    Cb   = 0.5 * (Th.sinh(Th.dot(U,U.T))*eU).T*eU
    regular  = 0.1*Th.sum( Th.cosh(Th.sum(U*U,axis=1)) )
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
#        Cbm1       = 0.5*(Cbm1+Cbm1.T)                                  #
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
        U = reshape(U,(-1,len(STA[0])))
        print sum( U*U, axis=1)

    def  f(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.posterior , U, (N_spikes,STA,STC))
    def df(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.dposterior, U, (N_spikes,STA,STC))
    def MAP(self,data,x0): return Opt.fmin_ncg(self.f,x0.flatten(),fprime=self.df,avextol=1.1e-3,args=data,callback=self.callback)

data, U = simulate_LNLNP()
baye    = posterior(sin_model,prior=1.)

N_spikes,STA,STC=data
print baye.f(U,N_spikes,STA,STC)

## Check derivative
#df = baye.df(U,N_spikes,STA,STC)
#UU = U.flatten()
#dh = 0.00000001
#ee = eye(100)*dh
#for i in arange(100):
#    print (baye.f(UU+ee[:,i],N_spikes,STA,STC)-baye.f(UU,N_spikes,STA,STC))/dh , df[i]

#x0  = U
x0  = array( STA ).T

x  = baye.MAP(data,x0.flatten())
x  = reshape( x , (-1,10) )


#K  = [10,8,6]
#x  = [R.randn(5,10) for i in K+[1]]
#
#x  = eye(10)
#x  = x[1:10:2,:] * 0.005
#
#x  = baye.MAP(data,x.flatten())
#x  = reshape( x , (-1,10) )

#for i,k in enumerate(K):
#    x[i]  = exp( k*x[max(0,i-1)] )
#    x[i]  = reshape( x[i] , (-1,10) )
#    x[i]  = x[i] * 0.01 / sqrt(sum(x[i]*x[i],axis=1))[:,newaxis]
#    print sum(x[i]*x[i],axis=1)
#    x[i]  = baye.MAP(data,x[i].flatten())
#    x[i]  = reshape( x[i] , (-1,10) )
