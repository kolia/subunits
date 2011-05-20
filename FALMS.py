
from numpy import sqrt

#import theano as Th
#        self.f       = f
#        self.fx      = Th.function( x     , f )
#        self.f_grad  = Th.grad( cost = f  , wrt = x )
#        self.fx_grad = Th.function( self.f_grad, x )

def add_quadratic_term( objective ):
    def optimize( self , X, Y, L, mu):
        return X
    objective.optimize = optimize
    return objective

def initialize_FALM(X): return (X, X , X*0 , X , 1 , 1 , 0 )

def FALM_step( F , G , rho , mu , prob ):
    
    # Unpack loop variables
    X,Y,Ym,Z,t,tm,skipped = prob

    # minimize f(X) - <G.fx_grad(Z),X-Z> + ||X-Z||**2/(2*mu) in X for fixed Z
    X , fX , objX = F.optimize( X, Z, -G.objective.df(Z) , 1/(2*mu) )
    
    # to skip or not to skip
    skip = G.objective.f(X) - G.objective.f(Z) < objX - fX
    if not skip:
        t  = (1 + sqrt(1+4*(1.+skipped)*tm**2))/2
        X  = Y + (tm-1)/t * (Y-Ym)
        Z  = X

    # minimize g(Y) +<f_grad(X),Y-X> + ||Y-X||^2/(2*mu)
    Yp = G.optimize( Y, X, F.objective.df(X) , 1/(2*mu) )

    tm = t
    t  = (1 + sqrt(1+2*(1.+skip)*t**2))/2

    Z  = Yp + (tm-1)/t * (Yp-Y)
    
    return (X,Yp,Y,Z,t,tm,skip)