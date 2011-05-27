from numpy import sqrt, dot, sum

def add_quadratic_term( objective ):
    '''Generic: add quadratic term to objective, using its own optimizer.'''
    def quad_optimize( X, Y, L, mu):
        def quad_f (x): objective.f (x) + dot(L,x-Y) + sum((x-Y)*(x-Y))*mu
        def quad_df(x): objective.df(x) +     L      +         2*(x-Y) *mu
        return objective.optimize( x0=X , f=quad_f , df=quad_df )
    objective.optimize_quadratic = quad_optimize
    return objective



def initialize_FALM(X): return (X, X , X*0 , X , 1 , 1 , 0 )

def FALM_step( F , G , rho , mu , problem ):
    
    # Unpack loop variables
    X,Y,Ym,Z,t,tm,skipped = problem

    # minimize f(X) - <G.fx_grad(Z),X-Z> + ||X-Z||**2/(2*mu) in X for fixed Z
    X , fX , objX = F.optimize_quadratic( X, Z, -G.df(Z) , 1/(2*mu) )
    
    # to skip or not to skip
    skip = G.f(X) - G.f(Z) < objX - fX
    if not skip:
        t  = (1 + sqrt(1+4*(1.+skipped)*tm**2))/2
        X  = Y + (tm-1)/t * (Y-Ym)
        Z  = X

    # minimize g(Y) +<f_grad(X),Y-X> + ||Y-X||^2/(2*mu)
    Yp = G.optimize_quadratic( Y, X, F.df(X) , 1/(2*mu) )

    tm = t
    t  = (1 + sqrt(1+2*(1.+skip)*t**2))/2

    Z  = Yp + (tm-1)/t * (Yp-Y)
    
    return (X,Yp,Y,Z,t,tm,skip)