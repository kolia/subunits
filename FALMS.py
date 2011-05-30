from numpy import sqrt, sum

class Objective: pass

def add_quadratic_term( objective, f='f', df='df', optimize='optimize' ):
    '''
    Generic: add quadratic term to objective, using its own optimizer.
    Adds method optimize_quadratic( X, Z, mu, **optimize_params), which
    minimizes f(X) + 0.5*||X-Z||**2 / mu in X for fixed Z.
    '''
    if isinstance(optimize,type('')):
        optimize = getattr(objective,optimize)
    def optimize_quadratic( X, Y, mu, **optimize_params):
        def quad_f (x): getattr(objective, f)(x) + sum((x-Y)*(x-Y))/mu*0.5
        def quad_df(x): getattr(objective,df)(x) +           (x-Y) /mu
        return optimize( x0=X , f=quad_f , df=quad_df , **optimize_params)
    objective.optimize_quadratic = optimize_quadratic
    return objective

#def nuclear_L2( rho ):
#    '''
#    Provides object with methods:
#        f(X)  = rho * ||X||_* 
#        df(X) = a subgradient of f
#        optimize_quadratic( X, Z, mu, **optimize_params), which
#            minimizes f(X) + 0.5*||X-Z||**2 / mu   in X for fixed Z.
#    '''
#    def optimize_quadratic( X, Y, mu ):
#        def quad_f (x): f(x) + sum((x-Y)*(x-Y))/mu*0.5
#        def quad_df(x): df(x) +           (x-Y) /mu
#        return 0
#    objective.optimize_quadratic = optimize_quadratic
#    return objective


def initialize_FALM(X): return (X, X , X*0 , X , 1 , 1 , 0 )

def FALM_step( F , G , mu , problem ):
    
    # Unpack loop variables
    X,Y,Ym,Z,t,tm,skipped = problem

    # minimize f(X) - <G.df(Z),X-Z> + ||X-Z||**2/(2*mu) in X for fixed Z
    Gdf = G.df(Z)
    X , objX , _ = F.optimize_quadratic( X, Z+Gdf , mu , full_output=True)
    
    # to skip or not to skip
    skip = G.f(X) - G.f(Z) < objX - mu * sum(Gdf*Gdf) - F.f(X)
    if not skip:
        t  = (1 + sqrt(1+4*(1.+skipped)*tm**2))/2
        X  = Y + (tm-1)/t * (Y-Ym)
        Z  = X

    # minimize g(Y) +<f_grad(X),Y-X> + ||Y-X||^2/(2*mu)
    Yp = G.optimize_quadratic( Y, X-F.df(X) , mu )

    tm = t
    t  = (1 + sqrt(1+2*(1.+skip)*t**2))/2

    Z  = Yp + (tm-1)/t * (Yp-Y)
    
    return (X,Yp,Y,Z,t,tm,skip)