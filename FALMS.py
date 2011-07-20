from numpy        import sqrt, sum , dot , maximum, minimum, sign, abs
from numpy.linalg import svd
from optimize import optimizer
from copy import copy

#from IPython.Debugger import Tracer; debug_here = Tracer()

def set_argument( objective , argument , fields=['f','callback','df','barrier'] ):
    def add_field(o,field):
        try:
            fff = copy(getattr(o, field))
            def ff(x,*_): return fff(x,argument)
            setattr(o, field, copy(ff))
        except:
            print 'Warning: ', field, ' method not found in objective: could not set_argument.'
        return o
    for field in fields: objective = add_field(objective,field)
    return objective

class Objective: pass

def add_L2_term( objective, f='f', df='df' ):
    '''
    Generic: add quadratic term to objective, using its own optimizer.
    Adds method optimize_quadratic( X, Z, mu, **optimize_params), which
    minimizes f(X) + 0.5*||X-Z||**2 / mu in X for fixed Z.
    '''
    def optimize_L2( X, Y, mu, **optimize_params):
        def L2_f (x,*_): return getattr(objective, f)(x) + 0.5*sum((x-Y)*(x-Y))/mu
        def L2_df(x,*_): return getattr(objective,df)(x) +               (x-Y) /mu
        objective.L2_f  = L2_f
        objective.L2_df = L2_df
        opter = optimizer( objective , init_params=X , f='L2_f' , df='L2_df' , 
                          maxiter=100, gtol=1.1e-6, disp=0,  **optimize_params)
        return opter()
    objective.optimize_L2 = optimize_L2
    return objective

def nuclear_L2( rho , get_matrix=None ):
    '''
    Provides object with methods:
        f(X)  = rho * ||X||_* 
        df(X) = a subgradient of f
        optimize_quadratic( X, Z, mu ), which
            minimizes f(X) + 0.5*||X-Z||**2 / mu   in X for fixed Z.
    '''
    def get_svd(X):
        if get_matrix:
            X = get_matrix(X)
        [U,ss,V] = svd(X)
        U = U[:,0:len(ss)]
        V = V[0:len(ss),:]
        return U,ss,V
    def f( X ):
        [U,s,V] = get_svd(X)
        return rho * sum(s)
    def df( X ):
        [U,ss,V] = get_svd(X)
        return rho * dot(U,V).flatten()
    def optimize_L2( _, Z, mu ):
        [U,s,V] = get_svd(Z)
        return dot( U * maximum(s-rho*mu,0) , V ).flatten()
    O = Objective()
    O.f           = f
    O.df          = df
    O.optimize_L2 = optimize_L2
    return O


def L1_L2( rho , get_matrix=None ):
    '''
    Provides object with methods:
        f(X)  = rho * ||X||_1 
        df(X) = a subgradient of f
        optimize_quadratic( X, Z, mu ), which
            minimizes f(X) + 0.5*||X-Z||**2 / mu   in X for fixed Z.
    '''
    def f( X ):  return rho * sum(abs(X))
    def df( X ): return rho * sign(X)
    def optimize_L2( _, Z, mu ):
        X = Z
        X[X>0] = maximum( 0, X[X>0]-mu*rho )
        X[X<0] = minimum( 0, X[X<0]+mu*rho )
        return X
    O = Objective()
    O.f           = f
    O.df          = df
    O.optimize_L2 = optimize_L2
    return O

def initialize(X):
    '''Initialize FALM with initial condition X.'''
    return [X, X , X , X , 1 , 1 , 0 ]

def step( F , G , mu , problem ):
    
    # Unpack loop variables
    X,Y,Ym,Z,t,tm,skipped = problem

    # minimize f(X) + <G.df(Z),X-Z> + ||X-Z||**2/(2*mu) in X for fixed Z
    Gdf = G.df(Z)
    X , objX , _ = F.optimize_L2( X, Z+mu*Gdf , mu , full_output=True)
    
    # to skip or not to skip
    skip = G.f(X) - G.f(Z) > objX - mu/2 * sum(Gdf*Gdf) - F.f(X)
    if not skip:
        t  = (1 + sqrt(1+4*(1.+skipped)*tm**2))/2
        X  = Y + (tm-1)/t * (Y-Ym)
        Z  = X

    # minimize g(Y) +<f_grad(X),Y-X> + ||Y-X||^2/(2*mu)
    Yp = G.optimize_L2( Y, X-mu*F.df(X) , mu )

    tm = t
    t  = (1 + sqrt(1+2*(1.+skip)*t**2))/2

    Z  = Yp + (tm-1)/t * (Yp-Y)
    
    return [X,Yp,Y,Z,t,tm,skip]
    

def falms( init_params , F, G, mu=0.1, maxiter=200, ftol=2e-7, callback=None,verbose=True):
    '''FALMS optimization: main loop.  Not as simple as you'd hope, but works.
    Mu is ratcheted downwards whenever objective increases; if objective 
    increases by more than 0.2 in an iteration, that iteration is disgarded.
    Returns the last unskipped iteration.
    '''
    print
    print
    print 'Starting FALMS optimization...'
    # Compile symbolic objectives into numerical objective.
    
    current_X  = F.flat(init_params)
    old_X      = current_X
    unskipped  = current_X
    objval_old = 1e10
    falmer     = initialize( current_X )
    for j in range(maxiter):
        oldfalmer   = falmer
        falmer = step( F , G , mu , oldfalmer )
        objval = F.f(falmer[0])+G.f(falmer[0])
        L2change= sum((falmer[0]-old_X)**2) + sum((falmer[0]-current_X)**2)
        if verbose:
            print 'STEP %3d   mu: %6.2g   f: %10.8g  dL2: %7.2g' % (j,mu,objval,L2change),
        if not falmer[6]:
            unskipped = falmer[0]
        if callback is not None:    callback(falmer)
        if mu<1e-8 or (j>30 and L2change < ftol): break
        old_X = current_X
        current_X = falmer[0]
        if objval>=objval_old:
            mu = mu/2
            falmer = oldfalmer
        else:
            objval_old  = objval
    return F.unflat( unskipped )