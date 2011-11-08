"""
Minor changes to the BFGS optimizer in scipy.optimize
"""
from scipy.optimize import approx_fprime
import numpy
from numpy import asarray, sqrt, Inf, isinf, minimum, isfinite
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2

from kolia_base import flat

#from IPython.Debugger import Tracer; debug_here = Tracer()

_epsilon = sqrt(numpy.finfo(float).eps)

def __get_attribute(o,attribute):
    if isinstance(attribute,type('')):
        try:
            return getattr(o,attribute)
        except AttributeError:
            return None
    return attribute

def optimizer( objective , f='f' , df='df', barrier='barrier', maxiter=500,
               callback='callback', init_params='init_params' , gtol=1.1e-7,
               disp=1, **options):
    '''Return a function which optimizes over an objective function.
    By default, the objective is objective.f and its gradient objective.df.
    Optional barrier  function defaulting to objective.barrier.
    Optional callback function defaulting to objective.callback.
    Optional default init_params defaulting to objective.init_params.
    '''
    barrier     = __get_attribute( objective, barrier  )
    callback    = __get_attribute( objective, callback )
    init_params = __get_attribute( objective, init_params)
    df          = __get_attribute( objective, df       )
    f           = getattr(objective, f)
    if 'full_output' not in options:
        full_output = False
    else:
        full_output = options['full_output']
    def optimize(init_params=init_params, f=f, 
                 df=df, barrier=barrier, callback=callback, gtol=gtol, 
                 maxiter=maxiter , full_output=full_output , **options):
        init_params = flat(init_params)
        x, fx, dfx, _, _, _, _ = fmin_barrier_bfgs(f,init_params,fprime=df,
                                                   gtol=gtol,maxiter=maxiter,
                                                   callback=callback,
                                                   barrier=barrier, disp=disp,
                                                   full_output=True, **options)
        if full_output:
            return (x,fx,dfx)
        else:
            return x
    return optimize
#        return Opt.fmin_ncg(self.f,params,fprime=self.df,avextol=1.1e-5,
#                            maxiter=10000,args=data,
#                            callback=cb)
#        return Opt.fmin_bfgs(self.f,params,fprime=self.df,
#                             gtol=1.1e-6,maxiter=10000,args=data,callback=cb)

init_a = [0.001]
def backtrack(xk,pk,barrier):
    if barrier is None: return 500
    """Find large(st) alpha such that barrier(xk+alpha*pk) is 0.

    Parameters
    ----------
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    barrier : callable
        barrier(x) returns true iff a barrier has been jumped.
        
    """
    # initial phase: find a point on other side of barrier by *1.5
    a  = init_a[0]
    while True:
        if a>5000.:
            init_a[0] = 3500.
            return 5000.
        if barrier(xk + a*pk): break
        a = a * 1.5

    # refinement phase: 8 rounds of dichotomy
    left  = 0
    right = a
    while True:
        if barrier(xk + (right+left)/2.*pk):
            right = (right+left)/2.
        else:
            left  = (right+left)/2.
        if left>0 or right<1e-300: break
#    print 'amax : ', left
    init_a[0] = left
    return left


def simple_search(f,xk,pk,amax):
    oldval = f(xk)
    alpha  = min(amax,1)
    for i in range(20):
        current = f(xk+alpha*pk)        
        if current<oldval: return alpha, current
        alpha = alpha/3.
    return None, None

def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(abs(x))
    elif ord == -Inf:
        return numpy.amin(abs(x))
    else:
        return numpy.sum(abs(x)**ord,axis=0)**(1.0/ord)


def wrap_function(function):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x)
    return ncalls, function_wrapper


def fmin_barrier_bfgs(f, x0, fprime=None, gtol=1e-6, norm=Inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None, barrier=None):
    """Minimize a function using the BFGS algorithm without jumping a barrier.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable f'(x,*args)
        Gradient of f.
    args : tuple
        Extra arguments passed to f and fprime.
    gtol : float
        Gradient norm must be less than gtol before succesful termination.
    norm : float
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray
        If fprime is approximated, use this value for the step size.
    callback : callable
        An optional user-supplied function to call after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    barrier : callable
        barrier(x) returns true iff a barrier has been jumped.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e. the inverse hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
    allvecs  :  list
        Results at each iteration.  Only returned if retall is True.

    Other Parameters
    ----------------
    maxiter : int
        Maximum number of iterations to perform.
    full_output : bool
        If True,return fopt, func_calls, grad_calls, and warnflag
        in addition to xopt.
    disp : bool
        Print convergence message if True.
    retall : bool
        Return a list of results at each iteration if True.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS) See Wright, and Nocedal 'Numerical
    Optimization', 1999, pg. 198.

    """
    x0 = asarray(x0).squeeze()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0)*200
    func_calls, f    = wrap_function(f      )
    barr_calls, barr = wrap_function(barrier)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime)
#    debug_here()
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N,dtype=int)
    Hk = I
    old_fval = f(x0)
    old_old_fval = old_fval + 5000
    xk = x0
    if retall:
        allvecs = [x0]
    sk = [2*gtol]
    warnflag = 0
    gnorm = vecnorm(gfk,ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -numpy.dot(Hk,gfk)

        amax = backtrack(xk,pk,barr)          # scipy.optimize.fmin_bfgs 
                                                # modified here 
                                                # and line_searches below!
#        amax = 50.
        famax = f(xk+amax*pk)
        bamax = barr(xk+amax*pk)
        if disp:
            print 'amax%d:%15g  f:%10g f(amax):%10g  #b:%d  #f:%d' \
                % (bamax,amax,old_fval,famax,barr_calls[0],func_calls[0]),
        if callback is not None:
            callback(xk)

        if (bamax == 0) and (famax < old_fval):
            alpha_k  = amax
            old_fval2 = famax
        else:
            alpha_k = None
            try:
                alpha_k, fc, gc, old_fval2, old_old_fval2, gfkp1 = \
                   line_search_wolfe2(f,myfprime,xk,pk,gfk,
                                      old_fval,old_old_fval,amax=amax)
            except: 
                if disp : print 'Warning: error in line_search_wolfe2..'
                
            if alpha_k is not None:
                print 'w2 ',
            else:
                # line search failed: try different one.
                alpha_k, fc, gc, old_fval2, old_old_fval2, gfkp1 = \
                         line_search_wolfe1(f,myfprime,xk,pk,gfk,
                                            old_fval,old_old_fval,amax=amax)
    
            if alpha_k is None:
                alpha_k , old_fval2 = simple_search(f,xk,pk,amax)
                if alpha_k is not None: 
                    print ' simple ',
    
            if alpha_k is None:
                pk = -pk
                alpha_k , old_fval2 = simple_search(f,xk,pk,amax)
                if alpha_k is not None:
                    print ' simple2 ',

##        debug_here()
#        if old_fval>famax and isfinite(famax):
#            alpha_k = amax
#            old_fval = famax
#            gfkp1   = myfprime(xk + amax*pk)
#        else:
#            alpha_k = minimum(alpha_k,amax)
            
            print
            
            if alpha_k is not None:
                bval = barr(xk + alpha_k*pk)
            else:
                bval = 1
            if bval:
                if bamax:
                    warnflag = 2
                    break
                if famax<old_fval:
                    alpha_k = amax
                else:
                    alpha_k , old_fval = simple_search(f,xk,pk,amax)
                    print ' simple3 ',

#        if alpha_k is not None:
#            old_fval= f(xk + alpha_k*pk) 
#            gfkp1   = myfprime(xk + alpha_k*pk)
        
        if alpha_k is None:
            old_fval = f(xk)
            break
        
        old_old_fval = old_fval
        old_fval = old_fval2

        xkp1 = xk + alpha_k * pk
        gfkp1 = myfprime(xk + alpha_k * pk)

        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        
        old_fval = f(xk)    
        if not isfinite(old_fval):
            pass
#            debug_here()
        
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        gnorm = vecnorm(gfk,ord=norm)
        if (gnorm <= gtol):
            break

        try: # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk,sk))
        except ZeroDivisionError:
            rhok = 1000.0
            print "Divide-by-zero encountered: rhok assumed large"
        if isinf(rhok): # this is patch for numpy
            rhok = 1000.0
            print "Divide-by-zero encountered: rhok assumed large"
        A1 = I - sk[:,numpy.newaxis] * yk[numpy.newaxis,:] * rhok
        A2 = I - yk[:,numpy.newaxis] * sk[numpy.newaxis,:] * rhok
        Hk = numpy.dot(A1,numpy.dot(Hk,A2)) + rhok * sk[:,numpy.newaxis] \
                 * sk[numpy.newaxis,:]

    if disp or full_output:
        fval = old_fval
    if warnflag == 2:
        if disp:
            print "Warning: Desired error not necessarily achieved " \
                  "due to precision loss"
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: %d" % func_calls[0]
            print "         Barrier  evaluations: %d" % barr_calls[0]
            print "         Gradient evaluations: %d" % grad_calls[0]

    elif k >= maxiter:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: %d" % func_calls[0]
            print "         Barrier  evaluations: %d" % barr_calls[0]
            print "         Gradient evaluations: %d" % grad_calls[0]
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: %d" % func_calls[0]
            print "         Barrier  evaluations: %d" % barr_calls[0]
            print "         Gradient evaluations: %d" % grad_calls[0]

    if full_output:
        retlist = xk, fval, gfk, Hk, func_calls[0], grad_calls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = xk
        if retall:
            retlist = (xk, allvecs)

    return retlist
