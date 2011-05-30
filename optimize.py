# -*- coding: utf-8 -*-
"""
@author: Slight modification of scipy.optimize.fmin_bfgs by kolia.
"""

from scipy.optimize import approx_fprime, line_search

import numpy
from numpy import asarray, sqrt, Inf, isinf
import scipy.optimize.linesearch

# from IPython.Debugger import Tracer; debug_here = Tracer()

_epsilon = sqrt(numpy.finfo(float).eps)

def get_attribute(o,attribute):
    try:
        return getattr(o,attribute)
    except AttributeError:
        return None

def optimizer( objective , f='f' , df='df', barrier='barrier', 
               callback='callback', args='args', init='init' ):
    barrier  = get_attribute( objective, barrier  )
    callback = get_attribute( objective, callback )
    args     = get_attribute( objective, args     )
    init     = get_attribute( objective, init     )
    df       = get_attribute( objective, df       )
    def optimize(init=init, args=args, f=getattr(objective, f), df=df,
                 barrier=barrier, callback=callback, gtol=1.1e-6, 
                 maxiter=1000 , full_output=False ):
        if callback is None:
            cb = None
        else:
            def cb(para): callback(para,args)
        x, fx, dfx, _, _, _, _ = fmin_barrier_bfgs(f,init,fprime=df,
                                                   gtol=1.1e-6,maxiter=1000,
                                                   args=args,callback=cb,
                                                   barrier=barrier,
                                                   full_output=True)
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


def backtrack(f,xk,pk,barrier):
    if barrier is None: return 50
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x)
        Objective function.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    barrier : callable
        barrier(x) returns true iff a barrier has been jumped.
        
    """
    # initial phase: find a point on other side of barrier by *2.
    a  = 0.00001
    while True:
        if a>500.:
            return 500.
        if barrier(xk + a*pk): break
        a = a * 2.

    # refinement phase: 8 rounds of dichotomy
    left  = 0
    right = a
    while True:
        if barrier(xk + (right+left)/2.*pk):
            right = (right+left)/2.
        else:
            left  = (right+left)/2.
        if left>0 or right<1e-12: break    
#    print 'amax : ', left
    return left


def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(abs(x))
    elif ord == -Inf:
        return numpy.amin(abs(x))
    else:
        return numpy.sum(abs(x)**ord,axis=0)**(1.0/ord)


def wrap_function(function, arg):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, arg)
    return ncalls, function_wrapper


def fmin_barrier_bfgs(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf,
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
    func_calls, f    = wrap_function(f      , args)
    barr_calls, barr = wrap_function(barrier, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
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

#        debug_here()

        amax = backtrack(f,xk,pk,barr)          # scipy.optimize.fmin_bfgs 
                                                # modified here 
                                                # and line_searches below!
#        amax = 50.
        
        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
           scipy.optimize.linesearch.line_search(f,myfprime,xk,pk,gfk,
                                  old_fval,old_old_fval,amax=amax)
        if alpha_k is None:  # line search failed try different one.
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search(f,myfprime,xk,pk,gfk,
                                 old_fval,old_old_fval,amax=amax)
            if alpha_k is None:
                # This line search also failed to find a better solution.
                warnflag = 2
                break
        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
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
            print "Warning: Desired error not necessarily achieved" \
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
