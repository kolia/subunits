
## License for the Python wrapper
## ==============================

## Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>

## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.

## Modifications by Travis Oliphant and Enthought, Inc.  for inclusion in SciPy

from numpy import array, asarray, float64, int32, zeros, finfo, sqrt, mod, sign
from scipy.optimize import approx_fprime
from numpy.compat import asbytes

import _klbfgsb
reload(_klbfgsb)

import ipdb

from kolia_base import flat

__all__ = ['fmin_l_bfgs_barrier','optimizer']


_epsilon = sqrt(finfo(float).eps)

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
#        x, fx, dfx, _, _, _, _ = fmin_bfgs(f,init_params,fprime=df,
#                                                   gtol=gtol,maxiter=maxiter,
#                                                   callback=callback,
#                                                   barrier=barrier, disp=disp,
#                                                   full_output=True, **options)                                                   
        x , fx , d = fmin_l_bfgs_barrier(f, init_params, fprime=df, args=(),
                                         approx_grad=0, barrier=barrier,
                                         callback = callback,
                                         bounds=None, m=10, factr=1e0, 
                                         pgtol=1e-5, epsilon=1e-8,
                                         iprint=-1, maxfun=15000, disp=disp)
        if full_output:
            return (x,fx,d)
        else:
            return x
    return optimize
#        return Opt.fmin_ncg(self.f,params,fprime=self.df,avextol=1.1e-5,
#                            maxiter=10000,args=data,
#                            callback=cb)
#        return Opt.fmin_bfgs(self.f,params,fprime=self.df,
#                             gtol=1.1e-6,maxiter=10000,args=data,callback=cb)


def wrap_function(function):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x)
    return ncalls, function_wrapper

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


def fmin_l_bfgs_barrier(func, x0, fprime=None, args=(),
                        approx_grad=0, barrier=lambda _: False,
                        callback = None,
                        bounds=None, m=10, factr=1e1, 
                        pgtol=1e-5, epsilon=1e-8,
                        iprint=-1, maxfun=15000, disp=None):
    """
    Minimize a function func using the L-BFGS-B algorithm.

    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimise.
    x0 : ndarray
        Initial guess.
    fprime : callable fprime(x,*args)
        The gradient of `func`.  If None, then `func` returns the function
        value and the gradient (``f, g = func(x, *args)``), unless
        `approx_grad` is True in which case `func` returns only ``f``.
    args : sequence
        Arguments to pass to `func` and `fprime`.
    approx_grad : bool
        Whether to approximate the gradient numerically (in which case
        `func` returns only the function value).
    bounds : list
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction.
    m : int
        The maximum number of variable metric corrections
        used to define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms in an
        approximation to it.)
    factr : float
        The iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
        where ``eps`` is the machine precision, which is automatically
        generated by the code. Typical values for `factr` are: 1e12 for
        low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
        high accuracy.
    pgtol : float
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n} <= pgtol``
        where ``pg_i`` is the i-th component of the projected gradient.
    epsilon : float
        Step size used when `approx_grad` is True, for numerically
        calculating the gradient
    iprint : int
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint == 0`` means write messages to stdout; ``iprint > 1`` in
        addition means write logging information to a file named
        ``iterate.dat`` in the current working directory.
    disp : int, optional
        If zero, then no output.  If a positive number, then this over-rides
        `iprint` (i.e., `iprint` gets the value of `disp`).
    maxfun : int
        Maximum number of function evaluations.

    Returns
    -------
    x : array_like
        Estimated position of the minimum.
    f : float
        Value of `func` at the minimum.
    d : dict
        Information dictionary.

        * d['warnflag'] is

          - 0 if converged,
          - 1 if too many function evaluations,
          - 2 if stopped for another reason, given in d['task']

        * d['grad'] is the gradient at the minimum (should be 0 ish)
        * d['funcalls'] is the number of function calls made.

    Notes
    -----
    License of L-BFGS-B (Fortran code):

    The version included here (in fortran code) is 2.1 (released in 1997).
    It was written by Ciyou Zhu, Richard Byrd, and Jorge Nocedal
    <nocedal@ece.nwu.edu>. It carries the following condition for use:

    This software is freely available, but we expect that all publications
    describing work using this software , or all commercial products using it,
    quote at least one of the references given below.

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing , 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550 - 560.

    """
    x0 = asarray(x0).ravel()
    n, = x0.shape

    if bounds is None:
        bounds = [(None,None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    if disp is not None:
        if disp == 0:
            iprint = -1
        else:
            iprint = disp

    if approx_grad:
        def func_and_grad(x):
            f = func(x, *args)
            g = approx_fprime(x, func, epsilon, *args)
            return f, g
    elif fprime is None:
        def func_and_grad(x):
            f, g = func(x, *args)
            return f, g
    else:
        def func_and_grad(x):
            f = func(x, *args)
            g = fprime(x, *args)
            return f, g

    barr_calls, barr = wrap_function(barrier)

    nbd = zeros(n, int32)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(None, None): 0,
              (1, None) : 1,
              (1, 1) : 2,
              (None, 1) : 3}
    for i in range(0, n):
        l,u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    x = array(x0, float64)
    f = array(0.0, float64)
    g = zeros(n, float64) 
#    g = zeros((n,), float64)
    amax = array(0.0, float64)
    wa = zeros(2*m*n+4*n + 12*m**2 + 12*m, float64)
    iwa = zeros(3*n, int32)
    task = zeros(1, 'S60')
    csave = zeros(1,'S60')
    lsave = zeros(4, int32)
    isave = zeros(44, int32)
    dsave = zeros(29, float64)
    d = array(x0, float64)

    task[:] = 'START'

    factor = 1.

    n_function_evals = 0
    while 1:
#        x, amax, d, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
        _klbfgsb.setulb(m, x, amax, d, low_bnd, upper_bnd, nbd, f, g, factr,
                        pgtol, wa, iwa, task, csave, lsave,
                        isave, dsave, iprint)
        task_str = task.tostring()
        if task_str.startswith(asbytes('FG')):
            # minimization routine wants f and g at the current x
            n_function_evals += 1
            # Overwrite f and g:
            f, g = func_and_grad(x)
            g = g*factor
        elif task_str.startswith(asbytes('BARRIER')):
            amax = backtrack(x,d,barr)
            print 'amax', amax
        elif task_str.startswith(asbytes('NEW_X')):
            if callback is not None and mod(iprint,disp) is 0:
                callback(x)
    
            # new iteration
            if n_function_evals > maxfun:
                task[:] = 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
        else:
            if factor<-20.:
                break
            else:
                factor=-(factor+sign(factor))
                task[:] = 'START'

    task_str = task.tostring().strip(asbytes('\x00')).strip()
    if task_str.startswith(asbytes('CONV')):
        warnflag = 0
    elif n_function_evals > maxfun:
        warnflag = 1
    else:
        warnflag = 2


    d = {'grad' : g,
         'task' : task_str,
         'funcalls' : n_function_evals,
         'warnflag' : warnflag
        }
    return x, f, d

if __name__ == '__main__':
    def func(x):
        f = 0.25*(x[0]-1)**2
        for i in range(1, x.shape[0]):
            f += (x[i] - x[i-1]**2)**2
        f *= 4
        return f
    def grad(x):
        g = zeros(x.shape, float64)
        t1 = x[1] - x[0]**2
        g[0] = 2*(x[0]-1) - 16*x[0]*t1
        for i in range(1, g.shape[0]-1):
            t2 = t1
            t1 = x[i+1] - x[i]**2
            g[i] = 8*t2 - 16*x[i]*t1
        g[-1] = 8*t1
        return g

    factr = 1e7
    pgtol = 1e-5

    n=25
    m=10

    bounds = [(None,None)] * n
    for i in range(0, n, 2):
        bounds[i] = (1.0, 100)
    for i in range(1, n, 2):
        bounds[i] = (-100, 100)

    x0 = zeros((n,), float64)
    x0[:] = 3

    x, f, d = fmin_l_bfgs_barrier(func, x0, fprime=grad, m=m,
                                  factr=factr, pgtol=pgtol)
    print x
    print f
    print d
    x, f, d = fmin_l_bfgs_barrier(func, x0, approx_grad=1,
                                  m=m, factr=factr, pgtol=pgtol)
    print x
    print f
    print d
