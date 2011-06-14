"""
Linear-Quadratic-Linear-Exponential-Poisson model for Nuclear Norm optimizer
@author: kolia
"""
from inspect import getargspec

from numpy  import add, concatenate, eye, isnan, iscomplex,\
                   Inf, arange, max, min, minimum, log, size
#from numpy.linalg import inv, slogdet, det
from theano import function
import theano.tensor  as Th
from optimize import optimizer

from ops import matrix_inverse, det

import pylab as p

def quadratic_Poisson( theta = Th.dvector(), M    = Th.dmatrix() ,
                       STA   = Th.dvector(), STC  = Th.dmatrix()):

    tinvMt = Th.sum(Th.as_tensor_variable(Th.dot(matrix_inverse(M),theta),ndim=2),theta)

    return (  Th.log(det(M)) \
#          - 1. / (detM-self.mindet) \
             - tinvMt \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) )) / 2.
    

class objective:
    def __init__(self, theano_objective, init_params=None, **other):
        arg_names,_,_,arg_defaults = getargspec(theano_objective)
        self.init_params = sorted(init_params.items(), key=lambda (k,_): k)
#        self.args        = [v for n,v in sorted(args.items(), key=lambda (k,_): k)]
#        self.arg_names   = [n for n,v in sorted(args.items(), key=lambda (k,_): k)]
        self.args        = arg_defaults
        self.arg_names   = arg_names

        args = {}
        for arg_name,arg_value in zip(arg_names,arg_defaults):
            args[arg_name] = arg_value
        
        # self.params unpacks a flat vector into a dictionary of parameters
        n = 0
        params = Th.dvector()
        for name,template in self.init_params:
            args[name] = Th.reshape( params[n:n+size(template)], template.shape)
            n = n + size(template)

        flat_args  = [params] + self.args
        f  = theano_objective(**args)
        df = Th.grad( cost = L , wrt = params , consider_constant=self.args)
        
        def package(some_function):
            def packaged_function(self,flat_parameter,arg_dictionary):
                args = [arg_dictionary[n] for n in self.arg_names]
                return some_function(flat_parameter,*args)
            return packaged_function

        self.f     = package(function( flat_args,  f ))
        self.df    = package(function( flat_args, df ))
        for name,var in other:
            setattr(self,name, package(function( flat_args,  var ))

    def sum_RGC(self,op,g,(U,V2,V1),(N_spikes,STA,STC)):
        result = None
        for i,(n,sta,stc) in enumerate(zip(N_spikes,STA,STC)):
            IM = eye(self.N)-self.M(U,V2,V1[i,:])
#            detIM = det(IM)
#            print 'det(IM) : ', detIM
            term = n * g(U,V2,V1[i,:], inv(IM), det(IM), sta, stc)
            if any(isnan(term.flatten())):
                print 'oups'
                term = None
#                raise ArithmeticError('nan')
            if result is not None and term is not None:
                result = op( result , term )
            else:
                if term is not None:
                    result = term
        if result is None:
            return Inf
        else:
            return -result