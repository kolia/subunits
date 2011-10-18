import kolia_base as kb
from copy      import copy
import numpy
import numpy.linalg
from theano  import function
import theano.tensor  as Th
from theano.gof import Op, Apply
from theano.sandbox.linalg import matrix_inverse

#from IPython.Debugger import Tracer; debug_here = Tracer()

class LogDet(Op):
    """matrix determinant
    TODO: move this op to another file that request scipy.
    """
    def make_node(self, x):
        x = Th.as_tensor_variable(x)
        o = Th.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])
    def perform(self, node, (x,), (z, )):
        try:
            s,ldet = numpy.linalg.slogdet(x)
            z[0] = numpy.asarray(ldet, dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise
    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * matrix_inverse(x).T]
    def __str__(self):
        return "Det"
logdet = LogDet()

class Eig(Op):
    """SYMBOLIC matrix eigenvalues and eigenvectors, without gradient."""
    def make_node(self, x):
        x  = Th.as_tensor_variable(x)
        o1 = Th.vector(dtype=x.dtype)
        o2 = Th.matrix(dtype=x.dtype)
        return Apply(self, [x], [o1,o2])
    def perform(self, node, (x,), (z1,z2, )):
        try:
            w,v = numpy.linalg.eig(x)
            z1[0] = numpy.asarray(w, dtype=x.dtype)
            z2[0] = numpy.asarray(v, dtype=x.dtype)
        except:
            print 'Failed to compute eig', x
            raise
    def __str__(self):
        return "Eig"
eig = Eig()

def Hessian_along( cost , wrt , direction, consider_constant ):
    grad = Th.grad( cost=cost             , wrt=wrt , consider_constant=consider_constant )
    return Th.grad( Th.sum(grad*direction), wrt=wrt , consider_constant=consider_constant )

def shapely_tensor( name , x , dtype='float64'):
    '''Return SYMBOLIC tensor with the same dimensions and size as input.'''
    if isinstance(x,type(0)):
        return Th.dscalar(name)
    if isinstance(x,type(numpy.array([]))):
        dtensor_x = Th.TensorType(dtype, (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(name),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')


def differentiated(Obj=None,target=None,**Params):
    flatParam   = Th.concatenate([Th.flatten(Params[n]) for n in sorted(Obj.init_params.keys())])
#            flatParam   = Th.dvector()
    Params_Out = Obj.gen_Params_Out(flatParam,Params)
    arglist    = [Params_Out[name] for name in sorted(Obj.Args.keys())]
    return Th.grad( cost              = target(**Params_Out) ,
                    wrt               = flatParam ,
                    consider_constant = arglist)

import functools
class Base: pass

class Objective:
    '''Compile a list of theano expression generators given as named 
    arguments into an object with corresponding numerical functions, as a 
    function of the parameters in dictionary init_params.  All of these 
    functions can be transparently called either with parameters structured 
    as nested dictionaries and lists of numpy arrays, of by a flat vector of
    appropriate length concatenating all the parameters.
    
    Init_params only serves as a template for determining which variables
    are required, and their types and sizes are.
    All other dependencies of the theano targets should be provided as 
    arguments using objective.where(), which spawns off an objective with 
    the values of the arguments fixed to the specified values.

    For all parameter names appearing in init_params that are in the list 
    differentiate, a numerical function returning the gradient with respect 
    to parameters of that quantity will be provided, under the name 
    'd' + name; for example, the gradient of f will become objective.df

    Optionally, a callback function of the parameters'''
    def __init__(self, init_params=None, differentiate=[], mode=None, **theano):
        keydict = kb.getkwargs( theano.itervalues().next() )
#        self.defs          = [theano_defs]
        self.Args          = dict([(n,Th.as_tensor_variable(d,name=n)) for n,d in keydict.items()])

        self.init_params   = self.__intersect_dicts(self.Args.keys(),init_params)

        self.Params    = dict((n,shapely_tensor(n,x)) for n,x in self.init_params.items())
        self.flatParam  = Th.concatenate([Th.flatten(x) for _,x in sorted(self.Params.items())])
        for name in sorted(self.init_params.keys()):  del self.Args[name]

        self.Params_Out = self.gen_Params_Out(self.flatParam,self.Args)

        self.splitParam = \
        function([self.flatParam],[self.Params_Out[name] for name in sorted(self.Params.keys())])

        self.theano = theano
        self.theano_functions = {}
#        self.callback = callback
        self.differentiate = differentiate

        for name in self.differentiate:
            if ('d'+name) not in self.theano:
                self.theano['d'+name] = functools.partial(differentiated,Obj=self,target=self.theano[name])

        self.arglist = [self.Params_Out[name] for name in sorted(self.Args.keys())]

#        for name in self.hessian:
#            if ('H'+name) not in self.theano_functions and name in self.theano:
#                gen = self.theano[name]
#                direction = 
#                self.theano_functions['H'+name] = function( [self.flatParam]+self.arglist) ,
#                    Hessian_along( gen(self.theano[name]) ,  )

        for name, gen in self.theano.items():
            self.theano_functions[name] = function([self.flatParam]+self.arglist,gen(**self.Params_Out),mode=mode)

    def __get_state__(self):
        return (Args)
#        return (Args, theano_functions)

    def __set_state__(self,state):
        (Args) = state
#        (Args, theano_functions) = state
        self.Args             = Args
#        self.theano_functions = theano_functions
#        self.theano           = theano

    def where(self,**args):  #targets=self.theano,
        t = Base()
        t.args = [args[n] for n in sorted(self.Args.keys())]
        print 'Objective.where: the following arguments have been fixed:'
        print [(n,args[n].shape) for n in sorted(self.Args.keys())]
        def package(some_function):
            def packaged_function(params): return some_function(kb.flat(params),*t.args)
            return packaged_function
        for name,gen in self.theano.items():
            setattr(t,name,package(self.theano_functions[name]))
        def with_callback(callbk):
            def callback(params): return callbk(t,params)
            t.callback = callback
            return t
        t.with_callback = with_callback
#        if self.callback:
#            def callback(params): return self.callback(t,params)
#            t.callback = callback
        t.flat   = self.flat
        t.unflat = self.unflat
        return t

    def gen_Params_Out(self,flatParam,Args):
        Params_Out = copy(Args)
        n = 0
        for name,template in sorted(self.init_params.items()):
            Params_Out[name] = Th.reshape( flatParam[n:n+numpy.size(template)], template.shape)
            n = n + numpy.size(template)
        return Params_Out

    def flat(self,X): return kb.flat(X)
    def unflat(self,X): return kb.unflat(self.init_params,X)

#    def __differentiate(self,target):
#        def gen_differential(**Params):
#            flatParam   = Th.concatenate([Th.flatten(Params[n]) for n in sorted(self.init_params.keys())])
##            flatParam   = Th.dvector()
#            Params_Out = self.gen_Params_Out(flatParam,Params)
#            arglist    = [Params_Out[name] for name in sorted(self.Args.keys())]
#            return Th.grad( cost              = target(**Params_Out) ,
#                            wrt               = flatParam ,
#                            consider_constant = arglist)
#        return gen_differential

    def __intersect_dicts(self,names,d):
        out = {}
        for name in names:
            try:              out[name] = d[name]
            except KeyError:  ()
        return out
