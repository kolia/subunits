from inspect   import getargspec
from functools import partial
from copy      import copy
from numpy     import size, array, asarray, concatenate, reshape
import numpy.linalg
from theano  import function
import theano.tensor  as Th
from theano.gof import Op, Apply
from theano.sandbox.linalg import matrix_inverse

#from IPython.Debugger import Tracer; debug_here = Tracer()


class LogDet(Op):
    """SYMBOLIC log-determinant, with gradient."""
    def make_node(self, x):
        x  = Th.as_tensor_variable(x)
        o  = Th.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])
    def perform(self, node, (x,), (z, )):
        try:
            s,ldet = numpy.linalg.slogdet(x)
            z[0] = asarray(s   , dtype=x.dtype)
        except:
            print 'Failed to compute determinant',
            raise
    def grad(self, inputs, g_outputs):
        gz = g_outputs
        x, = inputs
#        debug_here()
        return [gz * matrix_inverse(x).T]
    def __str__(self):
        return "LogDet"
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
            z1[0] = asarray(w, dtype=x.dtype)
            z2[0] = asarray(v, dtype=x.dtype)
        except:
            print 'Failed to compute eig', x
            raise
    def __str__(self):
        return "Eig"
eig = Eig()


def shapely_tensor( name , x , dtype='float64'):
    '''Return SYMBOLIC tensor with the same dimensions and size as input.'''
    if isinstance(x,type(0)):
        return Th.dscalar(name)
    if isinstance(x,type(array([]))):
        dtensor_x = Th.TensorType(dtype, (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(name),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')

def deep_iter(X):
    '''Generates depth-first iterator over numpy array values 
    in nested dictionaries and lists: useful for generic flattening of 
    structured parameters.'''
    if isinstance(X,type(dict())):
        for _,x in sorted(X.items()):
            for y in deep_iter(x): yield y
    if isinstance(X,type([])):
        for   x in X:
            for y in deep_iter(x): yield y
    if isinstance(X,type(array([]))):
        yield X

def flat(X):
    '''Flatten and concatenate all the numpy arrays contained in 
    possibly nested dictonaries and lists.'''
    return concatenate( [x.flatten() for x in deep_iter(X)] )

def __unflat(template,X,n=0):
    if isinstance(template,type(array([]))):
        return reshape( X[n:n+template.size] , template.shape) , n+template.size
    elif isinstance(template,type(dict())):
        iterset = sorted(template.items())
        result  = {}
    elif isinstance(template,type([])):
        iterset = enumerate(template)
        result  = [None for i in len(template)]
    else:
        raise TypeError('unflat expects numpy ndarray, list or dict')
    for key,x in iterset:
        rec , n  = __unflat(x,X,n=n)
        result[key] = rec
    return result,n

def unflat(template,X):
    '''Populate all the numpy arrays contained in 
    possibly nested dictonaries and lists of template, taking the 
    values from flat vector X of appropriate length: this is the 
    inverse operation of flat(X).'''    
    return __unflat(template,X)[0]

def __reparameterize(func,reparam,reducer,zero):
    output = reparam()
    if isinstance(output,type([])):
        def c(**dummy):
            result = zero
            for repar in reparam(**dummy):
                repar.update(dummy)
                new = func(**repar)
#                if result is None:  result = new*0
                result = reducer( result, new )
            return result
    else:
        def c(**dummy):  return func(**dummy.update(reparam(**dummy)))
    keydict = getargs( reparam )
    arguments,_,_,defaults = getargspec(func)
    if isinstance(output,type([])): output = output[0]
    for name,value in zip(arguments,defaults):
        if name not in output:  keydict[name] = value
    c.__name__ = func.__name__
    result = partial(c,**keydict)
    result.__name__ = func.__name__
    return result

def reparameterize(funcs,reparam,reducer=lambda r,x: r+x, zero=Th.as_tensor_variable(0.) ):
    '''Reparameterize a symbolic expression as a function of some new 
    variables.'''
    if isinstance(funcs,type([])):
        return [__reparameterize(f,reparam,reducer,zero) for f in funcs]
    elif isinstance(funcs,type({})):
        return dict( (name,__reparameterize(f,reparam,reducer,zero)) for name,f in funcs.items() )
    else:
        return __reparameterize(funcs,reparam)
    
def getargs(func):
    '''Get the named arguments of a function or partial, with 
    their default values.'''
    try:
        names,_,_,defaults = getargspec(func)
    except:
        [names,defaults] = zip( *func.keywords.items() )
    return dict( zip(names[-len(defaults):],defaults) )
    

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
    def __init__(self, init_params=None, differentiate=[], **theano):
        keydict = getargs( theano.itervalues().next() )
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
                self.theano['d'+name] = self.__differentiate(self.theano[name])

        self.arglist = [self.Params_Out[name] for name in sorted(self.Args.keys())]
        for name,gen in self.theano.items():
            self.theano_functions[name] = function([self.flatParam]+self.arglist,gen(**self.Params_Out))

    def where(self,**args):
        t = Base()
        t.args = [args[n] for n in sorted(self.Args.keys())]
        def package(some_function):
            def packaged_function(params): return some_function(flat(params),*t.args)
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
            Params_Out[name] = Th.reshape( flatParam[n:n+size(template)], template.shape)
            n = n + size(template)
        return Params_Out

    def flat(self,X): return flat(X)
    def unflat(self,X): return unflat(self.init_params,X)

    def __differentiate(self,target):
        def gen_differential(**Params):
            flatParam   = Th.concatenate([Th.flatten(Params[n]) for n in sorted(self.init_params.keys())])
#            flatParam   = Th.dvector()
            Params_Out = self.gen_Params_Out(flatParam,Params)
            arglist    = [Params_Out[name] for name in sorted(self.Args.keys())]
            return Th.grad( cost              = target(**Params_Out) ,
                            wrt               = flatParam ,
                            consider_constant = arglist)
        return gen_differential

    def __intersect_dicts(self,names,d):
        out = {}
        for name in names:
            try:              out[name] = d[name]
            except KeyError:  ()
        return out