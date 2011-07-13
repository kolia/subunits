from inspect   import getargspec
from functools import partial
from copy      import copy
from numpy     import size, array, asarray, concatenate, reshape
import numpy.linalg
from theano  import function, scan
import theano.tensor  as Th
from theano.gof import Op, Apply
from theano.sandbox.linalg import matrix_inverse

from IPython.Debugger import Tracer; debug_here = Tracer()


class LogDet(Op):
    """matrix determinant"""
    def make_node(self, x):
        x  = Th.as_tensor_variable(x)
        o  = Th.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])
    def perform(self, node, (x,), (z, )):
        try:
            s,ldet = numpy.linalg.slogdet(x)
            z[0] = asarray(s   , dtype=x.dtype)
        except:
            print 'Failed to compute determinant', x
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
    """matrix eigenvalues and eigenvectors"""
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
    if isinstance(x,type(0)):
        return Th.dscalar(name)
    if isinstance(x,type(array([]))):
        dtensor_x = Th.TensorType(dtype, (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(name),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')

def deep_iter(X):
    if isinstance(X,type(dict())):
        for _,x in sorted(X.items()):
            for y in deep_iter(x): yield y
    if isinstance(X,type([])):
        for   x in X:
            for y in deep_iter(x): yield y
    if isinstance(X,type(array([]))):
        yield X

def flat(X): return concatenate( [x.flatten() for x in deep_iter(X)] )

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

def unflat(template,X): return __unflat(template,X)[0]

def __reparameterize(func,reparam):
    output = reparam()
    if isinstance(output,type([])):
        def c(**dummy):
            result = Th.as_tensor_variable(0) 
            for repar in reparam(**dummy):
                repar.update(dummy)
                result = func(**repar)+result
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

def reparameterize(funcs,reparam):
    if isinstance(funcs,type([])):
        return [__reparameterize(f,reparam) for f in funcs]
    elif isinstance(funcs,type({})):
        return dict( (name,__reparameterize(f,reparam)) for name,f in funcs.items() )
    else:
        return __reparameterize(funcs,reparam)
    
def getargs(func):
    try:
        names,_,_,defaults = getargspec(func)
    except:
        [names,defaults] = zip( *func.keywords.items() )
    return dict( zip(names[-len(defaults):],defaults) )
    

class Base: pass

class term:
    '''
    Fields necessary for operating on objectives symbolically,
    and sufficient to build a numerical objective:
    self.theano, self.flatParam, self.Params, self.Args, self.init_params
    '''
    def __init__(self, init_params=None, differentiate=[], callback=None, **theano):
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
        self.callback = callback
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
        if self.callback:
            def callback(params): return self.callback(t,params)
            t.callback = callback
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



class sum_objective():
    def __init__(self, l, **other_fields):
        self.__common_init(l)

        for name,value in other_fields.items():  setattr(self,name,value)

        Params_Out = []
        n = 0
        for init_param,argz in zip(self.initParam,self.Args):
            Params_Out = Params_Out + [copy(argz)]
            for name,template in sorted(init_param.items()):
                Params_Out[-1][name] = Th.reshape( self.flatParam[n:n+size(template)], template.shape)
                n = n + size(template)

        names  = set(sum([objective.theano.keys() for objective in l],[]))
        outputs = dict((name,0) for name in names)
        for name in names:
            if (name[0] == 'd') and (name[1:] in l[0].theano):
                outputs[name] = Th.concatenate([o.theano[name](**P) for o,P in zip(self.terms,Params_Out)])
            else:            
                for objective,Params in zip(l,Params_Out):
                    try:    outputs[name] = outputs[name] + objective.theano[name](**Params)
                    except: print 'Error adding ', name, ' to ', objective
        for name,target in outputs.items():
            setattr(self, name, self.__wrap_theano(name,outputs[name]))

    def __wrap_theano(self,name,output):
        f = function([self.flatParam] + sum(self.__unpack(self.Args),[]), output)
        self.theano_functions[name] = f
        def wrapped_function(params,arg_dicts):
            argz = [[d[n] for n in sorted(args.keys())] for args,d in zip(self.Args,arg_dicts)]
            return f(self.flatten(params),*sum(argz,[]))
        return wrapped_function

    def __common_init(self, l):
        self.terms     = l
        self.Params    = [objective.Params      for objective in l]
        self.Args      = [objective.Args        for objective in l]
        self.initParam = [objective.init_params for objective in l]
        self.flatParam = Th.concatenate([objective.flatParam for objective in l])
        self.joinParam  = function( [ll.flatParam for ll in l] , self.flatParam )

        self.theano_functions = {}

        flatParams_Out = []
        n = 0
        for init_param in self.initParam:
            dn = sum([size(value) for _,value in sorted(init_param.items())])
            flatParams_Out = flatParams_Out + [self.flatParam[n:n+dn]]
            n = n + dn
        self.splitParam = function([self.flatParam], flatParams_Out )
    
    def __init2__(self, l):
        self.__common_init(l)
        for name in l[0].theano.keys():
            if (name[0] == 'd') and (name[1:] in l[0].theano):
                setattr(self,name, self.__wrap(name, lambda l : self.joinParam(*l) ))
            else:            
                setattr(self,name, self.__wrap(name, sum))

    def __wrap(self,name,reduction):
        def wrapped_function(params,arg_dicts):
#                argz = [[d[n] for n in sorted(args.keys())] for args,d in zip(self.Args,arg_dicts)]
            flats = self.splitParam( self.flatten(params) )                
            return reduction([ getattr(ll,name)(x,arg_dict) \
                   for ll,x,arg_dict in zip(self.terms,flats,arg_dicts)])
        return wrapped_function

    def __unpack(self,dict_list):
        return [[value for _,value in sorted(d.items())] for d in dict_list]

    def repack(self, dl, ll):
        return [term.repack(d,l) for term,d,l in zip(self.terms,dl,ll)]

    def inflate(self,x):
        if isinstance(x,type([])) and isinstance(x[0],type({})):
            return x
        if isinstance(x,type(array([]))):
            x = self.splitParam(x)
        if isinstance(x,type([])) and isinstance(x[0],type(array([]))):
            x = [term.splitParam(param) for term,param in zip(self.terms,x)]
        if isinstance(x,type([])):
            return self.repack(self.Params,x)
        raise TypeError('expects a numpy ndarray or a list of numpy ndarrays')

    def flat_list(self,params):
        if isinstance(params,type([])) and isinstance(params[0],type(dict())):
            params = self.__unpack(params)
        if isinstance(params,type([])) and isinstance(params[0],type([])):
            return [term.flatten(pl) for term,pl in zip(self.terms,params)]
        if isinstance(params,type(array([]))):
            return self.splitParam(params)
        raise TypeError('expects a list of dicts, list of lists or numpy ndarray')

    def flatten(self,params):
        if isinstance(params,type(array([]))):
            return params.flatten()
        return self.joinParam(*self.flat_list(params))
