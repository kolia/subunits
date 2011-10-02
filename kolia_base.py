# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:40:03 2011

@author: - kolia
"""
from inspect   import getargspec
def getkwargs(func):
    '''Get the named arguments of a function or partial, with 
    their default values.'''
    try:
        names,_,_,defaults = getargspec(func)
    except:
        [names,defaults] = zip( *func.keywords.items() )
    return dict( zip(names[-len(defaults):],defaults) )

def format_kwargs(func,arg_dict):
    kwargs = getkwargs(func)
    for name in kwargs:
        if arg_dict.has_key(name): kwargs[name] = arg_dict[name]
    return kwargs

from numpy import array , reshape , concatenate
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


from inspect   import getargspec
from functools import partial

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
    keydict = getkwargs( reparam )
    arguments,_,_,defaults = getargspec(func)
    if isinstance(output,type([])): output = output[0]
    for name,value in zip(arguments,defaults):
        if name not in output:  keydict[name] = value
    c.__name__ = func.__name__
    result = partial(c,**keydict)
    result.__name__ = func.__name__
    return result

def reparameterize(funcs,reparam,reducer=lambda r,x: r+x, zero=0. ):
#def reparameterize(funcs,reparam,reducer=lambda r,x: r+x, zero=Th.as_tensor_variable(0.) ):
    '''Reparameterize a symbolic expression as a function of some new 
    variables.'''
    if isinstance(funcs,type([])):
        return [__reparameterize(f,reparam,reducer,zero) for f in funcs]
    elif isinstance(funcs,type({})):
        return dict( (name,__reparameterize(f,reparam,reducer,zero)) for name,f in funcs.items() )
    else:
        return __reparameterize(funcs,reparam)