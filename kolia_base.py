# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:40:03 2011

@author: - kolia
"""
import matplotlib.collections
import matplotlib.transforms
import pylab
def plot_circles( sizes=1., offsets=[(0.,0.),(1.,1.)], linewidth=2.,
                 facecolors=(0.,0.,0.,0.1), edgecolors=(0.,0.,0.,0.3), **kwargs ):
    '''Plot a bunch of circles at locations offsets. List face- and edge-colors.'''
    ax = pylab.gca()
    circles = matplotlib.collections.EllipseCollection( sizes, sizes, 0.,
                   facecolors=facecolors, edgecolors=edgecolors, units='xy',
                   offsets=offsets,
                   transOffset=ax.transData, **kwargs)
    circles.set_linewidth(linewidth)
    ax.add_collection(circles)
    ax.autoscale_view()
    ax.set_aspect('equal')
    return circles
#pylab.show()

import matplotlib.colors
def values_to_color( values , color ):
    ''''''
    m = min(values)
    M = max(values)
    values = (values-m)/(M-m)
    (r,g,b,a)  = matplotlib.colors.ColorConverter().to_rgba(color)
    return [(r*v,g*v,b*v,a) for v in values]

def values_to_alpha( values , color ):
    ''''''
    m = min(values)
    M = max(values)
    values = (values-m+0.1*(M-m))/(1.1*(M-m))
    (r,g,b)  = matplotlib.colors.ColorConverter().to_rgb(color)
    return [(r,g,b,v) for v in values]


import numpy
from   matplotlib.ticker import *
import cPickle
def print_sparse_rows(V,precision=1e-2):
    lasti = -10
    for i,v in enumerate(V):
        if lasti == i-2: print
        if numpy.sum(numpy.abs(v))>precision:
            print i,' : ', v
            lasti = i

def plot_filters(X,same_scale=True):
    for i in range(X.shape[0]-1,-1,-1):
        ax = pylab.subplot(X.shape[0]*2,1,i*2+1)
        pylab.plot(numpy.arange(X.shape[1]),X[i,:])
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.yaxis.set_major_locator( LinearLocator(numticks=2) )
        ax.xaxis.set_major_locator( IndexLocator(overcompleteness,0) )
 
def save(result,name,where='/Users/kolia/Desktop'):
    savefile = open("%s/%s.pyckle" % (where,name),'w')
    cPickle.dump(result,savefile,protocol=2)
    savefile.close()
 
def load(name,where='/Users/kolia/Desktop'):
    savefile = open('%s/%s.pyckle' % (where,name),'r')
    result = cPickle.load(savefile)
    savefile.close()
    return result
     



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
    if isinstance(X,type(array([]))):
        yield X
    elif isinstance(X,type(dict())):
        for _,x in sorted(X.items()):
            for y in deep_iter(x): yield y
    elif isinstance(X,type([])):
        for   x in X:
            for y in deep_iter(x): yield y
    else:
        yield array(X)

def flat(X):
    '''Flatten and concatenate all the numpy arrays contained in 
    possibly nested dictonaries and lists.'''
    return concatenate( [x.flatten() for x in deep_iter(X)] )

import copy
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


def zeros_like(X):
    if isinstance(X,type(array([]))):
        return X * 0.
    elif isinstance(X,type(dict())):
        iterset = sorted(X.items())
        result  = {}
    elif isinstance(X,type([])):
        iterset = enumerate(X)
        result  = [None for i in len(X)]
    else:
        raise TypeError('zeros_like expects numpy ndarray, list or dict')
    for key,x in iterset:
        rec  = zeros_like(x)
        result[key] = rec
    return result


def unflat(template,X):
    '''Populate all the numpy arrays contained in 
    possibly nested dictonaries and lists of template, taking the 
    values from flat vector X of appropriate length: this is the 
    inverse operation of flat(X).'''
    if isinstance(X,type(array([]))):
        return __unflat(template,X)[0]
    else:
        return X


from inspect   import getargspec
from functools import partial
import copy
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
        def c(**dummy):
            new_kwargs = copy.deepcopy(dummy)
#            new_kwargs = dummy
            new_kwargs.update((reparam(**dummy)))
            return func(**new_kwargs)
    keydict   = getkwargs( reparam )
#    arguments,_,_,defaults = getargspec(func)
    func_args = getkwargs(func)
    if isinstance(output,type([])): output = output[0]
    for name,value in func_args.items(): #zip(arguments,defaults):
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
        return dict( (name,__reparameterize(f,reparam,reducer,zero)) 
                     for name,f in funcs.items() )
    else:
        return __reparameterize(funcs,reparam)

def _sum_objectives( objectives, attribute, x ):
    return sum([getattr(o,attribute)(x) for o in objectives])

class Sum_objectives( object ):
    def __init__( self, objectives, attributes=['f'] ):
        self.objectives = objectives
        self.attributes = attributes
        for fname in attributes:
            setattr(self,fname,partial(_sum_objectives, objectives, fname))
            setattr(getattr(self,fname),'__name__',fname)