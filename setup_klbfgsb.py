#!/usr/bin/env python

from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('optimize',parent_package, top_path)
	
    lapack = get_info('lapack_opt')
    sources=['klbfgsb.pyf','routines.f']
    config.add_extension('_klbfgsb',
                         sources=[join('klbfgsb',x) for x in sources],
                         **lapack)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
