import numpy as np
from distutils.core import setup 
from Cython.Distutils import build_ext 
from distutils.extension import Extension 

ext_modules = [
        Extension("plum.util.cprobs",["plum/util/cprobs.pyx"],include_dirs=[np.get_include()]),
        Extension("plum.util.cpruning",["plum/util/cpruning.pyx"],include_dirs=[np.get_include()]),
        Extension("plum.util.cfitting",["plum/util/cfitting.pyx"],include_dirs=[np.get_include()]),
        Extension("plum.util.ctree",["plum/util/ctree.pyx"],include_dirs=[np.get_include()])
]

setup(
        name='plum',
        description='Phylogenetic latent variable models (PLVM --> plum) for ancestral network reconstruction',
        version='0.1',
        author='Benjamin Liebeskind',
        packages=['plum','plum.models','plum.training','plum.util'],
        package_dir={'plum':'plum',
            'plum.models':'plum/models',
            'plum.training':'plum/training',
            'plum.util':'plum/util'},
        ext_modules = ext_modules,
        cmdclass = {'build_ext': build_ext},

)


