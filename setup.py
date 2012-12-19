from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import glob
import numpy as np

sources =['minifm/_libfm.pyx']

setup(
    name='libfm',
    version='0.1',
    description='Context aware matrix factorization with libfm',
    author='Corey Lynch',
    author_email='coreylynch9@gmail.com',
    url='',
    packages=['minifm'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('minifm._libfm',
                             sources=sources,
                             language='c++',
                             include_dirs=[np.get_include()]),
                   Extension('minifm._svmlight_format',
                             sources='minifm/_svmlight_format.pyx',
                             language='c',
                             include_dirs=[np.get_include()]), 

                    ],
)
