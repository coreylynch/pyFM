#from distutils.core import setup
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import Configuration
import numpy

setup(
	maintainer='Corey Lynch',
    name='pyfm',
    packages=['pyfm'],
    package_dir={'':'.'},
    url='https://github.com/coreylynch/pyFM',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pyfm_fast", ["pyfm_fast.pyx"],
    						 libraries=["m"],
    						 include_dirs=[numpy.get_include()])]
)