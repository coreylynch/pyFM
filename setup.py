from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

setup(
	maintainer='Corey Lynch',
    name='pyfm',
    packages=find_packages(),
    url='https://github.com/coreylynch/pyFM',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pyfm_fast", ["pyfm_fast.pyx"],
    						 libraries=["m"],
    						 include_dirs=[numpy.get_include()])]
)
