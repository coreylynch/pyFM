from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [ Extension ( "seq_dataset" , [ "seq_dataset.pyx" ], include_dirs=[np.get_include()] , libraries=["m"])]

setup (
name = 'Demos' ,
cmdclass = { 'build_ext': build_ext } ,
ext_modules = ext_modules
)