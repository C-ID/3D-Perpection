# setup.py

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(Extension(
    name = 'feature_generator',
    sources=["feature_generator.cpp"],
    language='c++',
    include_dirs=['/home/bai/Project/3D-Perpection/per', '/home/bai/Library/caffe/include', '/usr/local/include/pcl-1.8', '/usr/include/eigen3'],
    library_dirs=['/home/bai/Library/caffe/build1/lib', '/usr/local/lib', '/usr/lib'],
    libraries=['caffe'],
    extra_compile_args=['-std=c++11', '-DCPU_ONLY=ON'],
    extra_link_args=[]
)))