# setup.py

from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(ext_modules=cythonize(Extension(
    name = 'cluster_for_py',
    sources=["cluster_for_py.pyx", "cluster2d.cpp"],
    language='c++',
    include_dirs=['/home/tongyao.bai/Project/cnn_seg/extert_cpp'],
    library_dirs=['/usr/local/lib', '/usr/lib'],
    libraries=[],
    extra_compile_args=['-std=c++11'],
    extra_link_args=["-lglog"]
)))


#python setup.py build_ext --inplace
