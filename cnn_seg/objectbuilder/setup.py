# setup.py

from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(ext_modules=cythonize(Extension(
    name = 'Objectbuilder',
    sources=["cluster_for_py_.pyx", "cluster2d.cpp", "min_box.cpp", "types.cpp", "geometry_util.cpp"],
    language='c++',
    include_dirs=['/home/bai/Project/cnn_seg/objectbuilder', '/home/bai/Library/opencv/include/', '/usr/local/include/pcl-1.8', '/usr/include/eigen3'],
    library_dirs=['/home/bai/Library/opencv/build1/lib', '/usr/local/lib', '/usr/lib'],
    libraries=[],
    extra_compile_args=['-std=c++11', "-DPCL_NO_PRECOMPILE"],
    extra_link_args=["-lglog", "/usr/local/lib/libopencv_highgui.so", "/usr/local/lib/libpcl_kdtree.so", "/usr/local/lib/libpcl_registration.so", "/usr/local/lib/libpcl_search.so", "/usr/local/lib/libpcl_io.so", "/usr/local/lib/libpcl_keypoints.so", "/usr/local/lib/libpcl_common.so", "/usr/local/lib/libpcl_common.so", "/usr/local/lib/libpcl_surface.so"]
)))

