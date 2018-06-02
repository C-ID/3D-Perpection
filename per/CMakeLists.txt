
cmake_minimum_required (VERSION 3.0)
set(OpenCV_DIR /home/bai/Library/opencv/build1)
project (per)

SET(CMAKE_C_COMPILER /usr/bin/g++)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package (Glog)
set(GLOG_LIBRARIES glog::glog)


#include Caffe library
set(Caffe_DIR /home/bai/Library/caffe/build1)
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS} /home/bai/Library/caffe/include /home/bai/Library/caffe/build1/lib)
add_definitions(${Caffe_DEFINITIONS} -D CPU_ONLY=ON)



#inlcude PCl library
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS} -DPCL_NO_PRECOMPILE)

#include opencv2
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} /usr/local/lib /home/bai/Library/opencv/build1/lib)


# add include dir, set under the root of this dir
include_directories(${PROJECT_SOURCE_DIR}/ /usr/local/lib /usr/lib)

#add lib dir
link_directories(${PROJECT_BINARY_DIR}/libs)

#inlcude json 
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

#set lib output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)

#set excutable output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)

#add subdir lib
add_library(lib_demo feature_generator.cpp cnn_segmentation.cpp min_box.cpp geometry_util.cpp fortracking.cpp)
link_libraries(lib_demo)
add_executable(out run.cpp)

link_directories(/home/bai/Library/caffe/build/lib)
target_link_libraries(out ${Caffe_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} lib_demo glog)
