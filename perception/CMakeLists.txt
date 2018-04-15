
cmake_minimum_required (VERSION 2.8)
project (perception)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


#include Caffe library
set(Caffe_DIR /home/bai/caffe/cmake_build)
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS} /home/bai/caffe/cmake_build/install/include)
add_definitions(${Caffe_DEFINITIONS} -D CPU_ONLY=OFF) 



#inlcude PCl library
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add include dir, set under the root of this dir
include_directories(${PROJECT_SOURCE_DIR}/ /usr/local/lib /usr/lib)

#add lib dir
link_directories(${PROJECT_BINARY_DIR}/libs /home/bai/caffe/cmake_build/install/lib)

#inlcude json 
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})


#set lib output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)

#set excutable output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)



#add subdir lib
add_library(lib_demo feature_generator.cpp cnn_segmentation.cpp)
link_libraries(lib_demo)
add_executable(out run.cpp)

target_link_libraries(out ${Caffe_LIBRARIES} ${PCL_LIBRARIES} ${JSONCPP_LIBRARIES} lib_demo)
