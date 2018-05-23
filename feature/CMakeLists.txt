cmake_minimum_required(VERSION 3.10)
project(feature)

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

#inlcude json
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

link_directories(${PROJECT_BINARY_DIR}/libs)
#set lib output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
add_library(demo feature_generator.cpp )
link_libraries(demo)
add_executable(out main.cpp)

add_executable(feature main.cpp)
target_link_libraries(feature demo)