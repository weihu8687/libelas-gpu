cmake_minimum_required(VERSION 2.8.8)

# Project name
project(libelas-gpu)

# Include our cmake files
# SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# Include libraries
find_package(CUDA)
# find_package(Eigen3 REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system serialization system filesystem thread program_options date_time regex timer chrono)
# find_package(OpenCV 3 REQUIRED core imgcodecs videoio ximgproc)
find_package(OpenCV 3.4.2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system thread regex chrono date_time atomic)

add_library(boost INTERFACE)
target_include_directories(boost INTERFACE ${Boost_INCLUDE_DIR})
target_link_libraries(boost INTERFACE ${Boost_LIBRARIES} pthread)
target_compile_options(boost INTERFACE "-Wno-deprecated-declarations;")
message(STATUS "Boost library status:")
message(STATUS "    Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
message(STATUS "    Boost_LIBRARIES: ${Boost_LIBRARIES}")


add_library(opencv INTERFACE)
target_include_directories(opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
list(REMOVE_ITEM OpenCV_LIBS opencv_dnn) # remove caffe libs
list(REMOVE_ITEM OpenCV_LIBS opencv_hdf) # remove hdf5 libs
target_link_libraries(opencv INTERFACE ${OpenCV_LIBS})
message(STATUS "OpenCV library status:")
message(STATUS "    OPENCV_ROOT_DIR: ${OPENCV_ROOT_DIR}")
message(STATUS "    OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "    OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")



# Try to compile with c++11
# http://stackoverflow.com/a/25836953
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

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Enable debug flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3  -Wall")

# Enable SSE3 cpu commands
# If we are on ARM we need to find an alternative
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")

# Set our nvcc flags
# http://stackoverflow.com/a/13244930
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_61 -gencode arch=compute_61,code=sm_61)

# work around to fix the nvcc build cannot find header file problem
# Include our header files
include_directories(CPU
        GPU
        GPU_test
        ${OpenCV_INCLUDE_DIRS})

# Include all source files in each sub-directory
# file(GLOB_RECURSE sources_cpu CPU/*.cpp)
# file(GLOB_RECURSE sources_gpu GPU/*.cu)
# file(GLOB_RECURSE sources_cpu GPU_test/*.cpp)

# Set files the build
set(sources_cpu
  CPU/descriptor.cpp
  CPU/elas.cpp
  CPU/filter.cpp
  CPU/matrix.cpp
  CPU/triangle.cpp
)

# Set files the build
set(sources_gpu
  GPU/elas_gpu.cu
)

# CPU binary
add_executable(libelas_cpu main_cpu.cpp ${sources_cpu})

# GPU binary, convert the gpu code to cpu code, then build
#cuda_compile(sources_gpu_built main_gpu.cu ${sources_gpu})
#cuda_add_executable(libelas_gpu ${sources_gpu_built} ${sources_cpu})

cuda_add_library(
        libelas_gpu STATIC
        ${sources_gpu}
        )

target_include_directories(libelas_gpu PUBLIC
        ${CUDA_INCLUDE_DIRS}
        )

target_link_libraries(libelas_gpu
        ${CUDA_TARGET}
        )

add_executable(elas_gpu
        main_gpu.cpp
        ${sources_cpu})

target_include_directories(elas_gpu PUBLIC
        ${CUDA_INCLUDE_DIRS}
        )

target_link_libraries(elas_gpu
        libelas_gpu
        opencv
        boost
        ${CUDA_TARGET}
        )

# Testing binary
add_executable(libelas_test main_test.cpp ${sources_cpu})
