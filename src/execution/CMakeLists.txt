cmake_minimum_required(VERSION 3.15)
project(ThreeBrainEngine CXX)

set(CMAKE_CXX_STANDARD 17)

# We expect ONNXRUNTIME_ROOT to be passed or set in the environment
# e.g. -DONNXRUNTIME_ROOT="C:/onnxruntime"
if (NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT is not defined. Please download the ONNXRuntime C++ Release for Windows and set this variable.")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_library(inference_core SHARED inference_core.cpp)

# Link against the onnxruntime shared library
target_link_libraries(inference_core PRIVATE onnxruntime)
