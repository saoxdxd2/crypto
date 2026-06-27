cmake_minimum_required(VERSION 3.20)
project(crypto_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC Hard Optimization Flags for AVX2/AVX-512
if(MSVC)
    add_compile_options(
        /O2            # Maximize speed
        /Ob2           # Aggressive inline
        /Oi            # Intrinsic functions
        /Ot            # Favor fast code
        /arch:AVX2     # Enforce AVX2 (can be changed to AVX512 if hardware supports)
        /fp:fast       # Fast floating point
        /GS-           # Disable buffer security check (performance critical path)
        /GL            # Whole program optimization
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
endif()

# Build shared library to be called by Python via ctypes/cffi or pybind11
add_library(crypto_native SHARED math_engine.cpp)

# Output to Python accessible directory
set_target_properties(crypto_native PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../crypto_research"
)
