################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.1)
project(Tencers)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

################################################################################
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
    )

set(CMAKE_CXX_FLAGS_RELEASENATIVE        "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASENATIVE          "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE    "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELEASENATIVE
    CMAKE_C_FLAGS_RELEASENATIVE
    CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE
    CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE
    )

set(CMAKE_CXX_FLAGS_DEBUGNATIVE        "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_C_FLAGS_DEBUGNATIVE          "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE    "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEBUGNATIVE
    CMAKE_C_FLAGS_DEBUGNATIVE
    CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE
    )

# # Enable address sanitizer in debug mode.
# # WARNING: this causes JupyterLab/Python to crash when loading our module, at
# # least on macOS: care must be taken to preload the asan verison of libclang_rt,
# # which is tricky since lldb/JupyterLab strip the `DYLD_INSERT_LIBRARIES` variable...
# set (CMAKE_C_FLAGS_DEBUG      "${CMAKE_C_FLAGS_DEBUG}      -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG}    -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

# Color output
include(UseColors)

# Make sure warnings/errors are still colorized when using Ninja for building.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_definitions(-fdiagnostics-color=always)
endif()

# Make sure warnings/errors are still colorized when using Ninja for building.
add_definitions(-fdiagnostics-color=always)

# Export compile flags(used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Common libraries
################################################################################

if(NOT TARGET ElasticRods)
    add_subdirectory(${THIRD_PARTY_DIR}/ElasticRods)
endif()

# libigl
# option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
# option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
# option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)
# option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
# option(LIBIGL_USE_STATIC_LIBRARY     "Use static library" ON)

# find_package(LIBIGL QUIET)

################################################################################
# Tencers library
################################################################################

add_library(Tencers
    ./src/helpers.hh
    ./src/SlidingNode.hh
    ./src/SlidingNode.cc
    ./src/Spring.hh 
    ./src/Spring.cc
    ./src/Tencer.hh
    ./src/Tencer.cc
    ./src/weighted_target_curve_fitter.hh
    ./src/tencer_optimization.hh
    ./src/tencer_optimization.cc
    ./src/tencers_knitro_problem.hh
    ./src/symmetry_optimization.hh
    ./src/symmetry_optimization.cc 
    ./src/TencerEquilibriumProblem.hh
    ./src/DesignOptimization/design_optimization_terms.hh
    ./src/DesignOptimization/design_optimization_terms.cc 
    ./src/DesignOptimization/EquilibriumOptimizationObject.hh
    ./src/DesignOptimization/EquilibriumOptimizationObject.cc 
    ./src/DesignOptimization/OptimizationObject.hh
    ./src/DesignOptimization/OptimizationObjectWithLinearChangeOfVariable.hh
    ./src/DesignOptimization/OptimizationObjectWithLinearChangeOfVariable.cc
    ./src/DesignOptimization/knitro.hh
)
set_target_properties(Tencers PROPERTIES CXX_STANDARD 14)
set_target_properties(Tencers PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(Tencers PUBLIC ElasticRods MeshFEM igl::core)
target_include_directories(Tencers PUBLIC ./src)
target_include_directories(Tencers PUBLIC ${THIRD_PARTY_DIR})

################################################################################
# Third Party Libraries
################################################################################
# Knitro

option(DISABLE_Knitro  "Whether to disable the Knitro parts even if some version of Knitro is available" OFF)

if (NOT DISABLE_Knitro)
    find_package(Knitro QUIET)
    if(Knitro_FOUND)
        target_link_libraries(Tencers PUBLIC knitro::knitro)
    else()
        message(STATUS "Knitro not found; knitro solvers and dependent binaries disabled")
    endif()
else()
    message(STATUS "Knitro was manually disabled")
endif()

################################################################################
# Python Bindings
################################################################################
add_subdirectory(python_bindings)