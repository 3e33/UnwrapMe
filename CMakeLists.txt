# Copyright (C) 2023-2026 Elias V.
# SPDX-License-Identifier: GPL-3.0-only

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_ASAN "Compile with AddressSanitizer" OFF)

if(WIN32)
  set(DEP_DIR "${CMAKE_SOURCE_DIR}/dep/windows")
elseif(UNIX AND NOT APPLE)
  set(DEP_DIR "${CMAKE_SOURCE_DIR}/dep/linux")
else()
  set(DEP_DIR "${CMAKE_SOURCE_DIR}/dep/unknown")
endif()
list(APPEND CMAKE_PREFIX_PATH "${DEP_DIR}")
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/out")

project(fastcore)

find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development.Module REQUIRED)

set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "${DEP_DIR}/OpenBLAS")
set(BLA_VENDOR OpenBLAS)
set(BLA_PKGCONFIG_BLAS "openblas")
set(BLA_PKGCONFIG_LAPACK "openblas")

# Make sure to link the standard maths libary when
# `find_package` runs its `check_function_exists` check
if(UNIX AND NOT APPLE)
  set(CMAKE_REQUIRED_LINK_OPTIONS "-Wl,--no-as-needed;-lm")
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
if(UNIX AND NOT APPLE)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)
endif()

set(XTENSOR_USE_XSIMD ON)
set(XTENSOR_USE_OPENMP ON)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(xtensor-blas REQUIRED)

set(SOURCES
src/fastcore.cpp
src/overlap_remove.cpp
src/bConnector.cpp
src/extraCuts.cpp
src/aStar.cpp
src/common.cpp
src/curvature.cpp
src/regionClustering.cpp
src/optim.cpp
src/autoSeam.cpp
src/autoSplit.cpp
src/edgeWalk.cpp
src/fitting.cpp
)

pybind11_add_module(fastcore ${SOURCES})

target_include_directories(fastcore PRIVATE "${CMAKE_SOURCE_DIR}/packages/boost_1_81_0")

if(WIN32)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
endif()

if(UNIX AND NOT APPLE)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

target_link_libraries(fastcore PUBLIC
pybind11::module
xtensor
xtensor::optimize
xtensor-python
xtensor-blas
BLAS::BLAS)


if(WIN32)
  target_compile_options(fastcore PRIVATE /EHsc /MT /bigobj -openmp:experimental)
  target_link_options(fastcore PRIVATE /OPT:NOREF)

  if(ENABLE_ASAN)
    target_compile_options(fastcore PRIVATE /fsanitize=address)
    target_link_options(fastcore PRIVATE /INCREMENTAL:NO)
  endif()
endif()

if(UNIX AND NOT APPLE)
  target_compile_options(fastcore PRIVATE -march=x86-64-v2 -fopenmp)
  target_link_options(fastcore PRIVATE -fopenmp)
endif()

set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)

set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)


#######################################

set(CMAKE_INSTALL_MODE COPY)
set(CMAKE_INSTALL_PREFIX "${OUTPUT_DIR}")
set(INSTALL_FOLDER UnwrapMe)

install(TARGETS fastcore DESTINATION ${INSTALL_FOLDER})
