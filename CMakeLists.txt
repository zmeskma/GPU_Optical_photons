cmake_minimum_required(VERSION 3.18)

project(optphot VERSION 0.1.0
        DESCRIPTION "Toy optical-photon transport in a scintillator: CPU reference + CUDA"
        LANGUAGES CXX)

option(WITH_CUDA "Build the CUDA implementation (requires nvcc)" OFF)
option(WITH_OPENMP "Use OpenMP for the multi-threaded CPU backend if available" ON)
option(OPTPHOT_BUILD_TESTS "Build unit and validation tests" ON)
option(OPTPHOT_CUDA_FMAD
       "Let nvcc contract a*b+c into fused multiply-add (nvcc's default). Turn OFF to study CPU/GPU bitwise agreement."
       ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Floating-point policy -------------------------------------------------
# No fast-math anywhere. On the host we also forbid FMA contraction so that
# host results do not depend on whether the target CPU happens to have FMA
# instructions (e.g. -march=native). See docs/LEARNING_NOTES.md and the
# README section on CPU/GPU reproducibility.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
elseif(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
endif()

# Warnings for our own targets only (not for fetched dependencies).
add_library(optphot_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(optphot_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow>)
elseif(MSVC)
  target_compile_options(optphot_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
endif()

# --- CUDA ------------------------------------------------------------------
if(WITH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 75 = Turing (T4 on Colab, GTX 16xx / RTX 20xx). Override e.g. with
    # -DCMAKE_CUDA_ARCHITECTURES="75;80;86" or "native" (CMake >= 3.24).
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  message(STATUS "optphot: CUDA enabled, architectures = ${CMAKE_CUDA_ARCHITECTURES}, fmad = ${OPTPHOT_CUDA_FMAD}")
endif()

# --- Shared physics (header-only, host + device) ----------------------------
add_library(optphot_physics INTERFACE)
target_include_directories(optphot_physics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(optphot_physics INTERFACE cxx_std_17)

if(OPTPHOT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
