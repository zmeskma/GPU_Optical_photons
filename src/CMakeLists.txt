# Host library: configuration, output and the CPU reference backend.
add_library(optphot_core STATIC
  core/compare.cpp
  core/config.cpp
  core/io.cpp
  cpu/cpu_backend.cpp)
target_link_libraries(optphot_core PUBLIC optphot_physics PRIVATE optphot_warnings)

if(WITH_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(optphot_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(optphot_core PUBLIC OPTPHOT_HAVE_OPENMP)
    message(STATUS "optphot: OpenMP found, multi-threaded CPU backend enabled")
  else()
    message(STATUS "optphot: OpenMP not found, CPU backend is single-threaded")
  endif()
endif()

# Umbrella target used by the applications and tests.
add_library(optphot INTERFACE)
target_link_libraries(optphot INTERFACE optphot_core)

if(WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(optphot_gpu STATIC gpu/gpu_backend.cu)
  # Static CUDA runtime: executables do not need cudart64_12.dll / libcudart.so
  # at run time (otherwise a missing DLL makes ctest silently drop the GPU tests).
  target_link_libraries(optphot_gpu PUBLIC optphot_physics CUDA::cudart_static)
  target_compile_definitions(optphot_gpu PUBLIC OPTPHOT_HAVE_CUDA)
  # -lineinfo: maps SASS back to source lines in Nsight Compute; no runtime cost.
  target_compile_options(optphot_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(NOT OPTPHOT_CUDA_FMAD)
    target_compile_options(optphot_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_compile_definitions(optphot_gpu PUBLIC OPTPHOT_NO_FMAD)
  endif()
  target_link_libraries(optphot INTERFACE optphot_gpu)
endif()
