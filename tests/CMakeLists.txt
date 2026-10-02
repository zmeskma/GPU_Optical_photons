include(FetchContent)
FetchContent_Declare(
  Catch2
  URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
  URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Pure unit tests of the shared header-only physics (host build of the same
# __host__ __device__ functions the GPU uses).
add_executable(unit_tests
  test_rng.cpp
  test_fresnel.cpp
  test_sampling.cpp
  test_geometry.cpp
  test_pmath.cpp)
target_link_libraries(unit_tests PRIVATE optphot_physics optphot_warnings Catch2::Catch2WithMain)
catch_discover_tests(unit_tests TEST_PREFIX "unit:")

add_executable(config_tests test_config.cpp)
target_link_libraries(config_tests PRIVATE optphot optphot_warnings Catch2::Catch2WithMain)
catch_discover_tests(config_tests TEST_PREFIX "config:")

# End-to-end transport validated against analytic results (CPU backend).
add_executable(validation_tests test_transport_cpu.cpp)
target_link_libraries(validation_tests PRIVATE optphot optphot_warnings Catch2::Catch2WithMain)
catch_discover_tests(validation_tests TEST_PREFIX "validation:")

if(WITH_CUDA)
  # GPU backend: analytic checks, CPU/GPU agreement, tally reproducibility.
  # Skipped at run time when no CUDA device is present.
  add_executable(gpu_tests test_gpu.cpp)
  target_link_libraries(gpu_tests PRIVATE optphot optphot_warnings Catch2::Catch2WithMain)
  catch_discover_tests(gpu_tests TEST_PREFIX "gpu:" DISCOVERY_MODE PRE_TEST)
endif()
