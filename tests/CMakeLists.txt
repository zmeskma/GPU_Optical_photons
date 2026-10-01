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
  test_geometry.cpp)
target_link_libraries(unit_tests PRIVATE optphot_physics optphot_warnings Catch2::Catch2WithMain)
catch_discover_tests(unit_tests TEST_PREFIX "unit:")
