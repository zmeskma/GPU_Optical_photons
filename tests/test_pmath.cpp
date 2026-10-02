// SPDX-License-Identifier: MIT
// Accuracy of the portable math functions (pmath.hpp) against the platform
// library evaluated in double precision.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

#include "optphot/philox.hpp"
#include "optphot/pmath.hpp"

using namespace optphot;

namespace {

// Distance in units of float ulp (at the magnitude of the reference value).
double ulp_error(float value, double reference) {
  const float ref_f = static_cast<float>(reference);
  const double ulp = std::nextafter(std::fabs(ref_f), INFINITY) - std::fabs(ref_f);
  return std::fabs(value - reference) / ulp;
}

}  // namespace

TEST_CASE("portable::log is accurate for 0 < x <= 1 and beyond", "[pmath]") {
  double worst = 0.0;
  // Every uniform deviate the generator can produce has the form k * 2^-24.
  for (uint32_t k = 1; k <= (1u << 24); k += 97) {
    const float u = k * (1.0f / 16777216.0f);
    if (u > 0.999f) continue;  // near 1 the absolute error matters, tested below
    worst = std::max(worst, ulp_error(portable::log(u), std::log(static_cast<double>(u))));
  }
  INFO("worst log error " << worst << " ulp");
  CHECK(worst < 4.0);
  CHECK(portable::log(1.0f) == 0.0f);
  for (float x : {0.9999999f, 0.99999f, 1.5f, 1000.0f, 1e-30f}) {
    CHECK(std::fabs(portable::log(x) - std::log(static_cast<double>(x))) <
          4e-7 * std::max(1.0, std::fabs(std::log(static_cast<double>(x)))));
  }
}

TEST_CASE("portable::sincos_turns is accurate over a full turn", "[pmath]") {
  const double two_pi = 6.283185307179586476925;
  double worst = 0.0;
  for (uint32_t k = 1; k <= (1u << 24); k += 101) {
    const float u = k * (1.0f / 16777216.0f);
    float s, c;
    portable::sincos_turns(u, s, c);
    // Absolute error relative to 1 (sin/cos values feed unit vectors).
    worst = std::max(worst, std::fabs(s - std::sin(two_pi * u)));
    worst = std::max(worst, std::fabs(c - std::cos(two_pi * u)));
  }
  INFO("worst sincos abs. error " << worst);
  CHECK(worst < 3e-7);
  float s, c;
  portable::sincos_turns(0.25f, s, c);
  CHECK(s == 1.0f);
  CHECK(c == 0.0f);
  portable::sincos_turns(1.0f, s, c);
  CHECK(s == 0.0f);
  CHECK(c == 1.0f);
}

TEST_CASE("portable::cbrt is accurate on the range used by Rayleigh sampling", "[pmath]") {
  double worst = 0.0;
  for (float x = 0.01f; x < 10.0f; x *= 1.0001f) {
    worst = std::max(worst, ulp_error(portable::cbrt(x), std::cbrt(static_cast<double>(x))));
  }
  INFO("worst cbrt error " << worst << " ulp");
  CHECK(worst < 2.0);
  CHECK(portable::cbrt(8.0f) == 2.0f);
  CHECK(portable::cbrt(1.0f) == 1.0f);
}
