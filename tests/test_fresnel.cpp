// SPDX-License-Identifier: MIT
// Tests of Fresnel coefficients and reflection / refraction directions.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "optphot/fresnel.hpp"
#include "optphot/philox.hpp"
#include "optphot/sampling.hpp"

using namespace optphot;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kPiD = 3.14159265358979323846;

// Independent double-precision reference using the angle form of the Fresnel
// equations: Rs = sin^2(ti - tt) / sin^2(ti + tt), Rp = tan^2(ti - tt) / tan^2(ti + tt).
double reference_R(double n1, double n2, double theta_i) {
  const double s = n1 / n2 * std::sin(theta_i);
  if (s >= 1.0) return 1.0;
  const double theta_t = std::asin(s);
  const double rs = std::sin(theta_i - theta_t) / std::sin(theta_i + theta_t);
  const double rp = std::tan(theta_i - theta_t) / std::tan(theta_i + theta_t);
  return 0.5 * (rs * rs + rp * rp);
}

float sin_angle(Vec3 a, Vec3 b) { return norm(cross(a, b)); }

}  // namespace

TEST_CASE("Fresnel: normal incidence gives ((n1-n2)/(n1+n2))^2", "[fresnel]") {
  for (auto [n1, n2] : {std::pair{1.63f, 1.0f}, {1.0f, 1.63f}, {1.63f, 1.5f}, {1.33f, 2.4f}}) {
    const float expected = ((n1 - n2) / (n1 + n2)) * ((n1 - n2) / (n1 + n2));
    const FresnelResult f = fresnel_unpolarized(n1, n2, 1.0f);
    CHECK_THAT(f.reflectance, WithinAbs(expected, 1e-7));
    CHECK(f.cos_t == 1.0f);
  }
  // Stilbene / air: R = (0.63/2.63)^2 = 0.05738...
  CHECK_THAT(fresnel_unpolarized(1.63f, 1.0f, 1.0f).reflectance, WithinAbs(0.0573812, 1e-6));
}

TEST_CASE("Fresnel: index-matched interface never reflects", "[fresnel]") {
  // Includes grazing angles where 1 - cos^2 rounds to 1 in float.
  for (float c : {0.0f, 1e-5f, 1e-4f, 1e-3f, 0.05f, 0.3f, 0.7f, 1.0f}) {
    const FresnelResult f = fresnel_unpolarized(1.63f, 1.63f, c);
    CHECK(f.reflectance == 0.0f);
    CHECK(f.cos_t == c);
  }
}

TEST_CASE("Fresnel: total internal reflection beyond the critical angle", "[fresnel]") {
  const double n1 = 1.63, n2 = 1.0;
  const double theta_c = std::asin(n2 / n1);  // 37.84 deg

  // Well beyond and just beyond the critical angle: R == 1 exactly.
  for (double dtheta : {1e-3, 1e-2, 0.2, 0.5}) {
    const auto f = fresnel_unpolarized(1.63f, 1.0f, static_cast<float>(std::cos(theta_c + dtheta)));
    CHECK(f.reflectance == 1.0f);
    CHECK(f.cos_t == 0.0f);
  }
  // Just below the critical angle: R < 1 but approaching 1 continuously.
  const auto just_below = fresnel_unpolarized(1.63f, 1.0f, static_cast<float>(std::cos(theta_c - 1e-4)));
  CHECK(just_below.reflectance < 1.0f);
  CHECK(just_below.reflectance > 0.9f);
  const auto below = fresnel_unpolarized(1.63f, 1.0f, static_cast<float>(std::cos(theta_c - 1e-2)));
  CHECK(below.reflectance < just_below.reflectance);

  // Grazing incidence from the optically thinner side reflects everything too.
  CHECK_THAT(fresnel_unpolarized(1.0f, 1.63f, 0.0f).reflectance, WithinAbs(1.0, 1e-6));
}

TEST_CASE("Fresnel: Brewster angle has Rp = 0, so R = Rs / 2", "[fresnel]") {
  const double n1 = 1.0, n2 = 1.63;
  const double theta_b = std::atan(n2 / n1);
  const double theta_t = std::asin(n1 / n2 * std::sin(theta_b));
  const double rs = std::sin(theta_b - theta_t) / std::sin(theta_b + theta_t);
  const auto f = fresnel_unpolarized(1.0f, 1.63f, static_cast<float>(std::cos(theta_b)));
  CHECK_THAT(f.reflectance, WithinAbs(0.5 * rs * rs, 1e-6));
}

TEST_CASE("Fresnel: agrees with the independent angle-form reference", "[fresnel]") {
  for (auto [n1, n2] : {std::pair{1.63, 1.0}, {1.0, 1.63}, {1.63, 1.5}, {1.5, 1.63}}) {
    for (double deg = 0.5; deg < 90.0; deg += 0.5) {
      const double th = deg * kPiD / 180.0;
      const auto f = fresnel_unpolarized(static_cast<float>(n1), static_cast<float>(n2),
                                         static_cast<float>(std::cos(th)));
      INFO("n1=" << n1 << " n2=" << n2 << " theta=" << deg);
      CHECK_THAT(f.reflectance, WithinAbs(reference_R(n1, n2, th), 2e-5));
    }
  }
}

TEST_CASE("Reflection: unit length, angle preserved, tangential part unchanged", "[fresnel]") {
  PhiloxStream rng(1u, 2u);
  for (int i = 0; i < 1000; ++i) {
    const Vec3 n = sample_isotropic(rng.uniform(), rng.uniform());
    Vec3 d = sample_isotropic(rng.uniform(), rng.uniform());
    if (dot(d, n) < 0.0f) d = -d;  // make it incident on the surface
    const Vec3 r = reflect(d, n);
    CHECK_THAT(norm(r), WithinAbs(1.0, 1e-6));
    CHECK_THAT(dot(r, n), WithinAbs(-dot(d, n), 1e-6));
    const Vec3 dt = d - dot(d, n) * n, rt = r - dot(r, n) * n;
    CHECK_THAT(norm(dt - rt), WithinAbs(0.0, 1e-6));
  }
}

TEST_CASE("Refraction: unit length, Snell's law, coplanarity, reversibility", "[fresnel]") {
  PhiloxStream rng(3u, 4u);
  for (auto [n1, n2] : {std::pair{1.63f, 1.0f}, {1.0f, 1.63f}, {1.63f, 1.5f}}) {
    int refracted = 0;
    for (int i = 0; i < 2000; ++i) {
      const Vec3 n = sample_isotropic(rng.uniform(), rng.uniform());
      Vec3 d = sample_isotropic(rng.uniform(), rng.uniform());
      if (dot(d, n) < 0.0f) d = -d;
      const float cos_i = dot(d, n);
      const FresnelResult f = fresnel_unpolarized(n1, n2, cos_i);
      if (f.reflectance == 1.0f) continue;  // TIR
      ++refracted;
      const Vec3 t = refract(d, n, n1 / n2, cos_i, f.cos_t);
      CHECK_THAT(norm(t), WithinAbs(1.0, 1e-6));
      CHECK(dot(t, n) > 0.0f);                                     // goes into medium 2
      CHECK_THAT(dot(t, n), WithinAbs(f.cos_t, 2e-6));             // the angle is cos_t
      CHECK_THAT(n1 * sin_angle(d, n), WithinAbs(n2 * sin_angle(t, n), 3e-6));  // Snell
      CHECK_THAT(dot(t, cross(d, n)), WithinAbs(0.0, 2e-6));      // plane of incidence

      // Going back through the interface recovers the original direction.
      const Vec3 back = refract(-t, -n, n2 / n1, dot(t, n), cos_i);
      CHECK_THAT(norm(back + d), WithinAbs(0.0, 2e-5));
    }
    CHECK(refracted > 300);
  }
}
