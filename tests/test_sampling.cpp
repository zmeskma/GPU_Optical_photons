// SPDX-License-Identifier: MIT
// Tests of the elementary sampling routines (isotropic, Rayleigh, Lambertian,
// exponential) against their analytic distributions.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "optphot/philox.hpp"
#include "optphot/sampling.hpp"
#include "optphot/stats.hpp"

using namespace optphot;
using Catch::Matchers::WithinAbs;

namespace {

constexpr int kN = 400000;
constexpr int kBins = 50;

// Histogram of x in [lo, hi) compared with the integral of `cdf` over each bin.
template <class Cdf>
stats::Chi2Result histogram_test(const std::vector<float>& xs, double lo, double hi, Cdf cdf) {
  std::vector<double> obs(kBins, 0.0), exp(kBins, 0.0);
  const double w = (hi - lo) / kBins;
  for (float x : xs) {
    if (x < lo || x > hi) continue;  // outside the histogrammed range
    const int b = std::clamp(static_cast<int>((x - lo) / w), 0, kBins - 1);
    obs[b] += 1.0;
  }
  for (int b = 0; b < kBins; ++b) {
    exp[b] = xs.size() * (cdf(lo + (b + 1) * w) - cdf(lo + b * w));
  }
  return stats::chi2_gof(obs, exp);
}

double mean(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += x;
  return s / v.size();
}

}  // namespace

TEST_CASE("Orthonormal basis is orthonormal, including near the poles", "[sampling]") {
  PhiloxStream rng(5u, 0u);
  std::vector<Vec3> normals = {{0, 0, 1},
                               {0, 0, -1},
                               {1, 0, 0},
                               {-1, 0, 0},
                               {0, 1, 0},
                               {0, -1, 0},
                               normalize(Vec3{1e-4f, 0, -1})};
  for (int i = 0; i < 1000; ++i) normals.push_back(sample_isotropic(rng.uniform(), rng.uniform()));
  for (const Vec3 n : normals) {
    Vec3 t1, t2;
    orthonormal_basis(n, t1, t2);
    CHECK_THAT(norm(t1), WithinAbs(1.0, 1e-6));
    CHECK_THAT(norm(t2), WithinAbs(1.0, 1e-6));
    CHECK_THAT(dot(t1, n), WithinAbs(0.0, 1e-6));
    CHECK_THAT(dot(t2, n), WithinAbs(0.0, 1e-6));
    CHECK_THAT(dot(t1, t2), WithinAbs(0.0, 1e-6));
  }
}

TEST_CASE("Isotropic directions: unit length, uniform cos(theta) and phi", "[sampling]") {
  PhiloxStream rng(6u, 0u);
  std::vector<float> cz, phi;
  double sx = 0, sy = 0, sz2 = 0;
  for (int i = 0; i < kN; ++i) {
    const Vec3 d = sample_isotropic(rng.uniform(), rng.uniform());
    REQUIRE_THAT(norm(d), WithinAbs(1.0, 1e-6));
    cz.push_back(d.z);
    phi.push_back(std::atan2(d.y, d.x));
    sx += d.x;
    sy += d.y;
    sz2 += d.z * d.z;
  }
  const double sigma = std::sqrt(1.0 / 3.0 / kN);
  CHECK(std::fabs(sx / kN) < 5 * sigma);
  CHECK(std::fabs(sy / kN) < 5 * sigma);
  CHECK_THAT(sz2 / kN, WithinAbs(1.0 / 3.0, 5 * std::sqrt(4.0 / 45.0 / kN)));
  CHECK(histogram_test(cz, -1, 1, [](double x) { return 0.5 * (x + 1); }).pvalue > 1e-4);
  const double pi = 3.14159265358979323846;
  CHECK(histogram_test(phi, -pi, pi, [pi](double x) { return (x + pi) / (2 * pi); }).pvalue > 1e-4);
}

TEST_CASE("Rayleigh cos(theta): exact inverse CDF of 3/8 (1 + mu^2)", "[sampling]") {
  // F(mu) = (mu^3 + 3 mu + 4) / 8 must invert sample_rayleigh_cos exactly.
  for (int i = 1; i <= 1000; ++i) {
    const float u = i / 1000.0f;
    const double mu = sample_rayleigh_cos(u);
    REQUIRE(mu >= -1.0);
    REQUIRE(mu <= 1.0);
    CHECK_THAT((mu * mu * mu + 3 * mu + 4) / 8, WithinAbs(u, 2e-6));
  }
  CHECK_THAT(sample_rayleigh_cos(0.5f), WithinAbs(0.0, 1e-7));
  CHECK_THAT(sample_rayleigh_cos(1.0f), WithinAbs(1.0, 1e-6));
}

TEST_CASE("Rayleigh scattering: phase function, moments and geometry", "[sampling]") {
  PhiloxStream rng(7u, 0u);
  const Vec3 in = normalize(Vec3{0.3f, -0.5f, 0.8f});
  std::vector<float> mus;
  for (int i = 0; i < kN; ++i) {
    const Vec3 out = sample_rayleigh_direction(in, rng.uniform(), rng.uniform());
    REQUIRE_THAT(norm(out), WithinAbs(1.0, 1e-6));
    mus.push_back(dot(in, out));
  }
  // <mu> = 0, <mu^2> = 2/5 for p(mu) = 3/8 (1 + mu^2)
  double m2 = 0;
  for (float m : mus) m2 += m * m;
  CHECK(std::fabs(mean(mus)) < 5 * std::sqrt(0.4 / kN));
  CHECK_THAT(m2 / kN, WithinAbs(0.4, 5 * std::sqrt(0.1 / kN)));
  const auto chi2 =
      histogram_test(mus, -1, 1, [](double x) { return (x * x * x + 3 * x + 4) / 8; });
  INFO("chi2/ndf = " << chi2.chi2 << "/" << chi2.ndf);
  CHECK(chi2.pvalue > 1e-4);
}

TEST_CASE("Lambertian reflection: cosine-weighted hemisphere", "[sampling]") {
  PhiloxStream rng(8u, 0u);
  for (const Vec3 n :
       {Vec3{0, 0, 1}, Vec3{0, 0, -1}, Vec3{1, 0, 0}, Vec3{0, -1, 0}, normalize(Vec3{1, 2, 3})}) {
    std::vector<float> cosines;
    for (int i = 0; i < kN / 4; ++i) {
      const Vec3 d = sample_lambertian(n, rng.uniform(), rng.uniform());
      REQUIRE_THAT(norm(d), WithinAbs(1.0, 1e-6));
      REQUIRE(dot(d, n) >= 0.0f);
      cosines.push_back(dot(d, n));
    }
    // p(c) = 2c on [0, 1]: CDF c^2, mean 2/3, variance 1/18.
    const double nn = cosines.size();
    CHECK_THAT(mean(cosines), WithinAbs(2.0 / 3.0, 5 * std::sqrt(1.0 / 18.0 / nn)));
    CHECK(histogram_test(cosines, 0, 1, [](double c) { return c * c; }).pvalue > 1e-4);
  }
}

TEST_CASE("Exponential path lengths and emission times", "[sampling]") {
  PhiloxStream rng(9u, 0u);
  std::vector<float> s, t;
  for (int i = 0; i < kN; ++i) {
    s.push_back(sample_exponential(250.0f, rng.uniform()));
    t.push_back(sample_emission_time(4.0f, rng.uniform()));
  }
  CHECK_THAT(mean(s), WithinAbs(250.0, 5 * 250.0 / std::sqrt(kN)));
  CHECK_THAT(mean(t), WithinAbs(4.0, 5 * 4.0 / std::sqrt(kN)));
  CHECK(histogram_test(s, 0, 1000, [](double x) { return 1 - std::exp(-x / 250.0); }).pvalue >
        1e-4);

  // Disabled process / prompt emission.
  CHECK(is_inf(sample_exponential(kInfinity, 0.5f)));
  CHECK(is_inf(sample_exponential(kInfinity, 1.0f)));  // no 0 * inf = NaN
  CHECK(sample_exponential(250.0f, 1.0f) == 0.0f);     // u = 1 is the largest deviate
  CHECK(sample_emission_time(0.0f, 0.3f) == 0.0f);
}

TEST_CASE("Uniform-in-box source", "[sampling]") {
  PhiloxStream rng(10u, 0u);
  double sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < kN; ++i) {
    const Vec3 p = sample_in_box(10.f, 20.f, 30.f, rng.uniform(), rng.uniform(), rng.uniform());
    REQUIRE(std::fabs(p.x) <= 10.f);
    REQUIRE(std::fabs(p.y) <= 20.f);
    REQUIRE(std::fabs(p.z) <= 30.f);
    sx += p.x;
    sy += p.y;
    sz += p.z;
  }
  CHECK(std::fabs(sx / kN) < 5 * 10.0 / std::sqrt(3.0 * kN));
  CHECK(std::fabs(sy / kN) < 5 * 20.0 / std::sqrt(3.0 * kN));
  CHECK(std::fabs(sz / kN) < 5 * 30.0 / std::sqrt(3.0 * kN));
}
