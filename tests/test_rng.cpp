// SPDX-License-Identifier: MIT
// Tests of the Philox4x32-10 generator and the per-photon stream.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "optphot/philox.hpp"
#include "optphot/stats.hpp"

using namespace optphot;

namespace {
void check_kat(U32x4 ctr, U32x2 key, U32x4 expected) {
  const U32x4 r = philox4x32_10(ctr, key);
  CHECK(r.x == expected.x);
  CHECK(r.y == expected.y);
  CHECK(r.z == expected.z);
  CHECK(r.w == expected.w);
}
}  // namespace

TEST_CASE("Philox4x32-10 reproduces the Random123 known-answer vectors", "[rng]") {
  check_kat({0u, 0u, 0u, 0u}, {0u, 0u}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u});
  check_kat({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu},
            {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu});
  check_kat({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u},
            {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u});
}

TEST_CASE("u32_to_uniform maps 32 bits exactly onto 0 < u <= 1", "[rng]") {
  CHECK(u32_to_uniform(0u) == 1.0f / 16777216.0f);
  CHECK(u32_to_uniform(0xffffffffu) == 1.0f);
  CHECK(u32_to_uniform(0x000000ffu) == u32_to_uniform(0u));  // low 8 bits are dropped
}

TEST_CASE("PhiloxStream is a pure function of (seed, photon id)", "[rng]") {
  PhiloxStream a(42u, 1000u), b(42u, 1000u);
  // Interleave another stream between draws: must not matter.
  PhiloxStream other(42u, 1001u);
  for (int i = 0; i < 37; ++i) {
    (void)other.next_u32();
    REQUIRE(a.next_u32() == b.next_u32());
  }
  CHECK(a.draws() == 37u);

  // The stream is exactly the concatenation of the Philox blocks.
  PhiloxStream c(0x0123456789abcdefull, 0xfedcba9876543210ull);
  const U32x2 key{0x89abcdefu, 0x01234567u};
  for (uint32_t block = 0; block < 3; ++block) {
    const U32x4 r = philox4x32_10({block, 0u, 0x76543210u, 0xfedcba98u}, key);
    CHECK(c.next_u32() == r.x);
    CHECK(c.next_u32() == r.y);
    CHECK(c.next_u32() == r.z);
    CHECK(c.next_u32() == r.w);
  }
}

TEST_CASE("Different photons and seeds give different streams", "[rng]") {
  PhiloxStream a(1u, 0u), b(1u, 1u), c(2u, 0u);
  int same_ab = 0, same_ac = 0;
  for (int i = 0; i < 1000; ++i) {
    const uint32_t x = a.next_u32();
    same_ab += (x == b.next_u32());
    same_ac += (x == c.next_u32());
  }
  CHECK(same_ab <= 1);
  CHECK(same_ac <= 1);
}

TEST_CASE("Uniform deviates are statistically uniform and uncorrelated", "[rng]") {
  // Draw the first 4 numbers of 250k consecutive photon streams: this is the
  // way the transport code uses the generator (many short streams).
  constexpr int kPhotons = 250000;
  constexpr int kBins = 100;
  std::vector<double> hist(kBins, 0.0);
  double sum = 0.0, sum_xy = 0.0;
  long n = 0;
  float prev_first = 0.5f;
  for (int i = 0; i < kPhotons; ++i) {
    PhiloxStream s(777u, static_cast<uint64_t>(i));
    const float first = s.uniform();
    for (float u : {first, s.uniform(), s.uniform(), s.uniform()}) {
      REQUIRE(u > 0.0f);
      REQUIRE(u <= 1.0f);
      sum += u;
      ++n;
      hist[std::min(kBins - 1, static_cast<int>(u * kBins))] += 1.0;
    }
    sum_xy += (first - 0.5) * (prev_first - 0.5);  // neighbouring photons
    prev_first = first;
  }
  const double mean = sum / n;
  CHECK(std::fabs(mean - 0.5) < 5.0 * std::sqrt(1.0 / 12.0 / n));

  // Correlation of first deviates of neighbouring photon streams.
  const double corr = (sum_xy / kPhotons) * 12.0;
  CHECK(std::fabs(corr) < 5.0 / std::sqrt(static_cast<double>(kPhotons)));

  const std::vector<double> expected(kBins, static_cast<double>(n) / kBins);
  const auto chi2 = stats::chi2_gof(hist, expected);
  INFO("chi2/ndf = " << chi2.chi2 << "/" << chi2.ndf << ", p = " << chi2.pvalue);
  CHECK(chi2.pvalue > 1e-4);
}

TEST_CASE("chi2 p-value helper matches known values", "[rng][stats]") {
  // Reference values from scipy.stats.chi2.sf
  CHECK_THAT(stats::chi2_pvalue(10.0, 10.0),
             Catch::Matchers::WithinRel(0.44049328506521257, 1e-10));
  CHECK_THAT(stats::chi2_pvalue(3.0, 1.0), Catch::Matchers::WithinRel(0.08326451666355039, 1e-10));
  CHECK_THAT(stats::chi2_pvalue(150.0, 100.0),
             Catch::Matchers::WithinRel(0.0009039320423540184, 1e-8));
}
