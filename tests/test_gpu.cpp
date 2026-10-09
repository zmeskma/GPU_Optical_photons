// SPDX-License-Identifier: MIT
//
// GPU backend tests (built only with WITH_CUDA=ON; skipped at run time if no
// CUDA device is present). The GPU runs the same transport_photon() as the
// CPU, so it must pass the same analytic validation, agree with the CPU
// photon by photon (up to documented floating-point effects) and give
// statistically compatible results for independent seeds.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "optphot/analytic.hpp"
#include "optphot/backend.hpp"
#include "optphot/compare.hpp"

using namespace optphot;

namespace {

SimParams black_box() {
  SimParams p;
  p.surface = Surface::Black;
  p.n_det = p.n_scint;
  p.abs_length = kInfinity;
  p.scat_length = kInfinity;
  p.tau = 0.0f;
  p.src_x = 10;
  p.src_y = -5;
  p.src_z = 3;
  return p;
}

double binomial_pull(uint64_t k, uint64_t n, double p) {
  return (static_cast<double>(k) - n * p) / std::sqrt(n * p * (1.0 - p));
}

uint64_t count(const PhotonRecords& r, Fate f) {
  uint64_t c = 0;
  for (uint8_t v : r.fate) c += (v == static_cast<uint8_t>(f));
  return c;
}

#define REQUIRE_GPU() \
  if (!gpu_available()) SKIP("no CUDA device")

}  // namespace

TEST_CASE("GPU: detection fraction equals solid angle / 4pi", "[gpu]") {
  REQUIRE_GPU();
  const SimParams p = black_box();
  const uint64_t n = 4000000;
  PhotonRecords rec;
  run_gpu_records(p, n, rec);
  const double expected = analytic::face_solid_angle_fraction(p, kReadoutFace);
  const double pull = binomial_pull(count(rec, Fate::Detected), n, expected);
  std::printf("[gpu] solid angle: MC %.6f analytic %.6f pull %.2f\n",
              double(count(rec, Fate::Detected)) / n, expected, pull);
  CHECK(std::fabs(pull) < 4.0);
}

TEST_CASE("GPU: bulk absorption against quadrature", "[gpu]") {
  REQUIRE_GPU();
  SimParams p = black_box();
  p.abs_length = 100.0f;
  const uint64_t n = 4000000;
  PhotonRecords rec;
  run_gpu_records(p, n, rec);
  const double expected = analytic::face_survival_fraction(p, kReadoutFace, p.abs_length);
  const double pull = binomial_pull(count(rec, Fate::Detected), n, expected);
  std::printf("[gpu] absorption L=100: MC %.6f quadrature %.6f pull %.2f\n",
              double(count(rec, Fate::Detected)) / n, expected, pull);
  CHECK(std::fabs(pull) < 4.0);
}

TEST_CASE("GPU vs CPU: photon-by-photon agreement with the same seed", "[gpu]") {
  REQUIRE_GPU();
  // A configuration exercising every process: polished walls (Fresnel, TIR),
  // Rayleigh scattering, absorption, volume source, emission time.
  SimParams p;
  p.scat_length = 300.0f;
  p.source = SourceType::UniformVolume;
  const uint64_t n = 1000000;
  PhotonRecords cpu, gpu;
  run_cpu_records(p, n, cpu, 0);
  run_gpu_records(p, n, gpu);
  const HistoryAgreement h = compare_histories(cpu, gpu);
  std::printf("[gpu] same seed: bitwise %.6f, same discrete history %.6f, fate differs %llu\n",
              double(h.bitwise_identical) / n, double(h.same_discrete) / n,
              static_cast<unsigned long long>(h.fate_differs));
#if defined(OPTPHOT_PORTABLE_MATH) && defined(OPTPHOT_NO_FMAD)
  // Only correctly rounded operations on both sides: identical to the last bit.
  CHECK(h.bitwise_identical == n);
#else
  // Math-library and FMA differences change the last bits of almost every
  // multi-step history, but flip a discrete decision only rarely.
  CHECK(double(h.same_discrete) / n > 0.99);
#endif
}

TEST_CASE("GPU vs CPU: statistical agreement with independent seeds", "[gpu]") {
  REQUIRE_GPU();
  SimParams p;
  p.scat_length = 300.0f;
  const uint64_t n = 2000000;
  PhotonRecords cpu, gpu;
  run_cpu_records(p, n, cpu, 0);
  SimParams pg = p;
  pg.seed = p.seed + 1;
  run_gpu_records(pg, n, gpu);
  const DistributionComparison d = compare_distributions(cpu, gpu, p);
  std::printf(
      "[gpu] independent seeds: eff z = %.2f, time chi2 p = %.3f, KS p = %.3f, xy chi2 p = %.3f\n",
      d.eff_z, d.time_chi2.pvalue, d.time_ks.pvalue, d.xy_chi2.pvalue);
  CHECK(std::fabs(d.eff_z) < 4.0);
  CHECK(d.time_chi2.pvalue > 1e-4);
  CHECK(d.time_ks.pvalue > 1e-4);
  CHECK(d.xy_chi2.pvalue > 1e-4);
}

TEST_CASE("GPU: tally mode agrees with records and is reproducible", "[gpu]") {
  REQUIRE_GPU();
  SimParams p;
  const uint64_t n = 1000000;
  const TallyConfig c;
  PhotonRecords rec;
  run_gpu_records(p, n, rec);
  Tally t1, t2;
  run_gpu_tally(p, c, n, t1);
  run_gpu_tally(p, c, n, t2, 128);  // different block size -> different atomic order
  for (int f = 0; f < kNumFates; ++f) {
    CHECK(t1.fate_counts[f] == count(rec, static_cast<Fate>(f)));
  }
  // Integer atomics: bit-identical regardless of scheduling.
  CHECK(t1.fate_counts == t2.fate_counts);
  CHECK(t1.time_hist == t2.time_hist);
  CHECK(t1.xy_hist == t2.xy_hist);
  CHECK(t1.boundary_sum == t2.boundary_sum);

  // The GPU tally equals the CPU tally exactly when histories agree; at least
  // the totals must match.
  Tally tc;
  run_cpu_tally(p, c, n, tc, 0);
  CHECK(tc.total() == t1.total());
}

TEST_CASE("GPU: records do not depend on the chunk size or block size", "[gpu]") {
  REQUIRE_GPU();
  SimParams p;
  p.scat_length = 300.0f;
  const uint64_t n = 300000;
  PhotonRecords a, b;
  run_gpu_records(p, n, a, 256, 0, 1u << 24);
  run_gpu_records(p, n, b, 64, 0, 70001);  // odd chunk size: several partial chunks
  const HistoryAgreement h = compare_histories(a, b);
  CHECK(h.bitwise_identical == n);
}

TEST_CASE("GPU event-based: records identical to history-based for any k", "[gpu]") {
  REQUIRE_GPU();
  SimParams polished;  // default cube: long TIR-trapped histories
  polished.scat_length = 300.0f;
  SimParams lambertian;
  lambertian.surface = Surface::Lambertian;
  lambertian.reflectivity = 0.95f;
  lambertian.n_det = 1.5f;
  lambertian.source = SourceType::UniformVolume;
  SimParams capped = polished;  // exercises the max_steps limit
  capped.max_steps = 5;
  const uint64_t n = 100003;  // not a multiple of the block size
  for (const SimParams& p : {polished, lambertian, capped}) {
    PhotonRecords hist;
    run_gpu_records(p, n, hist);
    uint64_t steps = 0;
    for (std::size_t i = 0; i < n; ++i) {
      steps += hist.n_boundary[i] + hist.n_scatter[i] +
               (hist.fate[i] == static_cast<uint8_t>(Fate::AbsorbedBulk));
    }
    for (const uint32_t k : {1u, 3u, 100000u}) {
      PhotonRecords ev;
      EventStats s;
      run_gpu_records_event(p, n, ev, k, 64, 0, 30011, &s);  // several partial chunks
      CHECK(compare_histories(hist, ev).bitwise_identical == n);
      CHECK(s.useful_lane_steps == steps);  // every step counted exactly once
      CHECK(s.issued_lane_steps >= s.useful_lane_steps);
    }
  }
}

TEST_CASE("GPU event-based: one launch reproduces the history-based SIMT efficiency", "[gpu]") {
  REQUIRE_GPU();
  const SimParams p;
  const uint64_t n = 200000;
  PhotonRecords hist, ev;
  run_gpu_records(p, n, hist);
  EventStats s;
  run_gpu_records_event(p, n, ev, p.max_steps, 256, 0, n, &s);
  // With k >= max_steps there is a single launch whose warps hold photons
  // 32w .. 32w + 31, exactly as in the history-based kernel.
  uint64_t issued = 0;
  for (std::size_t w = 0; w < n; w += 32) {
    uint32_t longest = 0;
    for (std::size_t i = w; i < std::min<std::size_t>(n, w + 32); ++i) {
      longest = std::max(longest, hist.n_boundary[i] + hist.n_scatter[i] +
                                      (hist.fate[i] == static_cast<uint8_t>(Fate::AbsorbedBulk)));
    }
    issued += 32ull * longest;
  }
  CHECK(s.launches == 1);
  CHECK(s.issued_lane_steps == issued);
}
