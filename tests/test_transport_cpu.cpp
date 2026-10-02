// SPDX-License-Identifier: MIT
//
// Validation of the full photon transport (CPU backend) against analytic and
// quadrature results. Each test uses a fixed seed, so it is deterministic;
// the statistical thresholds (|pull| < 4, p > 1e-4) are chosen such that a
// correct implementation passes for essentially any seed.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include "optphot/analytic.hpp"
#include "optphot/backend.hpp"
#include "optphot/compare.hpp"
#include "optphot/stats.hpp"

using namespace optphot;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr int kAllThreads = 0;

// INFO (shown on failure) + an always-printed line, so that `ctest -V` or
// running the executable directly shows the actual pulls / p-values.
#define REPORT(expr)                                                \
  INFO(expr);                                                       \
  do {                                                              \
    std::ostringstream report_os_;                                  \
    report_os_ << expr;                                             \
    std::printf("[validation] %s\n", report_os_.str().c_str());     \
  } while (0)

// Analytic-test baseline: index-matched readout, all other faces black, no
// bulk processes, prompt emission.
SimParams black_box() {
  SimParams p;
  p.surface = Surface::Black;
  p.n_det = p.n_scint;
  p.abs_length = kInfinity;
  p.scat_length = kInfinity;
  p.tau = 0.0f;
  return p;
}

std::array<uint64_t, kNumFates> count_fates(const PhotonRecords& r) {
  std::array<uint64_t, kNumFates> c{};
  for (uint8_t f : r.fate) ++c[f];
  return c;
}

// Normalised residual of a binomial count k out of n with expected probability p.
double binomial_pull(uint64_t k, uint64_t n, double p) {
  return (static_cast<double>(k) - n * p) / std::sqrt(n * p * (1.0 - p));
}

// Face on which a (snapped) end point lies, -1 if none.
int face_of(float x, float y, float z, const SimParams& p) {
  if (x == -p.hx) return 0;
  if (x == p.hx) return 1;
  if (y == -p.hy) return 2;
  if (y == p.hy) return 3;
  if (z == -p.hz) return 4;
  if (z == p.hz) return 5;
  return -1;
}

}  // namespace

TEST_CASE("Analytic helpers: quadrature reproduces the rectangle solid angle", "[analytic]") {
  SimParams p = black_box();
  for (auto [x, y, z] : {std::tuple{0.f, 0.f, 0.f}, {10.f, -5.f, 3.f}, {0.f, 0.f, 24.f},
                         {20.f, 20.f, -20.f}}) {
    p.src_x = x;
    p.src_y = y;
    p.src_z = z;
    double total = 0.0;
    for (int face = 0; face < 6; ++face) {
      const double omega = analytic::face_solid_angle_fraction(p, face);
      CHECK_THAT(analytic::face_survival_fraction(p, face, kInfinity), WithinRel(omega, 1e-9));
      total += omega;
    }
    CHECK_THAT(total, WithinAbs(1.0, 1e-12));  // the six faces close the box
  }
  p.src_x = p.src_y = p.src_z = 0.f;
  CHECK_THAT(analytic::face_solid_angle_fraction(p, 5), WithinAbs(1.0 / 6.0, 1e-14));
}

TEST_CASE("Validation 1: detection fraction equals solid angle / 4pi", "[validation]") {
  struct Case {
    float sx, sy, sz, size_x, size_y, size_z;
  };
  const Case cases[] = {{0, 0, 0, 50, 50, 50},     {10, -5, 3, 50, 50, 50},
                        {0, 0, 20, 50, 50, 50},    {20, 20, -20, 50, 50, 50},
                        {5, 10, -4, 30, 50, 20}};
  const uint64_t n = 2000000;
  for (const Case& c : cases) {
    SimParams p = black_box();
    p.src_x = c.sx;
    p.src_y = c.sy;
    p.src_z = c.sz;
    p.hx = c.size_x / 2;
    p.hy = c.size_y / 2;
    p.hz = c.size_z / 2;
    p.seed = 1000 + static_cast<uint64_t>(c.sx * 7 + c.sy * 3 + c.sz);
    PhotonRecords rec;
    run_cpu_records(p, n, rec, kAllThreads);

    // Per-face hit counts: the five black faces must ALSO match their solid angles.
    std::array<uint64_t, 6> per_face{};
    for (std::size_t i = 0; i < n; ++i) {
      const int f = face_of(rec.x[i], rec.y[i], rec.z[i], p);
      REQUIRE(f >= 0);
      ++per_face[f];
      REQUIRE(rec.n_boundary[i] == 1u);
      REQUIRE(rec.n_scatter[i] == 0u);
      REQUIRE(static_cast<Fate>(rec.fate[i]) ==
              (f == kReadoutFace ? Fate::Detected : Fate::AbsorbedSurface));
    }
    for (int f = 0; f < 6; ++f) {
      const double expected = analytic::face_solid_angle_fraction(p, f);
      const double pull = binomial_pull(per_face[f], n, expected);
      REPORT("source (" << c.sx << "," << c.sy << "," << c.sz << ") face " << f << ": MC "
                      << static_cast<double>(per_face[f]) / n << " analytic " << expected
                      << " pull " << pull);
      CHECK(std::fabs(pull) < 4.0);
    }
  }
}

TEST_CASE("Validation 1b: hit-position distribution on the readout face", "[validation]") {
  // Expected counts per bin are exact: the solid angle of each bin rectangle.
  SimParams p = black_box();
  p.src_x = 5;
  p.src_y = -3;
  p.src_z = -10;
  p.seed = 2024;
  const uint64_t n = 4000000;
  PhotonRecords rec;
  run_cpu_records(p, n, rec, kAllThreads);

  constexpr int kB = 20;
  std::vector<double> obs(kB * kB, 0.0), exp(kB * kB, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (static_cast<Fate>(rec.fate[i]) != Fate::Detected) continue;
    const int ix = std::min(kB - 1, static_cast<int>((rec.x[i] + p.hx) / (2 * p.hx) * kB));
    const int iy = std::min(kB - 1, static_cast<int>((rec.y[i] + p.hy) / (2 * p.hy) * kB));
    obs[iy * kB + ix] += 1.0;
  }
  const double h = p.hz - p.src_z;
  for (int iy = 0; iy < kB; ++iy) {
    for (int ix = 0; ix < kB; ++ix) {
      const double x1 = -p.hx + 2.0 * p.hx * ix / kB - p.src_x, x2 = x1 + 2.0 * p.hx / kB;
      const double y1 = -p.hy + 2.0 * p.hy * iy / kB - p.src_y, y2 = y1 + 2.0 * p.hy / kB;
      exp[iy * kB + ix] = n * analytic::rect_solid_angle(x1, x2, y1, y2, h) / (4 * analytic::kPi);
    }
  }
  const auto chi2 = stats::chi2_gof(obs, exp);
  REPORT("chi2/ndf = " << chi2.chi2 << "/" << chi2.ndf << ", p = " << chi2.pvalue);
  CHECK(chi2.ndf == kB * kB);
  CHECK(chi2.pvalue > 1e-4);
}

TEST_CASE("Validation 1c: time of flight and emission-time distribution", "[validation]") {
  SimParams p = black_box();
  p.src_x = -7;
  p.src_y = 4;
  p.src_z = 2;
  p.tau = 4.0f;
  p.seed = 77;
  const uint64_t n = 500000;
  PhotonRecords rec;
  run_cpu_records(p, n, rec, kAllThreads);

  const double inv_speed = static_cast<double>(p.n_scint) / kSpeedOfLight;
  std::vector<double> t_emit;
  for (std::size_t i = 0; i < n; ++i) {
    // Straight line from the source: the path length is the distance.
    const double dx = rec.x[i] - p.src_x, dy = rec.y[i] - p.src_y, dz = rec.z[i] - p.src_z;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    REQUIRE_THAT(rec.path[i], WithinRel(r, 1e-5));
    t_emit.push_back(rec.t[i] - rec.path[i] * inv_speed);
  }
  // t - path * n / c must be the sampled emission time, i.e. Exp(tau).
  const auto ks = stats::ks_one_sample(t_emit, [&](double t) { return 1.0 - std::exp(-t / p.tau); });
  REPORT("KS D = " << ks.d << ", p = " << ks.pvalue);
  CHECK(ks.pvalue > 1e-4);
}

TEST_CASE("Validation 2: bulk absorption against the attenuated solid angle", "[validation]") {
  // With black walls and an index-matched readout every photon travels in a
  // straight line from the source to one wall, so
  //   P(detected)       = (1/4pi) Int_{readout} exp(-r/L) dOmega
  //   P(absorbed bulk)  = 1 - (1/4pi) Int_{all faces} exp(-r/L) dOmega
  // evaluated by 2-D Gauss-Legendre quadrature (accurate to ~1e-12).
  const uint64_t n = 2000000;
  for (float L : {20.0f, 100.0f, 1000.0f}) {
    SimParams p = black_box();
    p.abs_length = L;
    p.src_x = 6;
    p.src_y = -8;
    p.src_z = -5;
    p.seed = 500 + static_cast<uint64_t>(L);
    PhotonRecords rec;
    run_cpu_records(p, n, rec, kAllThreads);
    const auto c = count_fates(rec);
    const double p_det = analytic::face_survival_fraction(p, kReadoutFace, L);
    const double p_abs = 1.0 - analytic::box_survival_fraction(p, L);
    const double pull_det = binomial_pull(c[0], n, p_det);
    const double pull_abs = binomial_pull(c[1], n, p_abs);
    REPORT("L = " << L << ": P_det MC " << double(c[0]) / n << " expected " << p_det
                << " (pull " << pull_det << "); P_abs MC " << double(c[1]) / n << " expected "
                << p_abs << " (pull " << pull_abs << ")");
    CHECK(std::fabs(pull_det) < 4.0);
    CHECK(std::fabs(pull_abs) < 4.0);
  }
}

TEST_CASE("Validation 3: Rayleigh scattering in a symmetric cube", "[validation]") {
  // Source at the centre of a cube whose six faces are all perfect absorbers
  // (5 black + the index-matched readout): by symmetry exactly 1/6 of the
  // photons are detected for ANY scattering length, and the unscattered
  // detected photons follow the attenuated solid angle with L = scat_length.
  SimParams p = black_box();
  p.scat_length = 15.0f;
  p.seed = 31;
  const uint64_t n = 2000000;
  PhotonRecords rec;
  run_cpu_records(p, n, rec, kAllThreads);
  const auto c = count_fates(rec);
  CHECK(c[0] + c[2] == n);
  const double pull = binomial_pull(c[0], n, 1.0 / 6.0);
  REPORT("P_det = " << double(c[0]) / n << " pull " << pull);
  CHECK(std::fabs(pull) < 4.0);

  uint64_t unscattered = 0, scatters = 0;
  for (std::size_t i = 0; i < n; ++i) {
    scatters += rec.n_scatter[i];
    if (rec.fate[i] == 0 && rec.n_scatter[i] == 0) ++unscattered;
  }
  const double expected = analytic::face_survival_fraction(p, kReadoutFace, p.scat_length);
  const double pull_u = binomial_pull(unscattered, n, expected);
  REPORT("unscattered detected " << double(unscattered) / n << " expected " << expected << " pull "
                               << pull_u);
  CHECK(std::fabs(pull_u) < 4.0);
  CHECK(scatters > n);  // mean number of scatters > 1 for L_s = 15 mm in a 50 mm cube
}

TEST_CASE("Validation 4: uniform volume source in a cube", "[validation]") {
  SimParams p = black_box();
  p.source = SourceType::UniformVolume;
  p.seed = 4;
  const uint64_t n = 2000000;
  PhotonRecords rec;
  run_cpu_records(p, n, rec, kAllThreads);
  const auto c = count_fates(rec);
  const double pull = binomial_pull(c[0], n, 1.0 / 6.0);
  REPORT("P_det = " << double(c[0]) / n << " pull " << pull);
  CHECK(std::fabs(pull) < 4.0);
}

TEST_CASE("Validation 5: escape cone of a perfectly specular box", "[validation]") {
  // Mirrors (R = 1) on five faces, readout to n_det = 1 (air), no bulk
  // processes. Only photons with |d_z| > cos(theta_c) can ever leave; the
  // others are trapped by TIR forever and hit the step limit.
  SimParams p;
  p.surface = Surface::Specular;
  p.reflectivity = 1.0f;
  p.n_det = 1.0f;
  p.abs_length = kInfinity;
  p.scat_length = kInfinity;
  p.max_steps = 1000;
  p.src_x = 3;
  p.src_y = -11;
  p.src_z = 7;
  p.seed = 5;
  const uint64_t n = 200000;
  PhotonRecords rec;
  run_cpu_records(p, n, rec, kAllThreads);
  const auto c = count_fates(rec);
  const double expected = analytic::mirror_box_escape_fraction(p.n_scint, p.n_det);
  const double pull = binomial_pull(c[0], n, expected);
  REPORT("P_det = " << double(c[0]) / n << " expected 1 - cos(theta_c) = " << expected << " pull "
                  << pull);
  CHECK(std::fabs(pull) < 4.0);
  CHECK(c[0] + c[4] == n);  // detected or trapped, nothing else
}

TEST_CASE("Validation 5b: lossless diffuse box detects everything", "[validation]") {
  // Lambertian walls with R = 1 randomise the direction, so every photon
  // eventually enters the escape cone of the readout face.
  SimParams p;
  p.surface = Surface::Lambertian;
  p.reflectivity = 1.0f;
  p.n_det = 1.0f;
  p.abs_length = kInfinity;
  p.seed = 6;
  const uint64_t n = 200000;
  PhotonRecords rec;
  run_cpu_records(p, n, rec, kAllThreads);
  const auto c = count_fates(rec);
  CHECK(c[0] == n);
}

TEST_CASE("CPU results do not depend on the number of threads", "[determinism]") {
  SimParams p;  // default: polished walls, absorption
  p.scat_length = 200.0f;
  p.source = SourceType::UniformVolume;
  const uint64_t n = 200000;
  PhotonRecords serial, parallel;
  run_cpu_records(p, n, serial, 1);
  run_cpu_records(p, n, parallel, kAllThreads);
  CHECK(serial.fate == parallel.fate);
  CHECK(serial.n_boundary == parallel.n_boundary);
  CHECK(serial.n_scatter == parallel.n_scatter);
  CHECK(std::memcmp(serial.t.data(), parallel.t.data(), n * sizeof(float)) == 0);
  CHECK(std::memcmp(serial.x.data(), parallel.x.data(), n * sizeof(float)) == 0);
  CHECK(std::memcmp(serial.path.data(), parallel.path.data(), n * sizeof(float)) == 0);

  // Photon i's history depends only on (seed, i): a run starting at photon
  // 1000 reproduces the tail of the full run.
  PhotonRecords tail;
  run_cpu_records(p, 500, tail, 1, 1000);
  for (std::size_t i = 0; i < 500; ++i) REQUIRE(tail.t[i] == serial.t[1000 + i]);

  // Tally mode agrees with the records.
  TallyConfig tc;
  Tally tally;
  run_cpu_tally(p, tc, n, tally, kAllThreads);
  const auto c = count_fates(serial);
  for (int f = 0; f < kNumFates; ++f) CHECK(tally.fate_counts[f] == c[f]);
  uint64_t hist_total = 0;
  for (uint64_t v : tally.time_hist) hist_total += v;
  CHECK(hist_total == c[0]);
}

TEST_CASE("Comparison tools: identical runs and independent seeds", "[compare]") {
  SimParams p;
  p.scat_length = 300.0f;
  const uint64_t n = 300000;
  PhotonRecords a, b, c;
  run_cpu_records(p, n, a, kAllThreads);
  run_cpu_records(p, n, b, 1);
  SimParams pc = p;
  pc.seed = p.seed + 1;
  run_cpu_records(pc, n, c, kAllThreads);

  const HistoryAgreement same = compare_histories(a, b);
  CHECK(same.bitwise_identical == n);
  CHECK(same.first_divergent == -1);

  // Different seeds: histories are unrelated, distributions are compatible.
  const HistoryAgreement diff = compare_histories(a, c);
  CHECK(diff.same_discrete < n);
  const DistributionComparison d = compare_distributions(a, c, p);
  REPORT("independent seeds: eff z " << d.eff_z << ", time chi2 p " << d.time_chi2.pvalue
                                     << ", KS p " << d.time_ks.pvalue << ", xy chi2 p "
                                     << d.xy_chi2.pvalue);
  CHECK(std::fabs(d.eff_z) < 4.0);
  CHECK(d.time_chi2.pvalue > 1e-4);
  CHECK(d.time_ks.pvalue > 1e-4);
  CHECK(d.xy_chi2.pvalue > 1e-4);

  // A real difference must be detected: change the readout index.
  SimParams pd = pc;
  pd.n_det = 1.5f;
  PhotonRecords e;
  run_cpu_records(pd, n, e, kAllThreads);
  const DistributionComparison bad = compare_distributions(a, e, p);
  REPORT("n_det 1.63 vs 1.5: eff z " << bad.eff_z << ", xy chi2 p " << bad.xy_chi2.pvalue);
  CHECK(std::fabs(bad.eff_z) > 5.0);
}
