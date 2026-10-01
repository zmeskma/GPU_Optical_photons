// SPDX-License-Identifier: MIT
//
// CPU reference backend: a loop over photons calling the shared
// transport_photon(). Optionally parallelised with OpenMP. Because every
// photon owns its random stream, the result does not depend on the number of
// threads or on the scheduling (checked in tests/test_transport_cpu.cpp).
#include <chrono>

#include "optphot/backend.hpp"

#if defined(OPTPHOT_HAVE_OPENMP)
#include <omp.h>
#endif

namespace optphot {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Photon histories have very different lengths (a photon that is absorbed at
// the first wall vs. one trapped by total internal reflection), so a dynamic
// schedule with moderately large chunks balances the load between threads.
constexpr int kOmpChunk = 4096;

void accumulate(Tally& tally, const PhotonResult& r, const SimParams& p, const TallyConfig& c) {
  ++tally.fate_counts[static_cast<int>(r.fate)];
  tally.boundary_sum += r.n_boundary;
  tally.scatter_sum += r.n_scatter;
  if (r.fate == Fate::Detected) {
    ++tally.time_hist[time_bin(r.t, c)];
    ++tally.xy_hist[xy_bin(r.pos.x, r.pos.y, p, c)];
  }
}

}  // namespace

void PhotonRecords::resize(std::size_t n) {
  fate.resize(n);
  n_boundary.resize(n);
  n_scatter.resize(n);
  t.resize(n);
  x.resize(n);
  y.resize(n);
  z.resize(n);
  path.resize(n);
}

void Tally::reset(const TallyConfig& c) {
  fate_counts.fill(0);
  time_hist.assign(c.time_bins + 1, 0);
  xy_hist.assign(static_cast<std::size_t>(c.xy_bins) * c.xy_bins, 0);
  boundary_sum = scatter_sum = 0;
}

void Tally::add(const Tally& o) {
  for (int i = 0; i < kNumFates; ++i) fate_counts[i] += o.fate_counts[i];
  for (std::size_t i = 0; i < time_hist.size(); ++i) time_hist[i] += o.time_hist[i];
  for (std::size_t i = 0; i < xy_hist.size(); ++i) xy_hist[i] += o.xy_hist[i];
  boundary_sum += o.boundary_sum;
  scatter_sum += o.scatter_sum;
}

uint64_t Tally::total() const {
  uint64_t s = 0;
  for (uint64_t c : fate_counts) s += c;
  return s;
}

bool cpu_has_openmp() {
#if defined(OPTPHOT_HAVE_OPENMP)
  return true;
#else
  return false;
#endif
}

int cpu_max_threads() {
#if defined(OPTPHOT_HAVE_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

RunTiming run_cpu_records(const SimParams& p, uint64_t n, PhotonRecords& out, int threads,
                          uint64_t first_id) {
  const auto t0 = std::chrono::steady_clock::now();
  out.resize(n);
  const auto t1 = std::chrono::steady_clock::now();
  const int64_t count = static_cast<int64_t>(n);

  if (threads == 1 || !cpu_has_openmp()) {
    for (int64_t i = 0; i < count; ++i) {
      out.set(i, transport_photon(p, first_id + i));
    }
  } else {
#if defined(OPTPHOT_HAVE_OPENMP)
    const int nt = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, kOmpChunk) num_threads(nt)
    for (int64_t i = 0; i < count; ++i) {
      out.set(i, transport_photon(p, first_id + i));
    }
#endif
  }

  RunTiming timing;
  timing.kernel_ms = ms_since(t1);
  timing.total_ms = ms_since(t0);
  return timing;
}

RunTiming run_cpu_tally(const SimParams& p, const TallyConfig& c, uint64_t n, Tally& out,
                        int threads, uint64_t first_id) {
  const auto t0 = std::chrono::steady_clock::now();
  out.reset(c);
  const int64_t count = static_cast<int64_t>(n);

  if (threads == 1 || !cpu_has_openmp()) {
    for (int64_t i = 0; i < count; ++i) {
      accumulate(out, transport_photon(p, first_id + i), p, c);
    }
  } else {
#if defined(OPTPHOT_HAVE_OPENMP)
    const int nt = threads > 0 ? threads : omp_get_max_threads();
    // Each thread fills a private tally; the merge is a sum of integers, so
    // the result is exactly reproducible regardless of thread count/order.
#pragma omp parallel num_threads(nt)
    {
      Tally local;
      local.reset(c);
#pragma omp for schedule(dynamic, kOmpChunk) nowait
      for (int64_t i = 0; i < count; ++i) {
        accumulate(local, transport_photon(p, first_id + i), p, c);
      }
#pragma omp critical
      out.add(local);
    }
#endif
  }

  RunTiming timing;
  timing.total_ms = timing.kernel_ms = ms_since(t0);
  return timing;
}

}  // namespace optphot
