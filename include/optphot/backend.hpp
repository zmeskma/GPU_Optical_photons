// SPDX-License-Identifier: MIT
//
// backend.hpp - host-side API of the CPU and GPU implementations.
//
// Two output modes are provided by both backends:
//
//  * records: one record per photon (structure-of-arrays). Needed for the
//    photon-by-photon CPU/GPU comparison and for detailed analysis in Python.
//    Costs N * 29 bytes of memory and, on the GPU, a device-to-host copy that
//    can dominate the run time (see the benchmarks).
//
//  * tally: only aggregated quantities (fate counts, arrival-time histogram,
//    hit map). This is what a production code would do; on the GPU it is
//    implemented with integer atomics (see src/gpu/gpu_backend.cu).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "optphot/params.hpp"
#include "optphot/transport.hpp"

namespace optphot {

// Structure-of-arrays photon records. On the GPU each array is a separate
// device buffer, so that thread i writing element i of every array produces
// fully coalesced memory transactions.
struct PhotonRecords {
  std::vector<uint8_t> fate;
  std::vector<uint32_t> n_boundary;
  std::vector<uint32_t> n_scatter;
  std::vector<float> t, x, y, z, path;

  void resize(std::size_t n);
  std::size_t size() const { return fate.size(); }
  void set(std::size_t i, const PhotonResult& r) {
    fate[i] = static_cast<uint8_t>(r.fate);
    n_boundary[i] = r.n_boundary;
    n_scatter[i] = r.n_scatter;
    t[i] = r.t;
    x[i] = r.pos.x;
    y[i] = r.pos.y;
    z[i] = r.pos.z;
    path[i] = r.path;
  }
  static constexpr std::size_t kBytesPerPhoton = 1 + 4 + 4 + 5 * 4;
};

struct Tally {
  std::array<uint64_t, kNumFates> fate_counts{};
  std::vector<uint64_t> time_hist;  // time_bins + 1 (last = overflow)
  std::vector<uint64_t> xy_hist;    // xy_bins * xy_bins, row-major [iy][ix]
  uint64_t boundary_sum = 0;        // total boundary interactions (all photons)
  uint64_t scatter_sum = 0;         // total Rayleigh scatterings (all photons)

  void reset(const TallyConfig& c);
  void add(const Tally& other);
  uint64_t total() const;
};

// Wall-clock breakdown of a run, in milliseconds.
struct RunTiming {
  double total_ms = 0.0;   // everything, as seen by the caller
  double kernel_ms = 0.0;  // transport only (GPU: CUDA events around the kernels)
  double h2d_ms = 0.0;     // host -> device transfers (GPU only)
  double d2h_ms = 0.0;     // device -> host transfers (GPU only)
  double alloc_ms = 0.0;   // device allocation / memset (GPU only)
};

// ---- CPU backend (src/cpu/cpu_backend.cpp) --------------------------------
// threads: 1 = plain serial loop; 0 = all OpenMP threads; n > 1 = n threads.
// Photon indices are first_id, ..., first_id + n - 1.
RunTiming run_cpu_records(const SimParams& p, uint64_t n, PhotonRecords& out, int threads = 1,
                          uint64_t first_id = 0);
RunTiming run_cpu_tally(const SimParams& p, const TallyConfig& c, uint64_t n, Tally& out,
                        int threads = 1, uint64_t first_id = 0);
bool cpu_has_openmp();
int cpu_max_threads();

// ---- GPU backend (src/gpu/gpu_backend.cu, only with WITH_CUDA=ON) ----------
bool gpu_available();
std::string gpu_device_name();
// block_size: CUDA threads per block. Records are produced in chunks of at
// most `chunk` photons to bound device memory.
RunTiming run_gpu_records(const SimParams& p, uint64_t n, PhotonRecords& out, int block_size = 256,
                          uint64_t first_id = 0, uint64_t chunk = 1ull << 24);
RunTiming run_gpu_tally(const SimParams& p, const TallyConfig& c, uint64_t n, Tally& out,
                        int block_size = 256, uint64_t first_id = 0);

// ---- GPU event-based backend (src/gpu/gpu_event.cu) ------------------------
// Diagnostics of an event-based run.
struct EventStats {
  uint64_t launches = 0;           // step-kernel launches (all chunks)
  uint64_t useful_lane_steps = 0;  // transport steps done = sum of all photons' steps
  uint64_t issued_lane_steps = 0;  // per warp and launch: 32 x the most steps any lane did
  double kernels_ms = 0.0;         // sum of the step-kernel durations alone
  double simt_efficiency() const {
    return issued_lane_steps ? static_cast<double>(useful_lane_steps) / issued_lane_steps : 0.0;
  }
};

// Same records as run_gpu_records (bit for bit), computed event-based: each
// launch advances every alive photon by up to steps_per_launch steps, then the
// survivors are compacted so that the next launch runs full warps.
// RunTiming::kernel_ms is the whole transport loop including the launch and
// synchronisation gaps between kernels (comparable to run_gpu_records'
// kernel_ms); EventStats::kernels_ms is the kernels alone.
// block_size must be a multiple of 32 (warp-level compaction).
RunTiming run_gpu_records_event(const SimParams& p, uint64_t n, PhotonRecords& out,
                                uint32_t steps_per_launch, int block_size = 256,
                                uint64_t first_id = 0, uint64_t chunk = 1ull << 24,
                                EventStats* stats = nullptr);

}  // namespace optphot
