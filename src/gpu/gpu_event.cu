// SPDX-License-Identifier: MIT
//
// GPU backend, version 2: event-based ("wavefront") transport with periodic
// compaction of the alive photons.
//
// Why: in the history-based kernels a warp runs until its longest photon is
// done, and a few photons trapped by total internal reflection keep 31 lanes
// idle for hundreds of steps (python/divergence_estimate.py: only 16 % of the
// lanes do useful work in the default cube).
//
// How: the photon state lives in global memory between launches. Each launch
// of event_step_kernel advances every alive photon by at most k steps
// (steps_per_launch), using the same start_photon / step_photon as the
// history-based kernels (include/optphot/transport.hpp). Then:
//   * a photon that has finished writes its record (indexed by its photon
//     number, so the output order does not depend on the scheduling);
//   * a photon that is still alive writes its state to the next state buffer,
//     at a slot taken from a global counter. The slot is allocated once per
//     warp (warp-aggregated atomic: one atomicAdd for all alive lanes, each
//     lane's offset = number of alive lanes below it), so the survivors are
//     packed densely and the next launch runs full warps.
// The host repeats this until no photon is left, swapping the two state
// buffers. k = 1 is fully event-based; k >= max_steps is one launch, i.e. the
// history-based kernel again.
//
// The price: every launch reads and writes 48 bytes of state per alive photon
// (structure of arrays, so these accesses are coalesced), plus one kernel
// launch and one 4-byte device-to-host copy (the next count) per launch.
// The RNG state is a single counter (PhiloxStream::draws()): a counter-based
// generator can be resumed from the number of values consumed.
//
// The kernel also counts useful and issued lane-steps per warp, which gives
// the measured SIMT efficiency to compare with the model of
// divergence_estimate.py.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

#include "cuda_util.cuh"
#include "optphot/backend.hpp"
#include "optphot/transport.hpp"

namespace optphot {

namespace {

using namespace cuda_util;
using u64 = unsigned long long;  // the type CUDA's 64-bit atomicAdd takes

constexpr unsigned kFullMask = 0xffffffffu;

// Photon state between launches, structure of arrays: 12 x 4 = 48 bytes.
struct DeviceState {
  float *px, *py, *pz, *dx, *dy, *dz, *t, *path;
  uint32_t *nb, *ns;
  uint32_t* draws;  // words consumed from the photon's Philox stream
  uint32_t* id;     // photon number within the chunk (index of its record)
};

struct DeviceStateBuffers {
  DeviceBuffer<float> px, py, pz, dx, dy, dz, t, path;
  DeviceBuffer<uint32_t> nb, ns, draws, id;
  explicit DeviceStateBuffers(std::size_t cap)
      : px(cap),
        py(cap),
        pz(cap),
        dx(cap),
        dy(cap),
        dz(cap),
        t(cap),
        path(cap),
        nb(cap),
        ns(cap),
        draws(cap),
        id(cap) {}
  DeviceState view() const {
    return {px.ptr, py.ptr,   pz.ptr, dx.ptr, dy.ptr,    dz.ptr,
            t.ptr,  path.ptr, nb.ptr, ns.ptr, draws.ptr, id.ptr};
  }
};

__device__ __forceinline__ PhotonState load_state(const DeviceState& s, uint32_t j) {
  return {{s.px[j], s.py[j], s.pz[j]},
          {s.dx[j], s.dy[j], s.dz[j]},
          s.t[j],
          s.path[j],
          s.nb[j],
          s.ns[j]};
}

__device__ __forceinline__ void store_state(const DeviceState& s, uint32_t j, const PhotonState& st,
                                            uint32_t draws, uint32_t id) {
  s.px[j] = st.pos.x;
  s.py[j] = st.pos.y;
  s.pz[j] = st.pos.z;
  s.dx[j] = st.dir.x;
  s.dy[j] = st.dir.y;
  s.dz[j] = st.dir.z;
  s.t[j] = st.t;
  s.path[j] = st.path;
  s.nb[j] = st.n_boundary;
  s.ns[j] = st.n_scatter;
  s.draws[j] = draws;
  s.id[j] = id;
}

__device__ __forceinline__ uint32_t warp_max(uint32_t v) {
  for (int offset = 16; offset > 0; offset >>= 1) v = max(v, __shfl_xor_sync(kFullMask, v, offset));
  return v;
}

__device__ __forceinline__ uint32_t warp_sum(uint32_t v) {
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(kFullMask, v, offset);
  return v;
}

// One launch: thread j advances alive photon j by at most k steps.
// first_launch: no state yet, thread j emits photon j of the chunk.
// next_count: number of survivors written to `out` (zeroed by the host).
// lane_stats: [useful lane-steps, issued lane-steps], accumulated.
__global__ void event_step_kernel(SimParams p, uint64_t first_id, uint32_t count, bool first_launch,
                                  uint32_t k, DeviceState in, DeviceState out,
                                  unsigned int* next_count, u64* lane_stats, DeviceRecords rec) {
  // No early return: every lane must reach the warp-wide ballot and shuffles
  // below (the host guarantees whole warps: block_size % 32 == 0).
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = j < count;
  bool alive = false;
  uint32_t steps = 0;  // steps done by this lane in this launch
  uint32_t id = 0, draws = 0;
  PhotonState s{};
  if (active) {
    id = first_launch ? j : in.id[j];
    PhiloxStream rng(p.seed, first_id + id, first_launch ? 0u : in.draws[j]);
    s = first_launch ? start_photon(p, rng) : load_state(in, j);
    // Steps completed before this launch (see PhotonState).
    uint32_t done = s.n_boundary + s.n_scatter;
    Fate fate = Fate::MaxSteps;
    alive = done < p.max_steps;
    while (alive && steps < k) {
      alive = step_photon(p, rng, s, fate);
      ++steps;
      if (alive && ++done >= p.max_steps) alive = false;  // fate stays MaxSteps
    }
    if (!alive) store_record(rec, id, finish_photon(s, fate));
    draws = rng.draws();
  }

  // Compaction: one atomicAdd per warp reserves a contiguous range of slots
  // for the warp's alive lanes; each lane takes the slot given by the number
  // of alive lanes below it.
  const unsigned ballot = __ballot_sync(kFullMask, alive);
  const unsigned lane = threadIdx.x & 31u;
  unsigned base = 0;
  if (lane == 0 && ballot != 0u)
    base = atomicAdd(next_count, static_cast<unsigned>(__popc(ballot)));
  base = __shfl_sync(kFullMask, base, 0);
  if (alive) store_state(out, base + __popc(ballot & ((1u << lane) - 1u)), s, draws, id);

  // SIMT efficiency: the warp runs as many iterations as its busiest lane.
  const uint32_t warp_steps = warp_max(steps), useful = warp_sum(steps);
  if (lane == 0 && warp_steps > 0) {
    atomicAdd(&lane_stats[0], static_cast<u64>(useful));
    atomicAdd(&lane_stats[1], 32ull * warp_steps);
  }
}

}  // namespace

RunTiming run_gpu_records_event(const SimParams& p, uint64_t n, PhotonRecords& out,
                                uint32_t steps_per_launch, int block_size, uint64_t first_id,
                                uint64_t chunk, EventStats* stats) {
  if (block_size <= 0 || block_size % 32 != 0 || block_size > 1024) {
    throw std::invalid_argument("event-based backend: block_size must be a multiple of 32 <= 1024");
  }
  if (steps_per_launch == 0) {
    throw std::invalid_argument("event-based backend: steps_per_launch must be >= 1");
  }
  ensure_context();
  RunTiming timing;
  EventStats st;
  const auto t0 = std::chrono::steady_clock::now();
  out.resize(n);
  // 32-bit slot indices in the kernel.
  const uint64_t cap = std::max<uint64_t>(1, std::min({n, chunk, uint64_t{1} << 31}));

  const auto ta = std::chrono::steady_clock::now();
  const DeviceRecordBuffers records(cap);
  const DeviceStateBuffers state_a(cap), state_b(cap);
  DeviceBuffer<unsigned int> next_count(1);
  DeviceBuffer<u64> lane_stats(2);
  CUDA_CHECK(cudaMemset(lane_stats.ptr, 0, 2 * sizeof(u64)));
  timing.alloc_ms = ms_since(ta);

  EventPair loop_ev, kernel_ev, copy_ev;
  for (uint64_t done = 0; done < n; done += cap) {
    const uint64_t m = std::min(cap, n - done);
    DeviceState in = state_a.view(), next = state_b.view();
    uint32_t count = static_cast<uint32_t>(m);
    bool first_launch = true;

    CUDA_CHECK(cudaEventRecord(loop_ev.start));
    while (count > 0) {
      CUDA_CHECK(cudaMemsetAsync(next_count.ptr, 0, sizeof(unsigned int)));
      const unsigned grid = (count + block_size - 1) / block_size;
      CUDA_CHECK(cudaEventRecord(kernel_ev.start));
      event_step_kernel<<<grid, block_size>>>(p, first_id + done, count, first_launch,
                                              steps_per_launch, in, next, next_count.ptr,
                                              lane_stats.ptr, records.view());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(kernel_ev.stop));
      // The host needs the survivor count to size the next launch: this copy
      // waits for the kernel, which is part of the cost of the method.
      unsigned int survivors = 0;
      CUDA_CHECK(
          cudaMemcpy(&survivors, next_count.ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      st.kernels_ms += kernel_ev.elapsed_ms();
      ++st.launches;
      std::swap(in, next);
      count = survivors;
      first_launch = false;
    }
    CUDA_CHECK(cudaEventRecord(loop_ev.stop));
    timing.kernel_ms += loop_ev.elapsed_ms();

    CUDA_CHECK(cudaEventRecord(copy_ev.start));
    records.copy_to_host(out, done, m);
    CUDA_CHECK(cudaEventRecord(copy_ev.stop));
    timing.d2h_ms += copy_ev.elapsed_ms();
  }

  u64 host_stats[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(host_stats, lane_stats.ptr, sizeof(host_stats), cudaMemcpyDeviceToHost));
  st.useful_lane_steps = host_stats[0];
  st.issued_lane_steps = host_stats[1];
  if (stats) *stats = st;
  timing.h2d_ms = 0.0;
  timing.total_ms = ms_since(t0);
  return timing;
}

}  // namespace optphot
