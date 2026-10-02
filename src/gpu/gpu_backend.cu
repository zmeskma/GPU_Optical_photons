// SPDX-License-Identifier: MIT
//
// GPU backend, version 1: history-based transport, one CUDA thread per
// photon. Each thread calls exactly the same transport_photon() as the CPU
// reference (include/optphot/transport.hpp, compiled here for the device).
//
// Two kernels:
//
//  * transport_records_kernel: thread i transports photon first_id + i and
//    writes its record into structure-of-arrays device buffers. Consecutive
//    threads of a warp write consecutive addresses of each array, so every
//    store is coalesced. No atomics, no inter-thread communication; the
//    result is deterministic. The price is memory and a large
//    device-to-host copy (29 bytes per photon).
//
//  * transport_tally_kernel: only histograms leave the GPU. Many threads
//    update the same bins, which requires atomics. To keep contention low,
//    each block first accumulates into a private copy of the histograms in
//    shared memory (fast on-chip atomics, contention only within the block),
//    then adds its non-zero bins to the global histograms once. All counters
//    are integers, so the result is independent of the order in which the
//    atomics happen: tallies are exactly reproducible (unlike float atomics).
//    The kernel uses a grid-stride loop with a grid sized to fill the GPU, so
//    that each block processes many photons and the cost of initialising and
//    flushing its shared histograms is amortised.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "optphot/backend.hpp"
#include "optphot/transport.hpp"

namespace optphot {

namespace {

#define CUDA_CHECK(call)                                                                         \
  do {                                                                                           \
    const cudaError_t err_ = (call);                                                             \
    if (err_ != cudaSuccess) {                                                                   \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__) + " (" #call ")");      \
    }                                                                                            \
  } while (0)

// Device pointers of the structure-of-arrays record buffers.
struct DeviceRecords {
  uint8_t* fate;
  uint32_t* n_boundary;
  uint32_t* n_scatter;
  float *t, *x, *y, *z, *path;
};

// SimParams is passed by value: kernel arguments live in the constant bank,
// which is cached and broadcast to all threads of a warp reading the same
// address - exactly the access pattern here.
__global__ void transport_records_kernel(SimParams p, uint64_t first_id, uint64_t n,
                                         DeviceRecords out) {
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;  // the last block may be partially filled
  const PhotonResult r = transport_photon(p, first_id + i);
  out.fate[i] = static_cast<uint8_t>(r.fate);
  out.n_boundary[i] = r.n_boundary;
  out.n_scatter[i] = r.n_scatter;
  out.t[i] = r.t;
  out.x[i] = r.pos.x;
  out.y[i] = r.pos.y;
  out.z[i] = r.pos.z;
  out.path[i] = r.path;
}

using u64 = unsigned long long;  // the type CUDA's 64-bit atomicAdd takes

// Global tally layout: [fates (kNumFates) | time hist (time_bins + 1) |
// xy hist (xy_bins^2) | boundary sum | scatter sum], all u64.
struct TallyLayout {
  uint32_t fate_off, time_off, xy_off, sums_off, total;
  __host__ __device__ explicit TallyLayout(const TallyConfig& c)
      : fate_off(0),
        time_off(kNumFates),
        xy_off(kNumFates + c.time_bins + 1),
        sums_off(kNumFates + c.time_bins + 1 + c.xy_bins * c.xy_bins),
        total(sums_off + 2) {}
};

__global__ void transport_tally_kernel(SimParams p, TallyConfig c, uint64_t first_id, uint64_t n,
                                       u64* global_tally) {
  const TallyLayout L(c);
  // Block-private histograms in shared memory (32-bit counters: one block
  // never sees more than 2^32 photons per bin).
  extern __shared__ uint32_t s_hist[];
  const uint32_t n_bins = L.sums_off;  // everything except the two sums
  for (uint32_t b = threadIdx.x; b < n_bins; b += blockDim.x) s_hist[b] = 0u;
  __syncthreads();

  // Per-thread sums kept in registers, added once at the end.
  u64 boundary_sum = 0, scatter_sum = 0;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const PhotonResult r = transport_photon(p, first_id + i);
    atomicAdd(&s_hist[L.fate_off + static_cast<uint32_t>(r.fate)], 1u);
    if (r.fate == Fate::Detected) {
      atomicAdd(&s_hist[L.time_off + time_bin(r.t, c)], 1u);
      atomicAdd(&s_hist[L.xy_off + xy_bin(r.pos.x, r.pos.y, p, c)], 1u);
    }
    boundary_sum += r.n_boundary;
    scatter_sum += r.n_scatter;
  }
  atomicAdd(&global_tally[L.sums_off], boundary_sum);
  atomicAdd(&global_tally[L.sums_off + 1], scatter_sum);
  __syncthreads();

  // Flush: one global atomic per non-empty bin per block.
  for (uint32_t b = threadIdx.x; b < n_bins; b += blockDim.x) {
    if (s_hist[b] != 0u) atomicAdd(&global_tally[b], static_cast<u64>(s_hist[b]));
  }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// RAII wrapper so that device memory is released on exceptions too.
template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(std::size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct EventPair {
  cudaEvent_t start, stop;
  EventPair() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~EventPair() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// Creates the CUDA context outside the timed region (the first CUDA call of a
// process costs 0.1-1 s, which is not transport time).
void ensure_context() { CUDA_CHECK(cudaFree(nullptr)); }

}  // namespace

bool gpu_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string gpu_device_name() {
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    return "none";
  }
  return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
         std::to_string(prop.minor) + ", " + std::to_string(prop.multiProcessorCount) + " SMs)";
}

RunTiming run_gpu_records(const SimParams& p, uint64_t n, PhotonRecords& out, int block_size,
                          uint64_t first_id, uint64_t chunk) {
  ensure_context();
  RunTiming timing;
  const auto t0 = std::chrono::steady_clock::now();
  out.resize(n);
  const uint64_t cap = std::max<uint64_t>(1, std::min(n, chunk));

  const auto ta = std::chrono::steady_clock::now();
  DeviceBuffer<uint8_t> fate(cap);
  DeviceBuffer<uint32_t> nb(cap), ns(cap);
  DeviceBuffer<float> t(cap), x(cap), y(cap), z(cap), path(cap);
  const DeviceRecords dev{fate.ptr, nb.ptr, ns.ptr, t.ptr, x.ptr, y.ptr, z.ptr, path.ptr};
  timing.alloc_ms = ms_since(ta);

  EventPair kernel_ev, copy_ev;
  for (uint64_t done = 0; done < n; done += cap) {
    const uint64_t m = std::min(cap, n - done);
    const unsigned grid = static_cast<unsigned>((m + block_size - 1) / block_size);

    CUDA_CHECK(cudaEventRecord(kernel_ev.start));
    transport_records_kernel<<<grid, block_size>>>(p, first_id + done, m, dev);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_ev.stop));
    timing.kernel_ms += kernel_ev.elapsed_ms();

    // Pageable host memory (std::vector): the driver stages the copy through
    // a pinned buffer. Pinned host memory + streams would allow faster,
    // asynchronous copies overlapping the next chunk's kernel (see README).
    CUDA_CHECK(cudaEventRecord(copy_ev.start));
    CUDA_CHECK(
        cudaMemcpy(out.fate.data() + done, dev.fate, m * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.n_boundary.data() + done, dev.n_boundary, m * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.n_scatter.data() + done, dev.n_scatter, m * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.t.data() + done, dev.t, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.x.data() + done, dev.x, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.y.data() + done, dev.y, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.z.data() + done, dev.z, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(out.path.data() + done, dev.path, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(copy_ev.stop));
    timing.d2h_ms += copy_ev.elapsed_ms();
  }
  // Parameters travel as kernel arguments and photons are generated on the
  // device, so there is no bulk host-to-device transfer in this design.
  timing.h2d_ms = 0.0;
  timing.total_ms = ms_since(t0);
  return timing;
}

RunTiming run_gpu_tally(const SimParams& p, const TallyConfig& c, uint64_t n, Tally& out,
                        int block_size, uint64_t first_id) {
  ensure_context();
  RunTiming timing;
  const auto t0 = std::chrono::steady_clock::now();
  out.reset(c);
  const TallyLayout L(c);
  const std::size_t shared_bytes = static_cast<std::size_t>(L.sums_off) * sizeof(uint32_t);

  int dev = 0, n_sm = 0, max_shared = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, dev));
  if (shared_bytes > static_cast<std::size_t>(max_shared)) {
    throw std::runtime_error("tally histograms need " + std::to_string(shared_bytes) +
                             " bytes of shared memory; reduce xy_bins / time_bins");
  }
  // Grid sized to the number of blocks that can be resident at once.
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, transport_tally_kernel,
                                                           block_size, shared_bytes));
  const uint64_t needed = (n + block_size - 1) / block_size;
  const unsigned grid = static_cast<unsigned>(std::max<uint64_t>(
      1, std::min<uint64_t>(needed, static_cast<uint64_t>(blocks_per_sm) * n_sm)));
  // The block-private counters are 32-bit: a block must not see 2^32 photons.
  if (n / grid >= 0xffffffffull) {
    throw std::runtime_error("too many photons for one tally launch; split the run");
  }

  const auto ta = std::chrono::steady_clock::now();
  DeviceBuffer<u64> tally(L.total);
  CUDA_CHECK(cudaMemset(tally.ptr, 0, L.total * sizeof(u64)));
  timing.alloc_ms = ms_since(ta);

  EventPair kernel_ev, copy_ev;
  CUDA_CHECK(cudaEventRecord(kernel_ev.start));
  transport_tally_kernel<<<grid, block_size, shared_bytes>>>(p, c, first_id, n, tally.ptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_ev.stop));
  timing.kernel_ms = kernel_ev.elapsed_ms();

  std::vector<u64> host(L.total);
  CUDA_CHECK(cudaEventRecord(copy_ev.start));
  CUDA_CHECK(cudaMemcpy(host.data(), tally.ptr, L.total * sizeof(u64), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(copy_ev.stop));
  timing.d2h_ms = copy_ev.elapsed_ms();

  for (int f = 0; f < kNumFates; ++f) out.fate_counts[f] = host[L.fate_off + f];
  std::copy(host.begin() + L.time_off, host.begin() + L.xy_off, out.time_hist.begin());
  std::copy(host.begin() + L.xy_off, host.begin() + L.sums_off, out.xy_hist.begin());
  out.boundary_sum = host[L.sums_off];
  out.scatter_sum = host[L.sums_off + 1];
  timing.total_ms = ms_since(t0);
  return timing;
}

}  // namespace optphot
