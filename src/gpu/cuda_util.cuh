// SPDX-License-Identifier: MIT
//
// cuda_util.cuh - small host-side helpers shared by the GPU backends
// (gpu_backend.cu, gpu_event.cu): error checking, RAII device buffers, event
// timing, and the structure-of-arrays record buffers.
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "optphot/backend.hpp"

#define CUDA_CHECK(call)                                                                         \
  do {                                                                                           \
    const cudaError_t err_ = (call);                                                             \
    if (err_ != cudaSuccess) {                                                                   \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__) + " (" #call ")");      \
    }                                                                                            \
  } while (0)

namespace optphot {
namespace cuda_util {

inline double ms_since(std::chrono::steady_clock::time_point t0) {
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
  EventPair(const EventPair&) = delete;
  EventPair& operator=(const EventPair&) = delete;
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// Creates the CUDA context outside the timed region (the first CUDA call of a
// process costs 0.1-1 s, which is not transport time).
inline void ensure_context() { CUDA_CHECK(cudaFree(nullptr)); }

// Device pointers of the structure-of-arrays record buffers.
struct DeviceRecords {
  uint8_t* fate;
  uint32_t* n_boundary;
  uint32_t* n_scatter;
  float *t, *x, *y, *z, *path;
};

__device__ __forceinline__ void store_record(const DeviceRecords& out, uint64_t i,
                                             const PhotonResult& r) {
  out.fate[i] = static_cast<uint8_t>(r.fate);
  out.n_boundary[i] = r.n_boundary;
  out.n_scatter[i] = r.n_scatter;
  out.t[i] = r.t;
  out.x[i] = r.pos.x;
  out.y[i] = r.pos.y;
  out.z[i] = r.pos.z;
  out.path[i] = r.path;
}

// Device-side record buffers for up to `cap` photons.
struct DeviceRecordBuffers {
  DeviceBuffer<uint8_t> fate;
  DeviceBuffer<uint32_t> nb, ns;
  DeviceBuffer<float> t, x, y, z, path;
  explicit DeviceRecordBuffers(std::size_t cap)
      : fate(cap), nb(cap), ns(cap), t(cap), x(cap), y(cap), z(cap), path(cap) {}
  DeviceRecords view() const {
    return {fate.ptr, nb.ptr, ns.ptr, t.ptr, x.ptr, y.ptr, z.ptr, path.ptr};
  }
  // Copies records 0 .. m-1 to host records offset .. offset + m - 1.
  void copy_to_host(PhotonRecords& out, std::size_t offset, std::size_t m) const {
    CUDA_CHECK(cudaMemcpy(out.fate.data() + offset, fate.ptr, m * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.n_boundary.data() + offset, nb.ptr, m * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.n_scatter.data() + offset, ns.ptr, m * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.t.data() + offset, t.ptr, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.x.data() + offset, x.ptr, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.y.data() + offset, y.ptr, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.z.data() + offset, z.ptr, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(out.path.data() + offset, path.ptr, m * sizeof(float), cudaMemcpyDeviceToHost));
  }
};

}  // namespace cuda_util
}  // namespace optphot
