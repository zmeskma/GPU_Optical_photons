// SPDX-License-Identifier: MIT
//
// hd.hpp - portability macros shared by host (C++) and device (CUDA) code.
//
// Every physics function in this project is marked OPT_HD. When the header is
// compiled by nvcc this expands to `__host__ __device__`, so the *same source*
// is compiled once for the CPU and once for the GPU. When compiled by a plain
// C++ compiler (CPU-only build) it expands to nothing.
#pragma once

#include <math.h>

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define OPT_HD __host__ __device__
#define OPT_INLINE __forceinline__
#else
#define OPT_HD
#define OPT_INLINE inline
#endif

// True while compiling the device pass of a .cu file.
#if defined(__CUDA_ARCH__)
#define OPT_DEVICE_PASS 1
#else
#define OPT_DEVICE_PASS 0
#endif

namespace optphot {

// Units used throughout: length in mm, time in ns.
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kSpeedOfLight = 299.792458f;  // mm / ns
// numeric_limits rather than the INFINITY macro: MSVC defines INFINITY as an
// overflowing cast, which nvcc rejects in a constant expression.
constexpr float kInfinity = std::numeric_limits<float>::infinity();

// Portable test for +inf (used as "process disabled"); avoids the
// host/device differences of isinf / std::isinf overloads.
OPT_HD OPT_INLINE bool is_inf(float x) { return x == kInfinity; }

}  // namespace optphot
