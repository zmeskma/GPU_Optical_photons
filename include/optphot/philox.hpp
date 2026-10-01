// SPDX-License-Identifier: MIT
//
// philox.hpp - counter-based Philox4x32-10 random number generator.
//
// Reference: J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw,
// "Parallel random numbers: as easy as 1, 2, 3", SC11 (2011).
// This is the same algorithm as Random123's philox4x32_10 and cuRAND's
// Philox4_32_10; the known-answer vectors in tests/test_rng.cpp come from
// Random123.
//
// Why counter-based? A counter-based generator is a pure function
//   (counter, key) -> 128 random bits.
// There is no state to store, seed or skip ahead: photon i simply uses
// counters {0,1,2,...} with a key derived from the run seed, and photon i's
// random numbers are the same whichever thread, warp, GPU or CPU core
// processes it, and in whatever order. That is what makes a photon-by-photon
// comparison of the CPU and GPU implementations possible.
//
// Integer arithmetic only (32x32->64 multiply, xor, add), so the raw bits are
// bit-for-bit identical on every platform.
#pragma once

#include "optphot/hd.hpp"

namespace optphot {

struct U32x4 {
  uint32_t x, y, z, w;
};

struct U32x2 {
  uint32_t x, y;
};

namespace detail {

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3) - 1

// Returns the low 32 bits of a*b and stores the high 32 bits in `hi`.
OPT_HD OPT_INLINE uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t& hi) {
#if OPT_DEVICE_PASS
  hi = __umulhi(a, b);
  return a * b;
#else
  const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  hi = static_cast<uint32_t>(p >> 32);
  return static_cast<uint32_t>(p);
#endif
}

OPT_HD OPT_INLINE U32x4 philox_round(U32x4 c, U32x2 k) {
  uint32_t hi0, hi1;
  const uint32_t lo0 = mulhilo(kPhiloxM0, c.x, hi0);
  const uint32_t lo1 = mulhilo(kPhiloxM1, c.z, hi1);
  return {hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0};
}

}  // namespace detail

// One Philox4x32 block with 10 rounds: 128 random bits from (counter, key).
OPT_HD OPT_INLINE U32x4 philox4x32_10(U32x4 ctr, U32x2 key) {
#if defined(__CUDACC__)
#pragma unroll
#endif
  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      key.x += detail::kPhiloxW0;
      key.y += detail::kPhiloxW1;
    }
    ctr = detail::philox_round(ctr, key);
  }
  return ctr;
}

// Maps 32 random bits to a float in (0, 1].
// The top 24 bits are used, so the result is exactly k * 2^-24 with
// k in [1, 2^24]: every value is exactly representable in float, the result
// is never 0 (safe for -log(u)) and is identical on host and device.
OPT_HD OPT_INLINE float u32_to_uniform(uint32_t bits) {
  return static_cast<float>((bits >> 8) + 1u) * (1.0f / 16777216.0f);
}

// A stream of uniform random numbers for one photon.
//
// Counter layout: {block, 0, photon_id_lo, photon_id_hi}; key = 64-bit seed.
// Each block yields 4 numbers; the block index is incremented as needed, so
// one photon can draw 4 * 2^32 numbers before its stream would wrap.
// The second counter word is reserved (e.g. for an event index).
//
// The 4 buffered values are kept in named members and shifted rather than
// indexed with a runtime index: on the GPU an array indexed by a runtime
// value is typically placed in (slow) local memory, named scalars stay in
// registers.
class PhiloxStream {
 public:
  OPT_HD PhiloxStream(uint64_t seed, uint64_t photon_id)
      : key_{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)},
        id_lo_(static_cast<uint32_t>(photon_id)),
        id_hi_(static_cast<uint32_t>(photon_id >> 32)) {}

  OPT_HD OPT_INLINE uint32_t next_u32() {
    if (available_ == 0) {
      buf_ = philox4x32_10(U32x4{block_++, 0u, id_lo_, id_hi_}, key_);
      available_ = 4;
    }
    const uint32_t r = buf_.x;
    buf_.x = buf_.y;
    buf_.y = buf_.z;
    buf_.z = buf_.w;
    --available_;
    return r;
  }

  // Uniform float in (0, 1].
  OPT_HD OPT_INLINE float uniform() { return u32_to_uniform(next_u32()); }

  // Number of 32-bit words consumed so far (for diagnostics / tests).
  OPT_HD uint32_t draws() const { return 4u * block_ - available_; }

 private:
  U32x2 key_;
  uint32_t id_lo_, id_hi_;
  uint32_t block_ = 0;
  uint32_t available_ = 0;
  U32x4 buf_{0u, 0u, 0u, 0u};
};

}  // namespace optphot
