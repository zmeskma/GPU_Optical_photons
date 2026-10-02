// SPDX-License-Identifier: MIT
//
// pmath.hpp - the few transcendental functions the transport needs:
// log (exponential sampling), sin/cos of an angle given in turns (azimuths)
// and cbrt (Rayleigh sampling).
//
// Why this file exists: +, -, *, / and sqrt are correctly rounded by IEEE 754
// on both the CPU and the GPU, so they give bit-identical results everywhere.
// logf / sinf / cosf / cbrtf are NOT: glibc, the MSVC/MinGW runtimes and
// CUDA's libdevice each have their own implementations with errors of up to
// 1-2 ulp, which differ in the last bit for some inputs. A one-ulp difference
// is harmless physically, but it eventually flips a discrete decision
// (u <= R, which face is hit first) and the CPU and GPU histories of that
// photon diverge.
//
// Two implementations are provided, selected at compile time:
//   * default: the platform math library (logf, sinf, cosf, cbrtf);
//   * OPTPHOT_PORTABLE_MATH: implementations below that only use correctly
//     rounded operations. Combined with disabled FMA contraction on both
//     sides (host: -ffp-contract=off, always; device: nvcc --fmad=false via
//     -DOPTPHOT_CUDA_FMAD=OFF) the whole history becomes bit-reproducible
//     between CPU and GPU. They are accurate to a few ulp, which is plenty
//     for Monte Carlo sampling, but a bit slower than the hardware-tuned
//     library versions.
//
// The portable versions are always compiled (and unit-tested); the opt_*
// wrappers choose which one the physics uses.
#pragma once

#include <cstring>

#include "optphot/hd.hpp"

namespace optphot {

namespace portable {

OPT_HD OPT_INLINE uint32_t float_bits(float x) {
#if OPT_DEVICE_PASS
  return __float_as_uint(x);
#else
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
#endif
}

OPT_HD OPT_INLINE float bits_float(uint32_t u) {
#if OPT_DEVICE_PASS
  return __uint_as_float(u);
#else
  float x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
#endif
}

// Natural log for positive, normal, finite x (all we need: x is a uniform
// deviate in (0, 1]).
// x = 2^e * m with m in [sqrt(1/2), sqrt(2)); log(m) = 2 atanh(s) with
// s = (m - 1) / (m + 1), |s| <= 0.1716, expanded to s^13.
OPT_HD OPT_INLINE float log(float x) {
  const uint32_t bits = float_bits(x);
  int e = static_cast<int>((bits >> 23) & 0xffu) - 127;
  float m = bits_float((bits & 0x007fffffu) | 0x3f800000u);  // [1, 2)
  if (m > 1.41421356f) {
    m = 0.5f * m;
    ++e;
  }
  const float s = (m - 1.0f) / (m + 1.0f);
  const float s2 = s * s;
  const float poly =
      1.0f + s2 * (1.0f / 3.0f +
                   s2 * (1.0f / 5.0f + s2 * (1.0f / 7.0f + s2 * (1.0f / 9.0f + s2 * (1.0f / 11.0f)))));
  // ln 2 split as in fdlibm's logf: the high part has trailing zero bits so
  // that e * ln2_hi is exact; ln2_lo is the correction.
  const float ln2_hi = 6.9313812256e-01f;  // 0x3f317180
  const float ln2_lo = 9.0580006145e-06f;  // 0x3717f7d1
  const float fe = static_cast<float>(e);
  return fe * ln2_hi + (2.0f * s * poly + fe * ln2_lo);
}

// sin(2 pi u) and cos(2 pi u). The range reduction is exact: 4u is exact,
// and so are its integer part q and remainder r (Sterbenz), so the polynomial
// only sees x = (pi/2) r with |r| <= 1/2, i.e. |x| <= pi/4.
OPT_HD OPT_INLINE void sincos_turns(float u, float& s, float& c) {
  const float v = 4.0f * u;
  float q = floorf(v);
  float r = v - q;
  if (r > 0.5f) {
    r -= 1.0f;
    q += 1.0f;
  }
  const float x = r * 1.57079632679489662f;
  const float x2 = x * x;
  const float sp =
      x * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f + x2 * (-1.0f / 5040.0f + x2 * (1.0f / 362880.0f)))));
  const float cp =
      1.0f + x2 * (-0.5f + x2 * (1.0f / 24.0f +
                                x2 * (-1.0f / 720.0f + x2 * (1.0f / 40320.0f + x2 * (-1.0f / 3628800.0f)))));
  // Rotate by q quarter turns.
  const int quadrant = static_cast<int>(q) & 3;
  if (quadrant == 0) {
    s = sp;
    c = cp;
  } else if (quadrant == 1) {
    s = cp;
    c = -sp;
  } else if (quadrant == 2) {
    s = -sp;
    c = -cp;
  } else {
    s = -cp;
    c = sp;
  }
}

// Cube root for positive, normal x: exponent-based first guess, then Newton
// iterations y <- y - (y^3 - x) / (3 y^2) (quadratic convergence).
OPT_HD OPT_INLINE float cbrt(float x) {
  // Divide the biased exponent by 3 (integer arithmetic: exact everywhere).
  float y = bits_float(float_bits(x) / 3u + 0x2a514067u);
  for (int i = 0; i < 4; ++i) {
    y = y - (y * y * y - x) / (3.0f * y * y);
  }
  return y;
}

}  // namespace portable

OPT_HD OPT_INLINE float opt_log(float x) {
#if defined(OPTPHOT_PORTABLE_MATH)
  return portable::log(x);
#else
  return logf(x);
#endif
}

OPT_HD OPT_INLINE void opt_sincos_turns(float u, float& s, float& c) {
#if defined(OPTPHOT_PORTABLE_MATH)
  portable::sincos_turns(u, s, c);
#else
  const float phi = kTwoPi * u;
  s = sinf(phi);
  c = cosf(phi);
#endif
}

OPT_HD OPT_INLINE float opt_cbrt(float x) {
#if defined(OPTPHOT_PORTABLE_MATH)
  return portable::cbrt(x);
#else
  return cbrtf(x);
#endif
}

}  // namespace optphot
