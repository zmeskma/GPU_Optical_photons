// SPDX-License-Identifier: MIT
//
// sampling.hpp - elementary Monte Carlo sampling routines.
//
// Each function takes the uniform deviates it needs as arguments instead of a
// generator. That keeps the functions pure (easy to unit-test with chosen
// inputs) and makes the number of random numbers consumed per call explicit.
// All uniforms are in (0, 1] (see u32_to_uniform in philox.hpp).
#pragma once

#include "optphot/vec3.hpp"

namespace optphot {

// Distance to the next interaction for a process with mean free path
// `length`; kInfinity means "process disabled".
OPT_HD OPT_INLINE float sample_exponential(float length, float u) {
  return is_inf(length) ? kInfinity : -length * logf(u);
}

// Isotropic unit vector: cos(theta) uniform in [-1, 1], phi uniform.
OPT_HD OPT_INLINE Vec3 sample_isotropic(float u1, float u2) {
  const float cos_t = 2.0f * u1 - 1.0f;
  const float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
  const float phi = kTwoPi * u2;
  return {sin_t * cosf(phi), sin_t * sinf(phi), cos_t};
}

// Cosine of the scattering angle for unpolarised Rayleigh scattering,
// p(mu) = 3/8 (1 + mu^2) on [-1, 1].
//
// The CDF is F(mu) = (mu^3 + 3 mu + 4) / 8, so F(mu) = u is the depressed
// cubic mu^3 + 3 mu + (4 - 8u) = 0. Its discriminant is positive, so Cardano
// gives the single real root in closed form:
//   mu = A - 1/A,  A = cbrt(q + sqrt(q^2 + 1)),  q = 4u - 2.
// (For q < 0 we use the equivalent form A = 1 / cbrt(sqrt(q^2+1) - q) to avoid
// cancellation.) No rejection loop -> no warp divergence on the GPU.
OPT_HD OPT_INLINE float sample_rayleigh_cos(float u) {
  const float q = 4.0f * u - 2.0f;
  const float s = sqrtf(q * q + 1.0f);
  const float a = (q >= 0.0f) ? cbrtf(q + s) : 1.0f / cbrtf(s - q);
  return fminf(1.0f, fmaxf(-1.0f, a - 1.0f / a));
}

// New direction after Rayleigh scattering of a photon travelling along `dir`.
OPT_HD OPT_INLINE Vec3 sample_rayleigh_direction(Vec3 dir, float u1, float u2) {
  return rotate_about(dir, sample_rayleigh_cos(u1), kTwoPi * u2);
}

// Lambertian (cosine-weighted) direction in the hemisphere around the unit
// vector `normal`: p(cos t) d(cos t) = 2 cos t d(cos t), i.e. cos t = sqrt(u).
OPT_HD OPT_INLINE Vec3 sample_lambertian(Vec3 normal, float u1, float u2) {
  return rotate_about(normal, sqrtf(u1), kTwoPi * u2);
}

// Emission time from a single-exponential decay with time constant tau (ns).
OPT_HD OPT_INLINE float sample_emission_time(float tau, float u) {
  return tau > 0.0f ? -tau * logf(u) : 0.0f;
}

// Uniform point in the box [-hx,hx] x [-hy,hy] x [-hz,hz].
OPT_HD OPT_INLINE Vec3 sample_in_box(float hx, float hy, float hz, float u1, float u2, float u3) {
  return {hx * (2.0f * u1 - 1.0f), hy * (2.0f * u2 - 1.0f), hz * (2.0f * u3 - 1.0f)};
}

}  // namespace optphot
