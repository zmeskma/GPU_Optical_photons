// SPDX-License-Identifier: MIT
//
// vec3.hpp - minimal 3-vector of floats usable on host and device.
//
// Single precision is used everywhere: consumer and cloud GPUs (e.g. the T4
// on Colab) have 1/32-rate FP64, and optical-photon transport in a 5 cm box
// needs nowhere near double precision (float eps * 50 mm ~ 3 nm).
#pragma once

#include "optphot/hd.hpp"

namespace optphot {

struct Vec3 {
  float x, y, z;
};

OPT_HD OPT_INLINE Vec3 make_vec3(float x, float y, float z) { return Vec3{x, y, z}; }
OPT_HD OPT_INLINE Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
OPT_HD OPT_INLINE Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
OPT_HD OPT_INLINE Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
OPT_HD OPT_INLINE Vec3 operator*(float s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
OPT_HD OPT_INLINE Vec3 operator*(Vec3 a, float s) { return {s * a.x, s * a.y, s * a.z}; }
OPT_HD OPT_INLINE float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
OPT_HD OPT_INLINE Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
OPT_HD OPT_INLINE float norm(Vec3 a) { return sqrtf(dot(a, a)); }
OPT_HD OPT_INLINE Vec3 normalize(Vec3 a) { return (1.0f / norm(a)) * a; }

// Builds an orthonormal basis (t1, t2) perpendicular to the unit vector n,
// without branches on the hot path. Duff et al., "Building an Orthonormal
// Basis, Revisited", JCGT 6(1), 2017.
OPT_HD OPT_INLINE void orthonormal_basis(Vec3 n, Vec3& t1, Vec3& t2) {
  const float sign = copysignf(1.0f, n.z);
  const float a = -1.0f / (sign + n.z);
  const float b = n.x * n.y * a;
  t1 = {1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
  t2 = {b, sign + n.y * n.y * a, -n.y};
}

// Returns the unit vector with polar cosine `cos_theta` and azimuth `phi`
// measured around the unit vector `axis`.
OPT_HD OPT_INLINE Vec3 rotate_about(Vec3 axis, float cos_theta, float phi) {
  Vec3 t1, t2;
  orthonormal_basis(axis, t1, t2);
  const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
  const Vec3 d = (sin_theta * cosf(phi)) * t1 + (sin_theta * sinf(phi)) * t2 + cos_theta * axis;
  return normalize(d);
}

}  // namespace optphot
