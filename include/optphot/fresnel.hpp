// SPDX-License-Identifier: MIT
//
// fresnel.hpp - unpolarised Fresnel reflection/refraction at a smooth
// dielectric interface, and the specular reflection / refraction directions.
//
// Conventions: `normal` is the unit normal pointing from medium 1 (where the
// photon is) into medium 2, so for an incoming photon cos_i = dot(dir, normal)
// is > 0.
#pragma once

#include "optphot/vec3.hpp"

namespace optphot {

struct FresnelResult {
  float reflectance;  // R, unpolarised: (Rs + Rp) / 2. 1 for total internal reflection.
  float cos_t;        // cosine of the refraction angle (0 if TIR)
};

// Unpolarised Fresnel reflectance for light going from index n1 to n2 with
// incidence-angle cosine cos_i in [0, 1].
//
//   rs = (n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t)
//   rp = (n2 cos_i - n1 cos_t) / (n2 cos_i + n1 cos_t)
//   R  = (rs^2 + rp^2) / 2
//
// Unpolarised light is modelled as an incoherent 50/50 mixture of s and p;
// the polarisation state is *not* tracked after the interaction (a real
// simplification: after one reflection the light is partially polarised).
OPT_HD OPT_INLINE FresnelResult fresnel_unpolarized(float n1, float n2, float cos_i) {
  // Index-matched: there is no optical interface. Handled explicitly because
  // for grazing photons 1 - cos_i^2 rounds to 1 in float, which would
  // otherwise be mistaken for total internal reflection.
  if (n1 == n2) {
    return {0.0f, cos_i};
  }
  const float eta = n1 / n2;
  const float sin2_t = eta * eta * fmaxf(0.0f, 1.0f - cos_i * cos_i);
  if (sin2_t >= 1.0f) {
    return {1.0f, 0.0f};  // total internal reflection
  }
  const float cos_t = sqrtf(1.0f - sin2_t);
  const float rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
  const float rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
  return {0.5f * (rs * rs + rp * rp), cos_t};
}

// Specular reflection of `dir` about a surface with unit normal `normal`.
OPT_HD OPT_INLINE Vec3 reflect(Vec3 dir, Vec3 normal) {
  return dir - (2.0f * dot(dir, normal)) * normal;
}

// Refracted direction (Snell's law, vector form) for a photon crossing from
// n1 into n2 = n1 / eta. Requires no TIR, i.e. cos_t from fresnel_unpolarized.
//   t = eta * d + (cos_t - eta * cos_i) * N
OPT_HD OPT_INLINE Vec3 refract(Vec3 dir, Vec3 normal, float eta, float cos_i, float cos_t) {
  return normalize(eta * dir + (cos_t - eta * cos_i) * normal);
}

}  // namespace optphot
