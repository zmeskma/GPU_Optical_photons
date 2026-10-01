// SPDX-License-Identifier: MIT
//
// geometry.hpp - ray / box navigation for the axis-aligned scintillator box
// [-hx,hx] x [-hy,hy] x [-hz,hz].
//
// Face numbering: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z (5 is the readout face).
//
// Robustness strategy (important in single precision): when a photon reaches
// a face it is *snapped* onto the face plane exactly and its other two
// coordinates are clamped into the face. The next distance computation only
// considers, per axis, the face the photon is moving towards, so a photon
// sitting on a face after reflection can never re-hit the same face with a
// zero-length step, and no "push off the surface by epsilon" is needed.
#pragma once

#include "optphot/vec3.hpp"

namespace optphot {

struct BoxHit {
  float dist;  // distance along the direction to the exit face (>= 0)
  int face;    // index of that face
};

OPT_HD OPT_INLINE BoxHit distance_to_box_exit(Vec3 p, Vec3 d, float hx, float hy, float hz) {
  BoxHit hit{kInfinity, -1};
  if (d.x != 0.0f) {
    const bool pos = d.x > 0.0f;
    const float t = ((pos ? hx : -hx) - p.x) / d.x;
    if (t < hit.dist) hit = {t, pos ? 1 : 0};
  }
  if (d.y != 0.0f) {
    const bool pos = d.y > 0.0f;
    const float t = ((pos ? hy : -hy) - p.y) / d.y;
    if (t < hit.dist) hit = {t, pos ? 3 : 2};
  }
  if (d.z != 0.0f) {
    const bool pos = d.z > 0.0f;
    const float t = ((pos ? hz : -hz) - p.z) / d.z;
    if (t < hit.dist) hit = {t, pos ? 5 : 4};
  }
  // A coordinate can lie a rounding error outside the box after a corner hit;
  // the resulting tiny negative distance means "on the face already".
  hit.dist = fmaxf(hit.dist, 0.0f);
  return hit;
}

// Outward unit normal of a face.
OPT_HD OPT_INLINE Vec3 face_normal(int face) {
  const float s = (face & 1) ? 1.0f : -1.0f;
  const int axis = face >> 1;
  return {axis == 0 ? s : 0.0f, axis == 1 ? s : 0.0f, axis == 2 ? s : 0.0f};
}

// Puts `p` exactly on the plane of `face` and clamps the other coordinates
// into the face rectangle.
OPT_HD OPT_INLINE Vec3 snap_to_face(Vec3 p, int face, float hx, float hy, float hz) {
  const float s = (face & 1) ? 1.0f : -1.0f;
  const int axis = face >> 1;
  return {axis == 0 ? s * hx : fminf(fmaxf(p.x, -hx), hx),
          axis == 1 ? s * hy : fminf(fmaxf(p.y, -hy), hy),
          axis == 2 ? s * hz : fminf(fmaxf(p.z, -hz), hz)};
}

}  // namespace optphot
