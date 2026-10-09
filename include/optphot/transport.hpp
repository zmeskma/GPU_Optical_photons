// SPDX-License-Identifier: MIT
//
// transport.hpp - the complete history of one optical photon.
//
// transport_photon() is THE physics of this project. It is compiled for the
// host (CPU reference, called in a loop) and for the device (one CUDA thread
// per photon). Everything it needs is in SimParams plus the photon index, and
// all randomness comes from the photon's own Philox stream, so the same
// photon index gives the same history on any backend (up to the
// floating-point caveats documented in the README).
//
// Algorithm per step (history-based, "next event" tracking):
//   1. sample a distance to absorption and to Rayleigh scattering;
//   2. compute the distance to the box boundary along the current direction;
//   3. the smallest of the three wins: absorb, scatter or reach the boundary;
//   4. at the boundary apply the surface model of that face.
// Resampling both interaction distances at every step is exact because the
// exponential distribution is memoryless.
#pragma once

#include "optphot/fresnel.hpp"
#include "optphot/geometry.hpp"
#include "optphot/params.hpp"
#include "optphot/philox.hpp"
#include "optphot/sampling.hpp"

namespace optphot {

struct PhotonResult {
  Fate fate;
  uint32_t n_boundary;  // number of boundary interactions (incl. the final one)
  uint32_t n_scatter;   // number of Rayleigh scatterings
  float t;              // ns, time of the final interaction (incl. emission time)
  Vec3 pos;             // mm, position of the final interaction
  float path;           // mm, total path length travelled
};

// Bernoulli decision "event with probability prob happens" from u in (0, 1].
// Using <= (not <) makes prob = 1 certain and prob = 0 impossible: this matters
// for total internal reflection, where R is exactly 1.
OPT_HD OPT_INLINE bool happens(float u, float prob) { return u <= prob; }

// State of a photon in flight. The history-based kernels keep it in registers
// for the photon's whole life; the event-based kernel (src/gpu/gpu_event.cu)
// stores it in global memory between launches. Every completed step of an
// alive photon is either a scattering or a boundary reflection, so the number
// of steps done so far is n_boundary + n_scatter and needs no extra field.
struct PhotonState {
  Vec3 pos, dir;
  float t, path;
  uint32_t n_boundary, n_scatter;
};

// Emission: position, direction and emission time.
OPT_HD OPT_INLINE PhotonState start_photon(const SimParams& p, PhiloxStream& rng) {
  Vec3 pos;
  if (p.source == SourceType::Point) {
    pos = {p.src_x, p.src_y, p.src_z};
  } else {
    const float u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
    pos = sample_in_box(p.hx, p.hy, p.hz, u1, u2, u3);
  }
  const float ua = rng.uniform(), ub = rng.uniform();
  const Vec3 dir = sample_isotropic(ua, ub);
  const float t = sample_emission_time(p.tau, rng.uniform());
  return {pos, dir, t, 0.0f, 0u, 0u};
}

// One step of the transport loop. Returns true if the photon is still alive;
// otherwise sets `fate` and leaves s at the final interaction.
OPT_HD OPT_INLINE bool step_photon(const SimParams& p, PhiloxStream& rng, PhotonState& s,
                                   Fate& fate) {
  // Monochromatic and non-dispersive: phase and group velocity are both c/n.
  const float inv_speed = p.n_scint / kSpeedOfLight;  // ns / mm

  const float s_abs = sample_exponential(p.abs_length, rng.uniform());
  const float s_sca = sample_exponential(p.scat_length, rng.uniform());
  const BoxHit hit = distance_to_box_exit(s.pos, s.dir, p.hx, p.hy, p.hz);

  // --- Bulk absorption -------------------------------------------------------
  if (s_abs < hit.dist && s_abs <= s_sca) {
    s.pos = s.pos + s_abs * s.dir;
    s.t += s_abs * inv_speed;
    s.path += s_abs;
    fate = Fate::AbsorbedBulk;
    return false;
  }

  // --- Rayleigh scattering ---------------------------------------------------
  if (s_sca < hit.dist) {
    s.pos = s.pos + s_sca * s.dir;
    s.t += s_sca * inv_speed;
    s.path += s_sca;
    const float u1 = rng.uniform(), u2 = rng.uniform();
    s.dir = sample_rayleigh_direction(s.dir, u1, u2);
    ++s.n_scatter;
    return true;
  }

  // --- Boundary ----------------------------------------------------------------
  s.pos = snap_to_face(s.pos + hit.dist * s.dir, hit.face, p.hx, p.hy, p.hz);
  s.t += hit.dist * inv_speed;
  s.path += hit.dist;
  ++s.n_boundary;
  const Vec3 normal = face_normal(hit.face);  // outward
  const float cos_i = dot(s.dir, normal);     // > 0

  if (hit.face == kReadoutFace) {
    // Smooth optical interface to the photodetector side: transmitted
    // photons are detected, Fresnel-reflected ones continue.
    const FresnelResult f = fresnel_unpolarized(p.n_scint, p.n_det, cos_i);
    if (happens(rng.uniform(), f.reflectance)) {
      s.dir = reflect(s.dir, normal);
      return true;
    }
    fate = Fate::Detected;
    return false;
  }

  if (p.surface == Surface::Polished) {
    const FresnelResult f = fresnel_unpolarized(p.n_scint, p.n_out, cos_i);
    if (happens(rng.uniform(), f.reflectance)) {
      s.dir = reflect(s.dir, normal);
      return true;
    }
    fate = Fate::Escaped;  // refracted out; re-entry is not modelled
    return false;
  }
  if (p.surface == Surface::Black) {
    fate = Fate::AbsorbedSurface;
    return false;
  }
  // Specular or Lambertian reflector painted directly on the surface.
  if (!happens(rng.uniform(), p.reflectivity)) {
    fate = Fate::AbsorbedSurface;
    return false;
  }
  if (p.surface == Surface::Specular) {
    s.dir = reflect(s.dir, normal);
  } else {
    const float u1 = rng.uniform(), u2 = rng.uniform();
    s.dir = sample_lambertian(-normal, u1, u2);
  }
  return true;
}

OPT_HD OPT_INLINE PhotonResult finish_photon(const PhotonState& s, Fate fate) {
  return {fate, s.n_boundary, s.n_scatter, s.t, s.pos, s.path};
}

OPT_HD inline PhotonResult transport_photon(const SimParams& p, uint64_t photon_id) {
  PhiloxStream rng(p.seed, photon_id);
  PhotonState s = start_photon(p, rng);
  Fate fate = Fate::MaxSteps;
  for (uint32_t step = 0; step < p.max_steps; ++step) {
    if (!step_photon(p, rng, s, fate)) break;
  }
  return finish_photon(s, fate);
}

// ---------------------------------------------------------------------------
// Tally binning, shared by the CPU and GPU tally modes.

struct TallyConfig {
  uint32_t time_bins = 100;  // detected-photon arrival time histogram on [0, time_max)
  float time_max = 50.0f;    // ns; later arrivals go to an overflow bin (index time_bins)
  uint32_t xy_bins = 50;     // xy_bins x xy_bins hit map on the readout face
};

OPT_HD OPT_INLINE uint32_t time_bin(float t, const TallyConfig& c) {
  const float f = t * (static_cast<float>(c.time_bins) / c.time_max);
  return (f >= static_cast<float>(c.time_bins)) ? c.time_bins : static_cast<uint32_t>(f);
}

// Row-major index iy * xy_bins + ix of a hit at (x, y) on the readout face.
OPT_HD OPT_INLINE uint32_t xy_bin(float x, float y, const SimParams& p, const TallyConfig& c) {
  const float nb = static_cast<float>(c.xy_bins);
  const uint32_t ix = static_cast<uint32_t>(fminf((x + p.hx) / (2.0f * p.hx) * nb, nb - 1.0f));
  const uint32_t iy = static_cast<uint32_t>(fminf((y + p.hy) / (2.0f * p.hy) * nb, nb - 1.0f));
  return iy * c.xy_bins + ix;
}

}  // namespace optphot
