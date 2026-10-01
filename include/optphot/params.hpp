// SPDX-License-Identifier: MIT
//
// params.hpp - plain-old-data description of one simulation setup.
//
// SimParams is trivially copyable on purpose: it is passed *by value* as a
// CUDA kernel argument (kernel arguments live in the constant bank, so every
// thread reads the same parameters through a broadcast-friendly cache).
#pragma once

#include "optphot/hd.hpp"

namespace optphot {

// Surface model applied to the five non-readout faces of the box.
enum class Surface : uint32_t {
  Polished = 0,    // Fresnel interface to an outside medium of index n_out
  Black = 1,       // perfect absorber
  Specular = 2,    // mirror with reflectivity R (no air gap), else absorbed
  Lambertian = 3,  // diffuse reflector with reflectivity R, else absorbed
};

enum class SourceType : uint32_t {
  Point = 0,         // all photons start at (src_x, src_y, src_z)
  UniformVolume = 1  // uniform in the whole box
};

// Final state of a photon history.
enum class Fate : uint8_t {
  Detected = 0,         // transmitted through the readout face
  AbsorbedBulk = 1,     // absorbed in the scintillator bulk
  AbsorbedSurface = 2,  // absorbed at a black / reflector face
  Escaped = 3,          // transmitted through a polished non-readout face
  MaxSteps = 4,         // safety limit on the number of steps reached
};
constexpr int kNumFates = 5;

// Face numbering: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z. The readout face is +z.
constexpr int kReadoutFace = 5;

struct SimParams {
  // Geometry: box [-hx,hx] x [-hy,hy] x [-hz,hz], in mm.
  float hx = 25.0f, hy = 25.0f, hz = 25.0f;

  // Optical properties (monochromatic).
  float n_scint = 1.63f;          // scintillator refractive index
  float n_det = 1.63f;            // readout-side medium (photodetector window / grease)
  float n_out = 1.0f;             // outside medium for Surface::Polished
  float abs_length = 1000.0f;     // mm; kInfinity disables absorption
  float scat_length = kInfinity;  // mm; Rayleigh scattering length, kInfinity disables

  Surface surface = Surface::Polished;
  float reflectivity = 0.95f;  // for Specular / Lambertian

  // Source.
  SourceType source = SourceType::Point;
  float src_x = 0.0f, src_y = 0.0f, src_z = 0.0f;  // mm
  float tau = 4.0f;                                // ns, scintillation decay time (0 = prompt)

  // Safety limit on the number of transport steps (bulk + boundary interactions).
  uint32_t max_steps = 10000;

  uint64_t seed = 12345;
};

}  // namespace optphot
