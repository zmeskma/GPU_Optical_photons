// SPDX-License-Identifier: MIT
//
// analytic.hpp - analytic / quadrature reference values used to validate the
// transport (host only, double precision).
#pragma once

#include <cmath>
#include <vector>

#include "optphot/params.hpp"

namespace optphot::analytic {

constexpr double kPi = 3.14159265358979323846;

// Solid angle subtended at the origin by the rectangle [x1,x2] x [y1,y2] lying
// in the plane z = h > 0:
//   Omega = F(x2,y2) - F(x1,y2) - F(x2,y1) + F(x1,y1),
//   F(x, y) = atan( x y / (h sqrt(x^2 + y^2 + h^2)) ).
// F(x, y) is the solid angle of the rectangle [0,x] x [0,y] (signed), so the
// formula is the inclusion-exclusion over the four corner rectangles.
inline double rect_solid_angle(double x1, double x2, double y1, double y2, double h) {
  auto F = [h](double x, double y) { return std::atan(x * y / (h * std::sqrt(x * x + y * y + h * h))); };
  return F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1);
}

// A face of the box seen from the (point) source: perpendicular distance h
// from the source to the face plane and the face extent [u1,u2] x [v1,v2] in
// in-plane coordinates centred on the foot of the perpendicular.
struct FaceFrame {
  double h, u1, u2, v1, v2;
};

inline FaceFrame face_frame(const SimParams& p, int face) {
  const double half[3] = {p.hx, p.hy, p.hz};
  const double src[3] = {p.src_x, p.src_y, p.src_z};
  const int a = face >> 1;
  const double s = (face & 1) ? 1.0 : -1.0;
  const int ua = (a + 1) % 3, va = (a + 2) % 3;
  return {half[a] - s * src[a], -half[ua] - src[ua], half[ua] - src[ua], -half[va] - src[va],
          half[va] - src[va]};
}

// Fraction of isotropically emitted photons from the point source whose
// straight line hits `face`: Omega / (4 pi).
inline double face_solid_angle_fraction(const SimParams& p, int face) {
  const FaceFrame f = face_frame(p, face);
  return rect_solid_angle(f.u1, f.u2, f.v1, f.v2, f.h) / (4.0 * kPi);
}

// Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n).
inline void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(kPi * (i + 0.75) / (n + 0.5)), pp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 0; j < n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      const double dz = p1 / pp;
      z -= dz;
      if (std::fabs(dz) < 1e-15) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
}

// (1 / 4 pi) * Integral over `face` of exp(-r / L) dOmega, with
// dOmega = h dA / r^3 and r the distance from the source. This is the
// probability that a photon is emitted towards the face AND survives a bulk
// process of mean free path L along the straight line to it. L = inf gives
// the solid-angle fraction (used to check the quadrature itself).
//
// Composite Gauss-Legendre with panel edges at the foot of the perpendicular
// (where the integrand peaks when the source is close to the face).
inline double face_survival_fraction(const SimParams& p, int face, double L, int panels = 24,
                                     int order = 16) {
  const FaceFrame f = face_frame(p, face);
  std::vector<double> gx, gw;
  gauss_legendre(order, gx, gw);
  auto edges = [panels](double a, double b) {
    std::vector<double> e;
    // Split at 0 (foot point) if inside, then uniform panels on each side.
    const std::vector<double> cuts = (a < 0.0 && b > 0.0) ? std::vector<double>{a, 0.0, b}
                                                          : std::vector<double>{a, b};
    for (std::size_t k = 0; k + 1 < cuts.size(); ++k) {
      for (int i = 0; i < panels; ++i) e.push_back(cuts[k] + (cuts[k + 1] - cuts[k]) * i / panels);
    }
    e.push_back(b);
    return e;
  };
  const std::vector<double> eu = edges(f.u1, f.u2), ev = edges(f.v1, f.v2);
  double sum = 0.0;
  for (std::size_t i = 0; i + 1 < eu.size(); ++i) {
    const double cu = 0.5 * (eu[i] + eu[i + 1]), hu = 0.5 * (eu[i + 1] - eu[i]);
    for (std::size_t j = 0; j + 1 < ev.size(); ++j) {
      const double cv = 0.5 * (ev[j] + ev[j + 1]), hv = 0.5 * (ev[j + 1] - ev[j]);
      for (int a = 0; a < order; ++a) {
        const double u = cu + hu * gx[a];
        for (int b = 0; b < order; ++b) {
          const double v = cv + hv * gx[b];
          const double r = std::sqrt(u * u + v * v + f.h * f.h);
          const double att = std::isinf(L) ? 1.0 : std::exp(-r / L);
          sum += gw[a] * gw[b] * hu * hv * f.h / (r * r * r) * att;
        }
      }
    }
  }
  return sum / (4.0 * kPi);
}

// Probability that a photon from the point source reaches ANY face without a
// bulk interaction of mean free path L (sum over the six faces).
inline double box_survival_fraction(const SimParams& p, double L) {
  double s = 0.0;
  for (int face = 0; face < 6; ++face) s += face_survival_fraction(p, face, L);
  return s;
}

// Fraction of photons that can escape through the readout face of a box whose
// other walls are perfect specular mirrors, with no bulk absorption.
// Specular reflection on axis-aligned walls preserves |d_x|, |d_y|, |d_z|, so a
// photon can only ever leave through +z if |d_z| > cos(theta_c); for those,
// repeated Fresnel attempts eventually succeed. |d_z| is uniform on [0, 1] for
// isotropic emission, hence P = 1 - cos(theta_c) = 1 - sqrt(1 - (n_det/n_scint)^2).
inline double mirror_box_escape_fraction(double n_scint, double n_det) {
  if (n_det >= n_scint) return 1.0;
  const double s = n_det / n_scint;
  return 1.0 - std::sqrt(1.0 - s * s);
}

}  // namespace optphot::analytic
