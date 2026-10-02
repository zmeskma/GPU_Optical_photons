"""Analytic reference values, implemented independently of the C++ code
(include/optphot/analytic.hpp) so the two can cross-check each other.

Coordinates: box [-hx,hx] x [-hy,hy] x [-hz,hz] (mm), readout face z = +hz.
"""
import numpy as np
from scipy import integrate


def rect_solid_angle(x1, x2, y1, y2, h):
    """Solid angle of the rectangle [x1,x2]x[y1,y2] in the plane z=h, seen
    from the origin (inclusion-exclusion over corner rectangles)."""
    def F(x, y):
        return np.arctan(x * y / (h * np.sqrt(x * x + y * y + h * h)))
    return F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1)


def readout_frame(half, src):
    """(h, x1, x2, y1, y2) of the readout face +z seen from the source."""
    hx, hy, hz = half
    sx, sy, sz = src
    return hz - sz, -hx - sx, hx - sx, -hy - sy, hy - sy


def detection_fraction(half, src):
    """Omega(readout) / 4 pi for a point source, black walls, matched readout."""
    h, x1, x2, y1, y2 = readout_frame(half, src)
    return rect_solid_angle(x1, x2, y1, y2, h) / (4 * np.pi)


def attenuated_fraction(half, src, length):
    """(1/4pi) Int_readout exp(-r/L) dOmega, with dOmega = h dA / r^3, by
    adaptive 2-D quadrature (scipy dblquad). length = inf -> solid angle."""
    h, x1, x2, y1, y2 = readout_frame(half, src)

    def f(y, x):
        r = np.sqrt(x * x + y * y + h * h)
        att = 1.0 if np.isinf(length) else np.exp(-r / length)
        return h / r**3 * att

    val, _ = integrate.dblquad(f, x1, x2, y1, y2, epsabs=1e-13, epsrel=1e-11)
    return val / (4 * np.pi)


def mirror_box_escape_fraction(n_scint, n_det):
    """1 - cos(theta_c): escape cone of a box with perfect specular walls."""
    if n_det >= n_scint:
        return 1.0
    return 1.0 - np.sqrt(1.0 - (n_det / n_scint) ** 2)


def binomial_pull(k, n, p):
    return (k - n * p) / np.sqrt(n * p * (1 - p))
