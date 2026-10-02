// SPDX-License-Identifier: MIT
// Tests of the box navigation (distance to exit face, normals, snapping).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "optphot/fresnel.hpp"
#include "optphot/geometry.hpp"
#include "optphot/philox.hpp"
#include "optphot/sampling.hpp"

using namespace optphot;
using Catch::Matchers::WithinAbs;

TEST_CASE("Distance to exit along the axes", "[geometry]") {
  const float hx = 10, hy = 20, hz = 30;
  const Vec3 o{0, 0, 0};
  const struct {
    Vec3 d;
    int face;
    float dist;
  } cases[] = {{{1, 0, 0}, 1, hx},  {{-1, 0, 0}, 0, hx}, {{0, 1, 0}, 3, hy},
               {{0, -1, 0}, 2, hy}, {{0, 0, 1}, 5, hz},  {{0, 0, -1}, 4, hz}};
  for (const auto& c : cases) {
    const BoxHit h = distance_to_box_exit(o, c.d, hx, hy, hz);
    CHECK(h.face == c.face);
    CHECK(h.dist == c.dist);
    // The outward normal of the face is the direction itself.
    const Vec3 n = face_normal(h.face);
    CHECK(dot(n, c.d) == 1.0f);
  }
}

TEST_CASE("Distance to exit for an oblique ray", "[geometry]") {
  // From (5, 0, 0) towards (+1, +1, 0)/sqrt2 in a 10x20x30 half-size box:
  // reaches x = 10 after 5*sqrt2, before y = 20.
  const Vec3 d = normalize(Vec3{1, 1, 0});
  const BoxHit h = distance_to_box_exit({5, 0, 0}, d, 10, 20, 30);
  CHECK(h.face == 1);
  CHECK_THAT(h.dist, WithinAbs(5.0 * std::sqrt(2.0), 1e-5));
}

TEST_CASE("Random rays end on the reported face, inside the box", "[geometry]") {
  PhiloxStream rng(11u, 0u);
  const float hx = 25, hy = 15, hz = 40;
  for (int i = 0; i < 100000; ++i) {
    const Vec3 p = sample_in_box(hx, hy, hz, rng.uniform(), rng.uniform(), rng.uniform());
    const Vec3 d = sample_isotropic(rng.uniform(), rng.uniform());
    const BoxHit h = distance_to_box_exit(p, d, hx, hy, hz);
    REQUIRE(h.face >= 0);
    REQUIRE(h.dist >= 0.0f);
    const Vec3 q = p + h.dist * d;
    const Vec3 s = snap_to_face(q, h.face, hx, hy, hz);
    // Snapping moves the end point by at most a few ulps.
    CHECK(norm(s - q) < 1e-4f);
    CHECK(std::fabs(s.x) <= hx);
    CHECK(std::fabs(s.y) <= hy);
    CHECK(std::fabs(s.z) <= hz);
    CHECK(dot(face_normal(h.face), d) > 0.0f);  // leaving through that face
  }
}

TEST_CASE("After reflection on a face the photon does not re-hit it", "[geometry]") {
  // Many consecutive specular reflections in a closed mirror box: every step
  // must have strictly positive length (except at exact corners) and the
  // photon must stay inside the box.
  PhiloxStream rng(12u, 0u);
  const float hx = 25, hy = 25, hz = 25;
  for (int photon = 0; photon < 200; ++photon) {
    Vec3 p{0, 0, 0};
    Vec3 d = sample_isotropic(rng.uniform(), rng.uniform());
    int previous_face = -1;
    for (int step = 0; step < 500; ++step) {
      const BoxHit h = distance_to_box_exit(p, d, hx, hy, hz);
      REQUIRE(h.face != previous_face);
      p = snap_to_face(p + h.dist * d, h.face, hx, hy, hz);
      d = reflect(d, face_normal(h.face));
      previous_face = h.face;
      REQUIRE(std::fabs(p.x) <= hx);
      REQUIRE(std::fabs(p.y) <= hy);
      REQUIRE(std::fabs(p.z) <= hz);
      REQUIRE_THAT(norm(d), WithinAbs(1.0, 1e-5));
    }
  }
}

TEST_CASE("snap_to_face clamps the in-plane coordinates", "[geometry]") {
  const Vec3 s = snap_to_face({10.00001f, -20.5f, 3.0f}, 1, 10, 20, 30);
  CHECK(s.x == 10.0f);
  CHECK(s.y == -20.0f);
  CHECK(s.z == 3.0f);
}
