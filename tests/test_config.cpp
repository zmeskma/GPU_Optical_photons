// SPDX-License-Identifier: MIT
// Tests of configuration parsing and .npy output.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "optphot/config.hpp"

using namespace optphot;

TEST_CASE("Settings are parsed and validated", "[config]") {
  SimParams p;
  RunOptions o;
  apply_setting(p, o, "size_x", "30");
  apply_setting(p, o, "abs_length", "inf");
  apply_setting(p, o, "surface", "lambertian");
  apply_setting(p, o, "n_photons", "1e7");
  apply_setting(p, o, "seed", "18446744073709551615");
  CHECK(p.hx == 15.0f);
  CHECK(is_inf(p.abs_length));
  CHECK(p.surface == Surface::Lambertian);
  CHECK(o.n_photons == 10000000u);
  CHECK(p.seed == 18446744073709551615ull);
  CHECK_THROWS_AS(apply_setting(p, o, "nonsense", "1"), std::runtime_error);
  CHECK_THROWS_AS(apply_setting(p, o, "n_scint", "1.6x"), std::runtime_error);
  CHECK_THROWS_AS(apply_setting(p, o, "surface", "shiny"), std::runtime_error);
  CHECK_THROWS_AS(apply_setting(p, o, "n_photons", "-5"), std::runtime_error);
}

TEST_CASE("Config files and command-line overrides", "[config]") {
  const std::string path = "test_config_tmp.cfg";
  {
    std::ofstream f(path);
    f << "# comment\nsurface = black   # trailing comment\n\n n_det=1.5\nn_photons = 1000\n";
  }
  SimParams p;
  RunOptions o;
  const char* argv[] = {"optphot", "--config", path.c_str(), "--n_det", "1.4", "--tau=0"};
  REQUIRE(parse_command_line(6, const_cast<char**>(argv), p, o));
  CHECK(p.surface == Surface::Black);
  CHECK(p.n_det == 1.4f);  // command line wins over the file (applied later)
  CHECK(p.tau == 0.0f);
  CHECK(o.n_photons == 1000u);
  std::filesystem::remove(path);
}

TEST_CASE("NPY writer produces an aligned, well-formed header", "[config]") {
  const std::string path = "test_tmp.npy";
  const float data[3] = {1.0f, 2.0f, 3.0f};
  write_npy(path, data, "<f4", {3}, 4);
  std::ifstream f(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  f.close();
  std::filesystem::remove(path);
  REQUIRE(bytes.size() >= 10);
  CHECK(bytes.substr(1, 5) == "NUMPY");
  const std::size_t hlen = static_cast<unsigned char>(bytes[8]) | (static_cast<unsigned char>(bytes[9]) << 8);
  CHECK((10 + hlen) % 64 == 0);
  CHECK(bytes.size() == 10 + hlen + sizeof(data));
  CHECK(bytes.find("'shape': (3,)") != std::string::npos);
  CHECK(bytes[10 + hlen - 1] == '\n');
}
