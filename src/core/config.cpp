// SPDX-License-Identifier: MIT
#include "optphot/config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace optphot {

namespace {

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

float to_float(const std::string& key, const std::string& v) {
  if (v == "inf" || v == "INF" || v == "infinity") return kInfinity;
  try {
    std::size_t pos = 0;
    const float f = std::stof(v, &pos);
    if (pos != v.size()) throw std::invalid_argument(v);
    return f;
  } catch (const std::exception&) {
    throw std::runtime_error("bad number for '" + key + "': '" + v + "'");
  }
}

uint64_t to_u64(const std::string& key, const std::string& v) {
  try {
    std::size_t pos = 0;
    if (!v.empty() && v.find_first_not_of("0123456789") == std::string::npos) {
      return std::stoull(v, &pos);  // plain integer: full 64-bit range (seeds)
    }
    // Otherwise accept 1e6-style values, convenient for photon counts.
    const double d = std::stod(v, &pos);
    if (pos == v.size() && d >= 0.0 && d < 1.8e19) return static_cast<uint64_t>(d);
  } catch (const std::exception&) {
  }
  throw std::runtime_error("bad non-negative integer for '" + key + "': '" + v + "'");
}

std::string json_float(float f) {
  if (is_inf(f)) return "\"inf\"";
  std::ostringstream os;
  os.precision(9);
  os << f;
  return os.str();
}

}  // namespace

std::string surface_name(Surface s) {
  switch (s) {
    case Surface::Polished:
      return "polished";
    case Surface::Black:
      return "black";
    case Surface::Specular:
      return "specular";
    case Surface::Lambertian:
      return "lambertian";
  }
  return "?";
}

std::string source_name(SourceType s) { return s == SourceType::Point ? "point" : "volume"; }

std::string fate_name(Fate f) {
  switch (f) {
    case Fate::Detected:
      return "detected";
    case Fate::AbsorbedBulk:
      return "absorbed_bulk";
    case Fate::AbsorbedSurface:
      return "absorbed_surface";
    case Fate::Escaped:
      return "escaped";
    case Fate::MaxSteps:
      return "max_steps";
  }
  return "?";
}

void apply_setting(SimParams& p, RunOptions& o, const std::string& key, const std::string& v) {
  if (key == "size_x")
    p.hx = 0.5f * to_float(key, v);
  else if (key == "size_y")
    p.hy = 0.5f * to_float(key, v);
  else if (key == "size_z")
    p.hz = 0.5f * to_float(key, v);
  else if (key == "n_scint")
    p.n_scint = to_float(key, v);
  else if (key == "n_det")
    p.n_det = to_float(key, v);
  else if (key == "n_out")
    p.n_out = to_float(key, v);
  else if (key == "abs_length")
    p.abs_length = to_float(key, v);
  else if (key == "scat_length")
    p.scat_length = to_float(key, v);
  else if (key == "reflectivity")
    p.reflectivity = to_float(key, v);
  else if (key == "src_x")
    p.src_x = to_float(key, v);
  else if (key == "src_y")
    p.src_y = to_float(key, v);
  else if (key == "src_z")
    p.src_z = to_float(key, v);
  else if (key == "tau")
    p.tau = to_float(key, v);
  else if (key == "max_steps")
    p.max_steps = static_cast<uint32_t>(to_u64(key, v));
  else if (key == "seed")
    p.seed = to_u64(key, v);
  else if (key == "surface") {
    if (v == "polished")
      p.surface = Surface::Polished;
    else if (v == "black")
      p.surface = Surface::Black;
    else if (v == "specular")
      p.surface = Surface::Specular;
    else if (v == "lambertian")
      p.surface = Surface::Lambertian;
    else
      throw std::runtime_error("surface must be polished|black|specular|lambertian, got '" + v +
                               "'");
  } else if (key == "source") {
    if (v == "point")
      p.source = SourceType::Point;
    else if (v == "volume")
      p.source = SourceType::UniformVolume;
    else
      throw std::runtime_error("source must be point|volume, got '" + v + "'");
  } else if (key == "backend") {
    if (v != "cpu" && v != "gpu") throw std::runtime_error("backend must be cpu|gpu");
    o.backend = v;
  } else if (key == "mode") {
    if (v != "records" && v != "tally") throw std::runtime_error("mode must be records|tally");
    o.mode = v;
  } else if (key == "n_photons" || key == "n")
    o.n_photons = to_u64(key, v);
  else if (key == "threads")
    o.threads = static_cast<int>(to_u64(key, v));
  else if (key == "block_size")
    o.block_size = static_cast<int>(to_u64(key, v));
  else if (key == "out")
    o.out = v;
  else if (key == "time_bins")
    o.tally.time_bins = static_cast<uint32_t>(to_u64(key, v));
  else if (key == "time_max")
    o.tally.time_max = to_float(key, v);
  else if (key == "xy_bins")
    o.tally.xy_bins = static_cast<uint32_t>(to_u64(key, v));
  else
    throw std::runtime_error("unknown setting '" + key + "'");
}

void load_config_file(const std::string& path, SimParams& p, RunOptions& o) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config file '" + path + "'");
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": expected 'key = value'");
    }
    apply_setting(p, o, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
}

std::string usage() {
  return R"(usage: optphot [--config FILE] [--key value | --key=value]...

Run settings:
  backend     cpu | gpu                         (default cpu)
  mode        records | tally                   (default records)
  n_photons   number of photons, e.g. 1e6
  threads     CPU threads: 1 serial, 0 = all OpenMP threads
  block_size  GPU threads per block             (default 256)
  out         output directory (.npy arrays + meta.json); empty = none
  time_bins, time_max, xy_bins                  tally histogram binning

Physics settings (mm, ns):
  size_x size_y size_z    box dimensions        (default 50 50 50)
  n_scint n_det n_out     refractive indices    (default 1.63 1.63 1.0)
  abs_length scat_length  mean free paths, 'inf' disables (default 1000 inf)
  surface     polished | black | specular | lambertian (non-readout faces)
  reflectivity            for specular / lambertian (default 0.95)
  source      point | volume;  src_x src_y src_z  point-source position
  tau         scintillation decay time (default 4)
  max_steps   safety limit per photon (default 10000)
  seed        64-bit run seed
The readout face is +z.
)";
}

bool parse_command_line(int argc, char** argv, SimParams& p, RunOptions& o) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") return false;
    if (arg.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument '" + arg + "'");
    arg = arg.substr(2);
    std::string key, value;
    const auto eq = arg.find('=');
    if (eq != std::string::npos) {
      key = arg.substr(0, eq);
      value = arg.substr(eq + 1);
    } else {
      if (i + 1 >= argc) throw std::runtime_error("missing value for --" + arg);
      key = arg;
      value = argv[++i];
    }
    if (key == "config")
      load_config_file(value, p, o);
    else
      apply_setting(p, o, key, value);
  }
  return true;
}

std::string params_to_json(const SimParams& p, const RunOptions& o) {
  std::ostringstream os;
  os << "{\n"
     << "  \"size_x\": " << json_float(2 * p.hx) << ", \"size_y\": " << json_float(2 * p.hy)
     << ", \"size_z\": " << json_float(2 * p.hz) << ",\n"
     << "  \"n_scint\": " << json_float(p.n_scint) << ", \"n_det\": " << json_float(p.n_det)
     << ", \"n_out\": " << json_float(p.n_out) << ",\n"
     << "  \"abs_length\": " << json_float(p.abs_length)
     << ", \"scat_length\": " << json_float(p.scat_length) << ",\n"
     << "  \"surface\": \"" << surface_name(p.surface)
     << "\", \"reflectivity\": " << json_float(p.reflectivity) << ",\n"
     << "  \"source\": \"" << source_name(p.source) << "\", \"src_x\": " << json_float(p.src_x)
     << ", \"src_y\": " << json_float(p.src_y) << ", \"src_z\": " << json_float(p.src_z) << ",\n"
     << "  \"tau\": " << json_float(p.tau) << ", \"max_steps\": " << p.max_steps
     << ", \"seed\": " << p.seed << ",\n"
     << "  \"backend\": \"" << o.backend << "\", \"mode\": \"" << o.mode
     << "\", \"n_photons\": " << o.n_photons << ", \"threads\": " << o.threads
     << ", \"block_size\": " << o.block_size << ",\n"
     << "  \"time_bins\": " << o.tally.time_bins
     << ", \"time_max\": " << json_float(o.tally.time_max) << ", \"xy_bins\": " << o.tally.xy_bins
     << "\n"
     << "}";
  return os.str();
}

}  // namespace optphot
