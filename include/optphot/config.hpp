// SPDX-License-Identifier: MIT
//
// config.hpp - run configuration from "key = value" files and --key value
// command-line overrides, and output of results for Python.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "optphot/backend.hpp"

namespace optphot {

struct RunOptions {
  std::string backend = "cpu";  // cpu | gpu
  std::string mode = "records";  // records | tally
  uint64_t n_photons = 1000000;
  int threads = 1;  // CPU: 1 serial, 0 all OpenMP threads
  int block_size = 256;  // GPU threads per block
  std::string out;  // output directory ("" = no output)
  TallyConfig tally;
};

// Applies one setting; throws std::runtime_error on unknown key / bad value.
// Geometry keys are full lengths: size_x, size_y, size_z (mm).
// Length keys accept "inf" (process disabled).
void apply_setting(SimParams& p, RunOptions& o, const std::string& key, const std::string& value);

// Reads "key = value" lines; '#' starts a comment.
void load_config_file(const std::string& path, SimParams& p, RunOptions& o);

// Parses argv: --config FILE (applied first, in order) and --key value or
// --key=value overrides. Returns false if --help was requested.
bool parse_command_line(int argc, char** argv, SimParams& p, RunOptions& o);
std::string usage();

std::string surface_name(Surface s);
std::string source_name(SourceType s);
std::string fate_name(Fate f);

// JSON object (as text) describing the parameters, for metadata files.
std::string params_to_json(const SimParams& p, const RunOptions& o);

// ---- Output (src/core/io.cpp) ----------------------------------------------
// Writes a 1-D or 2-D array as a NumPy .npy file (format version 1.0).
void write_npy(const std::string& path, const void* data, const std::string& descr,
               const std::vector<std::size_t>& shape, std::size_t item_size);

// Writes <dir>/{fate,n_boundary,n_scatter,t,x,y,z,path}.npy and <dir>/meta.json.
void write_records(const std::string& dir, const PhotonRecords& rec, const std::string& meta_json);
// Writes <dir>/{fate_counts,time_hist,xy_hist}.npy and <dir>/meta.json.
void write_tally(const std::string& dir, const Tally& tally, const std::string& meta_json);

void write_text_file(const std::string& path, const std::string& text);

}  // namespace optphot
