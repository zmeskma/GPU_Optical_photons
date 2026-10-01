// SPDX-License-Identifier: MIT
//
// Output in NumPy's .npy format: a tiny self-describing binary format that
// np.load() reads directly, so no converter or extra dependency is needed.
// Format spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "optphot/config.hpp"

namespace optphot {

void write_npy(const std::string& path, const void* data, const std::string& descr,
               const std::vector<std::size_t>& shape, std::size_t item_size) {
  std::string shape_str = "(";
  std::size_t count = 1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) shape_str += ", ";
    shape_str += std::to_string(shape[i]);
    count *= shape[i];
  }
  shape_str += (shape.size() == 1) ? ",)" : ")";  // Python 1-tuple syntax
  std::string header =
      "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  // magic (6) + version (2) + header length (2) + header, padded with spaces
  // and terminated by '\n' so that the data starts at a multiple of 64 bytes.
  const std::size_t unpadded = 10 + header.size() + 1;
  header.append((64 - unpadded % 64) % 64, ' ');
  header.push_back('\n');

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write '" + path + "'");
  const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  f.write(magic, sizeof(magic));
  const uint16_t hlen = static_cast<uint16_t>(header.size());
  const char hlen_le[2] = {static_cast<char>(hlen & 0xff), static_cast<char>(hlen >> 8)};
  f.write(hlen_le, 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * item_size));
}

void write_text_file(const std::string& path, const std::string& text) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write '" + path + "'");
  f << text << "\n";
}

// All multi-byte arrays are written in native byte order, declared
// little-endian ('<'): every platform this code targets (x86-64, ARM64, CUDA
// hosts) is little-endian.
void write_records(const std::string& dir, const PhotonRecords& r, const std::string& meta) {
  std::filesystem::create_directories(dir);
  const std::vector<std::size_t> shape{r.size()};
  write_npy(dir + "/fate.npy", r.fate.data(), "|u1", shape, 1);
  write_npy(dir + "/n_boundary.npy", r.n_boundary.data(), "<u4", shape, 4);
  write_npy(dir + "/n_scatter.npy", r.n_scatter.data(), "<u4", shape, 4);
  write_npy(dir + "/t.npy", r.t.data(), "<f4", shape, 4);
  write_npy(dir + "/x.npy", r.x.data(), "<f4", shape, 4);
  write_npy(dir + "/y.npy", r.y.data(), "<f4", shape, 4);
  write_npy(dir + "/z.npy", r.z.data(), "<f4", shape, 4);
  write_npy(dir + "/path.npy", r.path.data(), "<f4", shape, 4);
  write_text_file(dir + "/meta.json", meta);
}

void write_tally(const std::string& dir, const Tally& t, const std::string& meta) {
  std::filesystem::create_directories(dir);
  write_npy(dir + "/fate_counts.npy", t.fate_counts.data(), "<u8", {t.fate_counts.size()}, 8);
  write_npy(dir + "/time_hist.npy", t.time_hist.data(), "<u8", {t.time_hist.size()}, 8);
  std::size_t nb = 0;
  while (nb * nb < t.xy_hist.size()) ++nb;
  write_npy(dir + "/xy_hist.npy", t.xy_hist.data(), "<u8", {nb, nb}, 8);
  write_text_file(dir + "/meta.json", meta);
}

}  // namespace optphot
