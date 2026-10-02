// SPDX-License-Identifier: MIT
//
// optphot - command-line driver: run N photons on the CPU or GPU backend,
// print a summary and optionally write the results for Python.
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>

#include "optphot/backend.hpp"
#include "optphot/config.hpp"

using namespace optphot;

namespace {

void print_summary(const std::array<uint64_t, kNumFates>& counts, uint64_t n, const RunTiming& tm,
                   double mean_boundaries) {
  std::printf("%-18s %12s %10s %10s\n", "fate", "count", "fraction", "stat.err");
  for (int f = 0; f < kNumFates; ++f) {
    const double frac = static_cast<double>(counts[f]) / n;
    std::printf("%-18s %12llu %10.6f %10.6f\n", fate_name(static_cast<Fate>(f)).c_str(),
                static_cast<unsigned long long>(counts[f]), frac, std::sqrt(frac * (1 - frac) / n));
  }
  std::printf("mean boundary interactions / photon: %.4f\n", mean_boundaries);
  std::printf("time: total %.2f ms, transport %.2f ms", tm.total_ms, tm.kernel_ms);
  if (tm.h2d_ms + tm.d2h_ms + tm.alloc_ms > 0) {
    std::printf(" (alloc %.2f ms, H2D %.3f ms, D2H %.2f ms)", tm.alloc_ms, tm.h2d_ms, tm.d2h_ms);
  }
  std::printf("\nthroughput: %.3e photons/s (transport only)\n", n / (tm.kernel_ms * 1e-3));
}

}  // namespace

int main(int argc, char** argv) {
  SimParams p;
  RunOptions o;
  try {
    if (!parse_command_line(argc, argv, p, o)) {
      std::cout << usage();
      return 0;
    }
#if !defined(OPTPHOT_HAVE_CUDA)
    if (o.backend == "gpu") {
      std::cerr << "error: this build has no CUDA support (configure with -DWITH_CUDA=ON)\n";
      return 2;
    }
#endif
    std::printf("optphot: %llu photons, backend %s, mode %s, surface %s, source %s\n",
                static_cast<unsigned long long>(o.n_photons), o.backend.c_str(), o.mode.c_str(),
                surface_name(p.surface).c_str(), source_name(p.source).c_str());
    if (o.backend == "cpu") {
      std::printf("CPU threads: %d (OpenMP %s)\n", o.threads == 0 ? cpu_max_threads() : o.threads,
                  cpu_has_openmp() ? "available" : "not available");
    }
#if defined(OPTPHOT_HAVE_CUDA)
    else {
      std::printf("GPU: %s\n", gpu_device_name().c_str());
    }
#endif

    const std::string meta = params_to_json(p, o);
    if (o.mode == "records") {
      PhotonRecords rec;
      RunTiming tm;
      if (o.backend == "cpu") tm = run_cpu_records(p, o.n_photons, rec, o.threads);
#if defined(OPTPHOT_HAVE_CUDA)
      else
        tm = run_gpu_records(p, o.n_photons, rec, o.block_size);
#endif
      std::array<uint64_t, kNumFates> counts{};
      double nb = 0;
      for (std::size_t i = 0; i < rec.size(); ++i) {
        ++counts[rec.fate[i]];
        nb += rec.n_boundary[i];
      }
      print_summary(counts, o.n_photons, tm, nb / o.n_photons);
      if (!o.out.empty()) write_records(o.out, rec, meta);
    } else {
      Tally tally;
      RunTiming tm;
      if (o.backend == "cpu") tm = run_cpu_tally(p, o.tally, o.n_photons, tally, o.threads);
#if defined(OPTPHOT_HAVE_CUDA)
      else
        tm = run_gpu_tally(p, o.tally, o.n_photons, tally, o.block_size);
#endif
      print_summary(tally.fate_counts, o.n_photons, tm,
                    static_cast<double>(tally.boundary_sum) / o.n_photons);
      if (!o.out.empty()) write_tally(o.out, tally, meta);
    }
    if (!o.out.empty()) std::printf("output written to %s\n", o.out.c_str());
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
