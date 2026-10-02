// SPDX-License-Identifier: MIT
//
// optphot_bench - throughput (photons / s) versus number of photons for the
// available backends, written as CSV for python/plot_benchmarks.py.
//
// Backends:  cpu1 (single thread), cpu_omp (all OpenMP threads), gpu.
// Modes:     tally (histograms only) and records (one record per photon,
//            which on the GPU includes the device-to-host copy).
//
// Every measurement is repeated --repeat times; all repeats are written so
// the plotting script can show the median and spread.
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "optphot/backend.hpp"
#include "optphot/config.hpp"

using namespace optphot;

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

struct BenchOptions {
  std::vector<uint64_t> sizes{10000, 100000, 1000000, 10000000, 100000000};
  std::vector<std::string> backends{"cpu1", "cpu_omp", "gpu"};
  std::vector<std::string> modes{"tally", "records"};
  int repeat = 3;
  uint64_t max_records = 10000000;  // records mode needs 29 B / photon of host memory
  uint64_t max_cpu1 = 10000000;     // single-thread CPU is slow; cap its sizes
  std::string csv = "bench.csv";
};

const char* kUsage = R"(usage: optphot_bench [bench options] [physics options as for optphot]
  --sizes 1e4,1e5,...      photon counts           (default 1e4..1e8)
  --backends cpu1,cpu_omp,gpu                      (default: all available)
  --modes tally,records                            (default both)
  --repeat N               repetitions per point   (default 3)
  --max_records N          largest N for records mode (default 1e7)
  --max_cpu1 N             largest N for single-thread CPU (default 1e7)
  --csv FILE               output file             (default bench.csv)
)";

}  // namespace

int main(int argc, char** argv) {
  SimParams p;
  RunOptions o;
  BenchOptions b;
  try {
    // Split argv into bench options and physics options.
    std::vector<char*> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto value = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "-h" || a == "--help") {
        std::cout << kUsage << "\n" << usage();
        return 0;
      } else if (a == "--sizes") {
        b.sizes.clear();
        for (const auto& s : split(value())) b.sizes.push_back(static_cast<uint64_t>(std::stod(s)));
      } else if (a == "--backends") {
        b.backends = split(value());
      } else if (a == "--modes") {
        b.modes = split(value());
      } else if (a == "--repeat") {
        b.repeat = std::stoi(value());
      } else if (a == "--max_records") {
        b.max_records = static_cast<uint64_t>(std::stod(value()));
      } else if (a == "--max_cpu1") {
        b.max_cpu1 = static_cast<uint64_t>(std::stod(value()));
      } else if (a == "--csv") {
        b.csv = value();
      } else {
        rest.push_back(argv[i]);
      }
    }
    if (!parse_command_line(static_cast<int>(rest.size()), rest.data(), p, o)) {
      std::cout << kUsage << "\n" << usage();
      return 0;
    }

#if defined(OPTPHOT_HAVE_CUDA)
    const bool have_gpu = gpu_available();
    const std::string device = have_gpu ? gpu_device_name() : "none";
#else
    const bool have_gpu = false;
    const std::string device = "none (built without CUDA)";
#endif
    std::printf("optphot_bench: surface %s, source %s; CPU threads %d; GPU %s\n",
                surface_name(p.surface).c_str(), source_name(p.source).c_str(), cpu_max_threads(),
                device.c_str());

    std::ofstream csv(b.csv);
    if (!csv) throw std::runtime_error("cannot write " + b.csv);
    csv << "backend,mode,n_photons,repeat,threads,total_ms,kernel_ms,alloc_ms,h2d_ms,d2h_ms,"
           "photons_per_s_kernel,photons_per_s_total,device\n";

    for (const std::string& backend : b.backends) {
      if (backend == "gpu" && !have_gpu) {
        std::printf("skipping gpu backend (not available)\n");
        continue;
      }
      if (backend == "cpu_omp" && !cpu_has_openmp()) {
        std::printf("skipping cpu_omp backend (no OpenMP)\n");
        continue;
      }
#if defined(OPTPHOT_HAVE_CUDA)
      if (backend == "gpu") {
        // Warm-up immediately before the GPU measurements: creates the context
        // and, importantly on laptops, lets the GPU leave its idle clock state
        // (it drops back while the CPU benchmarks run). ~0.5 s of real work.
        Tally warm;
        double busy_ms = 0.0;
        while (busy_ms < 500.0) busy_ms += run_gpu_tally(p, o.tally, 10000000, warm, o.block_size).total_ms;
      }
#endif
      const int threads = backend == "cpu1" ? 1 : 0;
      for (const std::string& mode : b.modes) {
        for (const uint64_t n : b.sizes) {
          if (mode == "records" && n > b.max_records) continue;
          if (backend == "cpu1" && n > b.max_cpu1) continue;
          for (int r = 0; r < b.repeat; ++r) {
            RunTiming t;
            if (mode == "tally") {
              Tally tally;
              if (backend == "gpu") {
#if defined(OPTPHOT_HAVE_CUDA)
                t = run_gpu_tally(p, o.tally, n, tally, o.block_size);
#endif
              } else {
                t = run_cpu_tally(p, o.tally, n, tally, threads);
              }
            } else {
              PhotonRecords rec;
              if (backend == "gpu") {
#if defined(OPTPHOT_HAVE_CUDA)
                t = run_gpu_records(p, n, rec, o.block_size);
#endif
              } else {
                t = run_cpu_records(p, n, rec, threads);
              }
            }
            const double pps_k = n / (t.kernel_ms * 1e-3);
            const double pps_t = n / (t.total_ms * 1e-3);
            csv << backend << "," << mode << "," << n << "," << r << ","
                << (backend == "gpu" ? 0 : (threads == 0 ? cpu_max_threads() : 1)) << ","
                << t.total_ms << "," << t.kernel_ms << "," << t.alloc_ms << "," << t.h2d_ms << ","
                << t.d2h_ms << "," << pps_k << "," << pps_t << ",\""
                << (backend == "gpu" ? device : "cpu") << "\"\n";
            std::printf(
                "%-8s %-8s n=%-10llu rep %d: kernel %10.2f ms  total %10.2f ms  "
                "(D2H %8.2f ms)  %.3e photons/s\n",
                backend.c_str(), mode.c_str(), static_cast<unsigned long long>(n), r, t.kernel_ms,
                t.total_ms, t.d2h_ms, pps_k);
            std::fflush(stdout);
          }
        }
      }
    }
    std::printf("written %s\n", b.csv.c_str());
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
