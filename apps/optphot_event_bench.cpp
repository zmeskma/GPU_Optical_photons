// SPDX-License-Identifier: MIT
//
// optphot_event_bench - history-based vs event-based GPU transport.
//
// For one configuration, measures the records-mode transport time of
//   * the history-based kernel (run_gpu_records), and
//   * the event-based kernel (run_gpu_records_event) for each requested
//     number of steps per launch k,
// checks that every event-based run reproduces the history-based records bit
// for bit, and writes a CSV for python/plot_event_based.py. The SIMT
// efficiency of the history-based kernel is computed exactly from its records
// (a warp = 32 consecutive photons); the event-based one is counted on the GPU.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "optphot/backend.hpp"
#include "optphot/compare.hpp"
#include "optphot/config.hpp"

using namespace optphot;

namespace {

const char* kUsage = R"(usage: optphot_event_bench [options] [physics options as for optphot]
  --k 1,2,4,...     steps per launch of the event-based kernel
                    (default 1,2,4,8,16,32,64,128,10000)
  --repeat N        repetitions                    (default 3)
  --csv FILE        output file                    (default event_bench.csv)
The number of photons is --n_photons (default 1e6) and the block size
--block_size, as for optphot.
)";

uint32_t steps_of(const PhotonRecords& r, std::size_t i) {
  return r.n_boundary[i] + r.n_scatter[i] + (r.fate[i] == static_cast<uint8_t>(Fate::AbsorbedBulk));
}

// Exact lane-step counts of the history-based kernel: warp w holds photons
// 32w .. 32w + 31 and runs as long as its longest history.
EventStats history_lane_steps(const PhotonRecords& r) {
  EventStats s;
  for (std::size_t w = 0; w < r.size(); w += 32) {
    uint32_t longest = 0;
    for (std::size_t i = w; i < std::min(r.size(), w + 32); ++i) {
      s.useful_lane_steps += steps_of(r, i);
      longest = std::max(longest, steps_of(r, i));
    }
    s.issued_lane_steps += 32ull * longest;
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  SimParams p;
  RunOptions o;
  std::vector<uint32_t> ks{1, 2, 4, 8, 16, 32, 64, 128, 10000};
  int repeat = 3;
  std::string csv_path = "event_bench.csv";
  try {
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
      } else if (a == "--k") {
        ks.clear();
        std::stringstream ss(value());
        for (std::string item; std::getline(ss, item, ',');) {
          if (!item.empty()) ks.push_back(static_cast<uint32_t>(std::stod(item)));
        }
      } else if (a == "--repeat") {
        repeat = std::stoi(value());
      } else if (a == "--csv") {
        csv_path = value();
      } else {
        rest.push_back(argv[i]);
      }
    }
    if (!parse_command_line(static_cast<int>(rest.size()), rest.data(), p, o)) {
      std::cout << kUsage << "\n" << usage();
      return 0;
    }
    if (!gpu_available()) {
      std::cerr << "error: no CUDA device available\n";
      return 2;
    }
    const uint64_t n = o.n_photons;
    const std::string device = gpu_device_name();
    std::printf("optphot_event_bench: %llu photons, surface %s, source %s, block %d, GPU %s\n",
                static_cast<unsigned long long>(n), surface_name(p.surface).c_str(),
                source_name(p.source).c_str(), o.block_size, device.c_str());

    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("cannot write " + csv_path);
    csv << "method,k,n_photons,repeat,transport_ms,kernels_ms,launches,useful_lane_steps,"
           "issued_lane_steps,simt_eff,bitwise_identical,photons_per_s,device\n";
    auto write = [&](const char* method, uint32_t k, int r, double transport_ms,
                     const EventStats& s, uint64_t identical) {
      const double pps = n / (transport_ms * 1e-3);
      csv << method << "," << k << "," << n << "," << r << "," << transport_ms << ","
          << s.kernels_ms << "," << s.launches << "," << s.useful_lane_steps << ","
          << s.issued_lane_steps << "," << s.simt_efficiency() << "," << identical << "," << pps
          << ",\"" << device << "\"\n";
      std::printf(
          "%-8s k=%-6u rep %d: transport %9.2f ms (kernels %9.2f ms, %5llu launches)  "
          "SIMT eff. %5.1f%%  %.3e photons/s  identical %llu/%llu\n",
          method, k, r, transport_ms, s.kernels_ms, static_cast<unsigned long long>(s.launches),
          100.0 * s.simt_efficiency(), pps, static_cast<unsigned long long>(identical),
          static_cast<unsigned long long>(n));
      std::fflush(stdout);
    };

    // Warm-up: context creation and, on laptops, leaving the idle clocks.
    {
      PhotonRecords warm;
      double busy_ms = 0.0;
      while (busy_ms < 500.0)
        busy_ms += run_gpu_records(p, std::min<uint64_t>(n, 4000000), warm, o.block_size).total_ms;
    }

    for (int r = 0; r < repeat; ++r) {
      PhotonRecords hist;
      const RunTiming th = run_gpu_records(p, n, hist, o.block_size, 0, n);
      EventStats hs = history_lane_steps(hist);
      hs.launches = 1;
      hs.kernels_ms = th.kernel_ms;
      write("history", 0, r, th.kernel_ms, hs, n);
      for (const uint32_t k : ks) {
        PhotonRecords ev;
        EventStats es;
        const RunTiming te = run_gpu_records_event(p, n, ev, k, o.block_size, 0, n, &es);
        const HistoryAgreement h = compare_histories(hist, ev);
        if (h.bitwise_identical != n) {
          std::fprintf(stderr, "error: event-based k=%u differs from history-based (%llu/%llu)\n",
                       k, static_cast<unsigned long long>(h.bitwise_identical),
                       static_cast<unsigned long long>(n));
        }
        write("event", k, r, te.kernel_ms, es, h.bitwise_identical);
      }
    }
    std::printf("written %s\n", csv_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
