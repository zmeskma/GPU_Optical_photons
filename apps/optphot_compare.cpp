// SPDX-License-Identifier: MIT
//
// optphot_compare - CPU vs GPU comparison for one configuration.
//
//   A: CPU backend, seed s
//   B: GPU backend, seed s      -> photon-by-photon history agreement A vs B
//   C: GPU backend, seed s + 1  -> statistical tests A vs C (independent samples)
//
// Prints a summary and, with --out DIR, writes DIR/compare.json plus the
// records of A, B and C (DIR/cpu, DIR/gpu, DIR/gpu_indep) for the Python plots.
#include <cstdio>
#include <exception>
#include <iostream>

#include "optphot/backend.hpp"
#include "optphot/compare.hpp"
#include "optphot/config.hpp"

using namespace optphot;

int main(int argc, char** argv) {
  SimParams p;
  RunOptions o;
  o.threads = 0;  // CPU reference uses all cores by default
  try {
    if (!parse_command_line(argc, argv, p, o)) {
      std::cout << "optphot_compare: same options as optphot (backend/mode are ignored)\n\n"
                << usage();
      return 0;
    }
    if (!gpu_available()) {
      std::cerr << "error: no CUDA device available\n";
      return 2;
    }
    const uint64_t n = o.n_photons;
    std::printf("optphot_compare: %llu photons, surface %s, GPU %s\n",
                static_cast<unsigned long long>(n), surface_name(p.surface).c_str(),
                gpu_device_name().c_str());

    PhotonRecords cpu, gpu, gpu_indep;
    const RunTiming tc = run_cpu_records(p, n, cpu, o.threads);
    const RunTiming tg = run_gpu_records(p, n, gpu, o.block_size);
    SimParams p_indep = p;
    p_indep.seed = p.seed + 1;
    run_gpu_records(p_indep, n, gpu_indep, o.block_size);

    const HistoryAgreement h = compare_histories(cpu, gpu);
    const DistributionComparison same = compare_distributions(cpu, gpu, p);
    const DistributionComparison indep = compare_distributions(cpu, gpu_indep, p);

    std::printf("\nSame seed, photon by photon (CPU vs GPU):\n");
    std::printf("  bitwise identical records : %llu / %llu (%.6f)\n",
                static_cast<unsigned long long>(h.bitwise_identical),
                static_cast<unsigned long long>(n), double(h.bitwise_identical) / n);
    std::printf("  same discrete history     : %llu / %llu (%.6f)\n",
                static_cast<unsigned long long>(h.same_discrete),
                static_cast<unsigned long long>(n), double(h.same_discrete) / n);
    std::printf("  different fate            : %llu\n",
                static_cast<unsigned long long>(h.fate_differs));
    std::printf("  max |dt| = %.3g ns, max |dpos| = %.3g mm (same discrete history)\n",
                h.max_abs_dt, h.max_abs_dpos);
    std::printf("\nIndependent seeds (CPU seed %llu vs GPU seed %llu):\n",
                static_cast<unsigned long long>(p.seed),
                static_cast<unsigned long long>(p_indep.seed));
    std::printf("  efficiency CPU %.6f +- %.6f, GPU %.6f +- %.6f, z = %.2f\n", indep.eff_a,
                indep.err_a, indep.eff_b, indep.err_b, indep.eff_z);
    std::printf("  arrival time: chi2/ndf = %.1f/%d (p = %.3f), KS D = %.2e (p = %.3f)\n",
                indep.time_chi2.chi2, indep.time_chi2.ndf, indep.time_chi2.pvalue, indep.time_ks.d,
                indep.time_ks.pvalue);
    std::printf("  hit position: chi2/ndf = %.1f/%d (p = %.3f)\n", indep.xy_chi2.chi2,
                indep.xy_chi2.ndf, indep.xy_chi2.pvalue);
    std::printf("\nTiming: CPU %.1f ms (%d threads); GPU kernel %.1f ms, D2H %.1f ms\n",
                tc.total_ms, o.threads == 0 ? cpu_max_threads() : o.threads, tg.kernel_ms,
                tg.d2h_ms);

    if (!o.out.empty()) {
      const std::string meta = params_to_json(p, o);
      write_records(o.out + "/cpu", cpu, meta);
      write_records(o.out + "/gpu", gpu, meta);
      write_records(o.out + "/gpu_indep", gpu_indep, params_to_json(p_indep, o));
      write_text_file(o.out + "/compare.json",
                      "{\n\"params\": " + meta + ",\n\"device\": \"" + gpu_device_name() +
                          "\",\n\"history\": " + to_json(h) + ",\n\"same_seed\": " + to_json(same) +
                          ",\n\"independent_seed\": " + to_json(indep) +
                          ",\n\"timing_ms\": {\"cpu_total\": " + std::to_string(tc.total_ms) +
                          ", \"gpu_kernel\": " + std::to_string(tg.kernel_ms) +
                          ", \"gpu_d2h\": " + std::to_string(tg.d2h_ms) + "}\n}");
      std::printf("written to %s\n", o.out.c_str());
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
