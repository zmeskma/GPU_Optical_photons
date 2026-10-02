// SPDX-License-Identifier: MIT
//
// compare.hpp - comparison of two sets of photon records (e.g. CPU vs GPU).
//
// Two complementary comparisons:
//
//  1. History agreement (same seed on both sides): because photon i uses the
//     same random stream everywhere, records can be compared one by one.
//       - bitwise identical: every field equal to the last bit;
//       - same discrete history: same fate, number of boundary interactions
//         and number of scatterings (continuous values may differ by
//         rounding);
//       - divergent: different fate or interaction counts. This happens when
//         a last-bit difference (FMA, math library) flips a decision such as
//         u <= R or which face is reached first.
//
//  2. Distribution agreement (normally with INDEPENDENT seeds): detection
//     efficiency z-test, two-sample chi-square of the arrival-time and hit
//     position histograms, two-sample KS on arrival times. Note that with the
//     same seed the two samples are almost perfectly correlated, so these
//     tests would pass trivially; they are meaningful for independent seeds.
#pragma once

#include <string>

#include "optphot/backend.hpp"
#include "optphot/stats.hpp"

namespace optphot {

struct HistoryAgreement {
  uint64_t n = 0;
  uint64_t bitwise_identical = 0;
  uint64_t same_discrete = 0;  // includes the bitwise identical ones
  uint64_t fate_differs = 0;
  double max_abs_dt = 0.0;    // ns, over histories with the same discrete path
  double max_abs_dpos = 0.0;  // mm, idem
  int64_t first_divergent = -1;
};

HistoryAgreement compare_histories(const PhotonRecords& a, const PhotonRecords& b);

struct DistributionComparison {
  uint64_t n_a = 0, n_b = 0, det_a = 0, det_b = 0;
  double eff_a = 0, eff_b = 0, err_a = 0, err_b = 0;
  double eff_z = 0;  // (eff_a - eff_b) / sqrt(err_a^2 + err_b^2)
  stats::Chi2Result time_chi2, xy_chi2;
  stats::KsResult time_ks;
};

// Histograms detected photons' arrival times on [0, t_max) with time_bins bins
// and hit positions on the readout face with xy_bins x xy_bins bins.
DistributionComparison compare_distributions(const PhotonRecords& a, const PhotonRecords& b,
                                             const SimParams& p, int time_bins = 100,
                                             float t_max = 50.0f, int xy_bins = 20);

std::string to_json(const HistoryAgreement& h);
std::string to_json(const DistributionComparison& d);

}  // namespace optphot
