// SPDX-License-Identifier: MIT
#include "optphot/compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace optphot {

namespace {

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

struct Histos {
  std::vector<double> time, xy, times;
  uint64_t detected = 0;
};

Histos histogram(const PhotonRecords& r, const SimParams& p, int time_bins, float t_max,
                 int xy_bins) {
  Histos h;
  h.time.assign(time_bins + 1, 0.0);  // last bin: overflow
  h.xy.assign(static_cast<std::size_t>(xy_bins) * xy_bins, 0.0);
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (static_cast<Fate>(r.fate[i]) != Fate::Detected) continue;
    ++h.detected;
    const int tb = std::min(time_bins, static_cast<int>(r.t[i] / t_max * time_bins));
    h.time[tb] += 1.0;
    const int ix = std::min(xy_bins - 1, static_cast<int>((r.x[i] + p.hx) / (2 * p.hx) * xy_bins));
    const int iy = std::min(xy_bins - 1, static_cast<int>((r.y[i] + p.hy) / (2 * p.hy) * xy_bins));
    h.xy[static_cast<std::size_t>(iy) * xy_bins + ix] += 1.0;
    h.times.push_back(r.t[i]);
  }
  return h;
}

}  // namespace

HistoryAgreement compare_histories(const PhotonRecords& a, const PhotonRecords& b) {
  if (a.size() != b.size()) throw std::runtime_error("compare_histories: different sizes");
  HistoryAgreement h;
  h.n = a.size();
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool discrete = a.fate[i] == b.fate[i] && a.n_boundary[i] == b.n_boundary[i] &&
                          a.n_scatter[i] == b.n_scatter[i];
    if (a.fate[i] != b.fate[i]) ++h.fate_differs;
    if (!discrete) {
      if (h.first_divergent < 0) h.first_divergent = static_cast<int64_t>(i);
      continue;
    }
    ++h.same_discrete;
    if (same_bits(a.t[i], b.t[i]) && same_bits(a.x[i], b.x[i]) && same_bits(a.y[i], b.y[i]) &&
        same_bits(a.z[i], b.z[i]) && same_bits(a.path[i], b.path[i])) {
      ++h.bitwise_identical;
    }
    h.max_abs_dt = std::max(h.max_abs_dt, std::fabs(static_cast<double>(a.t[i]) - b.t[i]));
    const double dx = a.x[i] - static_cast<double>(b.x[i]);
    const double dy = a.y[i] - static_cast<double>(b.y[i]);
    const double dz = a.z[i] - static_cast<double>(b.z[i]);
    h.max_abs_dpos = std::max(h.max_abs_dpos, std::sqrt(dx * dx + dy * dy + dz * dz));
  }
  return h;
}

DistributionComparison compare_distributions(const PhotonRecords& a, const PhotonRecords& b,
                                             const SimParams& p, int time_bins, float t_max,
                                             int xy_bins) {
  const Histos ha = histogram(a, p, time_bins, t_max, xy_bins);
  const Histos hb = histogram(b, p, time_bins, t_max, xy_bins);
  DistributionComparison d;
  d.n_a = a.size();
  d.n_b = b.size();
  d.det_a = ha.detected;
  d.det_b = hb.detected;
  d.eff_a = static_cast<double>(ha.detected) / a.size();
  d.eff_b = static_cast<double>(hb.detected) / b.size();
  d.err_a = std::sqrt(d.eff_a * (1 - d.eff_a) / a.size());
  d.err_b = std::sqrt(d.eff_b * (1 - d.eff_b) / b.size());
  const double err = std::sqrt(d.err_a * d.err_a + d.err_b * d.err_b);
  d.eff_z = err > 0 ? (d.eff_a - d.eff_b) / err : 0.0;
  d.time_chi2 = stats::chi2_two_sample(ha.time, hb.time);
  d.xy_chi2 = stats::chi2_two_sample(ha.xy, hb.xy);
  d.time_ks = stats::ks_two_sample(ha.times, hb.times);
  return d;
}

std::string to_json(const HistoryAgreement& h) {
  std::ostringstream os;
  os.precision(10);
  os << "{\"n\": " << h.n << ", \"bitwise_identical\": " << h.bitwise_identical
     << ", \"same_discrete\": " << h.same_discrete << ", \"fate_differs\": " << h.fate_differs
     << ", \"frac_bitwise\": " << static_cast<double>(h.bitwise_identical) / h.n
     << ", \"frac_same_discrete\": " << static_cast<double>(h.same_discrete) / h.n
     << ", \"max_abs_dt_ns\": " << h.max_abs_dt << ", \"max_abs_dpos_mm\": " << h.max_abs_dpos
     << ", \"first_divergent\": " << h.first_divergent << "}";
  return os.str();
}

std::string to_json(const DistributionComparison& d) {
  std::ostringstream os;
  os.precision(10);
  os << "{\"n_a\": " << d.n_a << ", \"n_b\": " << d.n_b << ", \"det_a\": " << d.det_a
     << ", \"det_b\": " << d.det_b << ", \"eff_a\": " << d.eff_a << ", \"eff_b\": " << d.eff_b
     << ", \"err_a\": " << d.err_a << ", \"err_b\": " << d.err_b << ", \"eff_z\": " << d.eff_z
     << ", \"time_chi2\": " << d.time_chi2.chi2 << ", \"time_ndf\": " << d.time_chi2.ndf
     << ", \"time_chi2_p\": " << d.time_chi2.pvalue << ", \"xy_chi2\": " << d.xy_chi2.chi2
     << ", \"xy_ndf\": " << d.xy_chi2.ndf << ", \"xy_chi2_p\": " << d.xy_chi2.pvalue
     << ", \"time_ks_d\": " << d.time_ks.d << ", \"time_ks_p\": " << d.time_ks.pvalue << "}";
  return os.str();
}

}  // namespace optphot
