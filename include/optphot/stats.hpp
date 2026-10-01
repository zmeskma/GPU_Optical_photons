// SPDX-License-Identifier: MIT
//
// stats.hpp - small host-only statistics helpers used by the tests and the
// CPU/GPU comparison harness (chi-square goodness of fit and its p-value).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace optphot::stats {

// Regularised upper incomplete gamma function Q(a, x) = Gamma(a, x) / Gamma(a)
// (series for x < a + 1, Lentz continued fraction otherwise; Numerical
// Recipes 3rd ed. section 6.2).
inline double gamma_q(double a, double x) {
  if (x <= 0.0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1.0) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 10000; ++n) {
      ap += 1.0;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  const double tiny = 1e-300;
  double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
  for (int i = 1; i < 10000; ++i) {
    const double an = -i * (i - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-15) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

// p-value of a chi-square statistic with `ndf` degrees of freedom.
inline double chi2_pvalue(double chi2, double ndf) { return gamma_q(0.5 * ndf, 0.5 * chi2); }

struct Chi2Result {
  double chi2 = 0.0;
  int ndf = 0;
  double pvalue = 1.0;
};

// Pearson chi-square of observed counts against expected counts. Bins with
// expected < min_expected are skipped (the chi-square approximation needs a
// reasonable number of entries per bin).
inline Chi2Result chi2_gof(const std::vector<double>& observed, const std::vector<double>& expected,
                           int fitted_params = 0, double min_expected = 5.0) {
  Chi2Result r;
  int nbins = 0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    if (expected[i] < min_expected) continue;
    const double d = observed[i] - expected[i];
    r.chi2 += d * d / expected[i];
    ++nbins;
  }
  r.ndf = nbins - fitted_params;
  r.pvalue = chi2_pvalue(r.chi2, r.ndf);
  return r;
}

}  // namespace optphot::stats
