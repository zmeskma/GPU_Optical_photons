// SPDX-License-Identifier: MIT
//
// stats.hpp - small host-only statistics helpers used by the tests and the
// CPU/GPU comparison harness: chi-square and Kolmogorov-Smirnov tests.
#pragma once

#include <algorithm>
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

// Two-sample chi-square test for histograms a and b (possibly different
// totals), NR 3rd ed. section 14.3: chi2 = sum (sqrt(B/A) a - sqrt(A/B) b)^2 / (a + b).
// Bins empty in both histograms are skipped. ndf = (non-empty bins) - 1.
inline Chi2Result chi2_two_sample(const std::vector<double>& a, const std::vector<double>& b) {
  double A = 0.0, B = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    A += a[i];
    B += b[i];
  }
  const double ra = std::sqrt(B / A), rb = std::sqrt(A / B);
  Chi2Result r;
  int nbins = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] + b[i] <= 0.0) continue;
    const double d = ra * a[i] - rb * b[i];
    r.chi2 += d * d / (a[i] + b[i]);
    ++nbins;
  }
  r.ndf = nbins - 1;
  r.pvalue = chi2_pvalue(r.chi2, r.ndf);
  return r;
}

// Kolmogorov distribution Q_KS(lambda) = 2 sum_j (-1)^(j-1) exp(-2 j^2 lambda^2).
inline double kolmogorov_q(double lambda) {
  if (lambda < 0.2) return 1.0;
  double sum = 0.0, sign = 1.0;
  for (int j = 1; j <= 100; ++j) {
    const double term = sign * std::exp(-2.0 * j * j * lambda * lambda);
    sum += term;
    if (std::fabs(term) < 1e-16) break;
    sign = -sign;
  }
  return std::clamp(2.0 * sum, 0.0, 1.0);
}

struct KsResult {
  double d = 0.0;  // max |F1 - F2|
  double pvalue = 1.0;
};

// One-sample KS test of `x` against the CDF `cdf` (sorts a copy of x).
template <class Cdf>
KsResult ks_one_sample(std::vector<double> x, Cdf cdf) {
  std::sort(x.begin(), x.end());
  const double n = static_cast<double>(x.size());
  KsResult r;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double f = cdf(x[i]);
    r.d = std::max({r.d, std::fabs((i + 1) / n - f), std::fabs(f - i / n)});
  }
  const double sn = std::sqrt(n);
  r.pvalue = kolmogorov_q((sn + 0.12 + 0.11 / sn) * r.d);
  return r;
}

// Two-sample KS test (sorts copies of the inputs). Assumes continuous data;
// ties between the samples are handled by advancing both.
inline KsResult ks_two_sample(std::vector<double> a, std::vector<double> b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  const double na = static_cast<double>(a.size()), nb = static_cast<double>(b.size());
  std::size_t i = 0, j = 0;
  KsResult r;
  while (i < a.size() && j < b.size()) {
    const double v = std::min(a[i], b[j]);
    while (i < a.size() && a[i] == v) ++i;
    while (j < b.size() && b[j] == v) ++j;
    r.d = std::max(r.d, std::fabs(i / na - j / nb));
  }
  const double ne = std::sqrt(na * nb / (na + nb));
  r.pvalue = kolmogorov_q((ne + 0.12 + 0.11 / ne) * r.d);
  return r;
}

}  // namespace optphot::stats
