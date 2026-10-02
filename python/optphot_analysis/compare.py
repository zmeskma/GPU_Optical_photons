"""Photon-by-photon and statistical comparison of two optphot record sets
(Python counterpart of include/optphot/compare.hpp)."""
import numpy as np
from scipy import stats

FLOAT_FIELDS = ["t", "x", "y", "z", "path"]


def history_agreement(a, b):
    """Same-seed comparison. Returns fractions of bitwise-identical records,
    of records with the same discrete history (fate, number of boundary
    interactions and scatterings), and the largest continuous differences
    among the latter."""
    n = len(a["fate"])
    discrete = ((a["fate"] == b["fate"]) & (a["n_boundary"] == b["n_boundary"])
                & (a["n_scatter"] == b["n_scatter"]))
    bitwise = discrete.copy()
    for k in FLOAT_FIELDS:
        bitwise &= a[k].view(np.uint32) == b[k].view(np.uint32)
    d = discrete
    dpos = np.sqrt((a["x"][d] - b["x"][d].astype(np.float64)) ** 2
                   + (a["y"][d] - b["y"][d].astype(np.float64)) ** 2
                   + (a["z"][d] - b["z"][d].astype(np.float64)) ** 2)
    dt = np.abs(a["t"][d] - b["t"][d].astype(np.float64))
    return dict(n=int(n), frac_bitwise=float(bitwise.mean()),
                frac_same_discrete=float(discrete.mean()),
                fate_differs=int((a["fate"] != b["fate"]).sum()),
                max_abs_dt_ns=float(dt.max()) if dt.size else 0.0,
                max_abs_dpos_mm=float(dpos.max()) if dpos.size else 0.0,
                divergent_index=np.flatnonzero(~discrete)[:20].tolist())


def two_sample_chi2(h1, h2):
    """Two-sample chi-square for histograms with different totals."""
    h1, h2 = np.asarray(h1, float), np.asarray(h2, float)
    A, B = h1.sum(), h2.sum()
    m = (h1 + h2) > 0
    chi2 = np.sum((np.sqrt(B / A) * h1[m] - np.sqrt(A / B) * h2[m]) ** 2 / (h1[m] + h2[m]))
    ndf = int(m.sum()) - 1
    return float(chi2), ndf, float(stats.chi2.sf(chi2, ndf))


def distribution_tests(a, b, half_x=25.0, half_y=25.0, t_max=None, time_bins=100, xy_bins=20):
    """Statistical comparison (meaningful for INDEPENDENT seeds)."""
    da, db = a["fate"] == 0, b["fate"] == 0
    na, nb = len(a["fate"]), len(b["fate"])
    ea, eb = da.mean(), db.mean()
    sa, sb = np.sqrt(ea * (1 - ea) / na), np.sqrt(eb * (1 - eb) / nb)
    if t_max is None:  # latest detected arrival in either sample (prompt setups: sub-ns)
        t_max = float(max(a["t"][da].max(), b["t"][db].max())) * (1 + 1e-6)
    tb = np.append(np.linspace(0, t_max, time_bins + 1), np.inf)
    ht1, _ = np.histogram(a["t"][da], tb)
    ht2, _ = np.histogram(b["t"][db], tb)
    ex = np.linspace(-half_x, half_x, xy_bins + 1)
    ey = np.linspace(-half_y, half_y, xy_bins + 1)
    hx1, _, _ = np.histogram2d(a["x"][da], a["y"][da], [ex, ey])
    hx2, _, _ = np.histogram2d(b["x"][db], b["y"][db], [ex, ey])
    ks = stats.ks_2samp(a["t"][da], b["t"][db])
    tc = two_sample_chi2(ht1, ht2)
    xc = two_sample_chi2(hx1.ravel(), hx2.ravel())
    return dict(eff_a=float(ea), eff_b=float(eb), err_a=float(sa), err_b=float(sb),
                eff_z=float((ea - eb) / np.hypot(sa, sb)),
                time_chi2=tc[0], time_ndf=tc[1], time_chi2_p=tc[2],
                xy_chi2=xc[0], xy_ndf=xc[1], xy_chi2_p=xc[2],
                time_ks_d=float(ks.statistic), time_ks_p=float(ks.pvalue))
