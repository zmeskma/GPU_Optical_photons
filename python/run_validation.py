#!/usr/bin/env python3
"""Runs the optphot executable over parameter scans that have analytic
answers and plots MC vs analytic, with pulls, into docs/figures/.

The analytic values are computed here in Python (scipy), independently of
the C++ implementation used by the unit tests.

usage: python python/run_validation.py --exe build/apps/optphot [--backend cpu|gpu]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import kstest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optphot_analysis import analytic, io, style  # noqa: E402

HALF = (25.0, 25.0, 25.0)
N_SCINT = 1.63
C_MM_NS = 299.792458

# Common settings for the analytic configurations: black walls, matched
# readout, no bulk processes, prompt emission.
BLACK_BOX = dict(surface="black", n_det=N_SCINT, abs_length="inf", scat_length="inf", tau=0)


class Runner:
    def __init__(self, exe, backend, threads, workdir):
        self.exe = os.path.abspath(exe)
        self.backend, self.threads, self.workdir = backend, threads, workdir
        self.count = 0

    def run(self, n, **settings):
        """Runs optphot in records mode and returns the records dict."""
        self.count += 1
        out = os.path.join(self.workdir, f"run{self.count}")
        cmd = [self.exe, "--backend", self.backend, "--threads", str(self.threads),
               "--n_photons", str(int(n)), "--out", out]
        for k, v in settings.items():
            cmd += ["--" + k, str(v)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        rec = io.load_records(out)
        for f in io.RECORD_FIELDS:  # free disk space early
            os.remove(os.path.join(out, f + ".npy"))
        return rec


def errbar(ax, x, p, n, label, color=style.BLUE, marker="o"):
    ax.errorbar(x, p, yerr=np.sqrt(p * (1 - p) / n), fmt=marker, color=color, ms=5,
                capsize=0, elinewidth=1.5, label=label, zorder=3)


def residual_figure():
    return plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True, layout="constrained",
                        gridspec_kw=dict(height_ratios=[3, 1.2]))


def validate_solid_angle(r, n, outdir, results):
    zs = np.linspace(-22, 22, 12)
    fig, (ax, axp) = residual_figure()
    zf = np.linspace(-24.9, 24.9, 400)
    for x0, color, marker, name in [(0.0, style.BLUE, "o", "on axis (x = 0)"),
                                    (15.0, style.ORANGE, "s", "off axis (x = 15 mm)")]:
        p_mc, p_an = [], []
        for z in zs:
            rec = r.run(n, src_x=x0, src_y=0, src_z=z, seed=int(1000 + z * 10 + x0), **BLACK_BOX)
            p_mc.append(np.mean(rec["fate"] == 0))
            p_an.append(analytic.detection_fraction(HALF, (x0, 0, z)))
        p_mc, p_an = np.array(p_mc), np.array(p_an)
        ax.plot(zf, [analytic.detection_fraction(HALF, (x0, 0, z)) for z in zf],
                color=style.INK_SECONDARY, lw=1.5, zorder=2)
        errbar(ax, zs, p_mc, n, f"MC, source {name}", color, marker)
        pulls = analytic.binomial_pull(p_mc * n, n, p_an)
        style.pull_panel(axp, zs, pulls, color, marker=marker)
        results[f"solid_angle_x{x0:g}"] = dict(z=zs.tolist(), mc=p_mc.tolist(),
                                              analytic=p_an.tolist(), pulls=pulls.tolist())
    ax.plot([], [], color=style.INK_SECONDARY, lw=1.5, label="analytic Ω/4π")
    ax.set_ylabel("detection fraction")
    ax.set_title(f"Solid angle: black walls, index-matched readout at z = +25 mm\n"
                 f"({n:.0e} photons per point; error bars smaller than markers)")
    ax.legend(loc="upper left")
    axp.set_xlabel("source z (mm)")
    style.save(fig, os.path.join(outdir, "validation_solid_angle.png"))


def validate_absorption(r, n, outdir, results):
    src = (6.0, -8.0, -5.0)
    lengths = np.geomspace(5, 5000, 10)
    no_abs = {k: v for k, v in BLACK_BOX.items() if k != "abs_length"}
    p_mc, p_an = [], []
    for L in lengths:
        rec = r.run(n, abs_length=L, src_x=src[0], src_y=src[1], src_z=src[2],
                    seed=int(2000 + L), **no_abs)
        p_mc.append(np.mean(rec["fate"] == 0))
        p_an.append(analytic.attenuated_fraction(HALF, src, L))
    p_mc, p_an = np.array(p_mc), np.array(p_an)
    pulls = analytic.binomial_pull(p_mc * n, n, p_an)
    fig, (ax, axp) = residual_figure()
    lf = np.geomspace(4, 6000, 60)
    ax.plot(lf, [analytic.attenuated_fraction(HALF, src, L) for L in lf],
            color=style.INK_SECONDARY, lw=1.5, label="quadrature (1/4π)∫exp(−r/L)dΩ")
    ax.axhline(analytic.detection_fraction(HALF, src), color=style.INK_SECONDARY, lw=1,
               ls="--", label="L → ∞ (solid angle)")
    errbar(ax, lengths, p_mc, n, "MC")
    ax.set_xscale("log")
    ax.set_ylabel("detection fraction")
    ax.set_title(f"Bulk absorption: point source at {src} mm\n({n:.0e} photons per point)")
    ax.legend(loc="lower right")
    style.pull_panel(axp, lengths, pulls, logx=True)
    axp.set_xlabel("absorption length L (mm)")
    style.save(fig, os.path.join(outdir, "validation_absorption.png"))
    results["absorption"] = dict(L=lengths.tolist(), mc=p_mc.tolist(), analytic=p_an.tolist(),
                                 pulls=pulls.tolist())


def validate_hitmap(r, n, outdir, results):
    src = (5.0, -3.0, -10.0)
    nb = 25
    rec = r.run(n, src_x=src[0], src_y=src[1], src_z=src[2], seed=3, **BLACK_BOX)
    det = rec["fate"] == 0
    edges = np.linspace(-HALF[0], HALF[0], nb + 1)
    obs, _, _ = np.histogram2d(rec["y"][det], rec["x"][det], bins=[edges, edges])
    h = HALF[2] - src[2]
    exp = np.empty((nb, nb))
    for iy in range(nb):
        for ix in range(nb):
            exp[iy, ix] = n * analytic.rect_solid_angle(
                edges[ix] - src[0], edges[ix + 1] - src[0],
                edges[iy] - src[1], edges[iy + 1] - src[1], h) / (4 * np.pi)
    pull = (obs - exp) / np.sqrt(exp)
    chi2 = float(np.sum(pull**2))
    pval = float(chi2_dist.sf(chi2, nb * nb))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    vmax = max(obs.max(), exp.max())
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    for ax, data, title in [(axs[0], obs, "MC hits per bin"),
                            (axs[1], exp, "analytic: N·Ω(bin)/4π")]:
        im = ax.imshow(data, origin="lower", extent=ext, cmap=style.SEQUENTIAL, vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85, label="photons / bin")
    axs[0].set_ylabel("y (mm)")
    im = axs[2].imshow(pull, origin="lower", extent=ext, cmap=style.DIVERGING, vmin=-4, vmax=4)
    axs[2].set_title("(MC − analytic) / √analytic")
    axs[2].set_xlabel("x (mm)")
    axs[2].grid(False)
    fig.colorbar(im, ax=axs[2], shrink=0.85, label="pull (σ)")
    fig.suptitle(f"Hit positions on the readout face, source at {src} mm ({n:.0e} photons): "
                 f"χ²/ndf = {chi2:.0f}/{nb * nb}, p = {pval:.2f}")
    style.save(fig, os.path.join(outdir, "validation_hitmap.png"))
    results["hitmap"] = dict(chi2=chi2, ndf=nb * nb, p=pval, pulls=pull.ravel().tolist())


def validate_escape_cone(r, n, outdir, results):
    n_dets = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    p_mc = []
    for nd in n_dets:
        rec = r.run(n, surface="specular", reflectivity=1, n_det=nd, abs_length="inf",
                    scat_length="inf", max_steps=1000, src_x=3, src_y=-11, src_z=7,
                    seed=int(nd * 100))
        p_mc.append(np.mean(rec["fate"] == 0))
    p_mc = np.array(p_mc)
    p_an = np.array([analytic.mirror_box_escape_fraction(N_SCINT, nd) for nd in n_dets])
    pulls = analytic.binomial_pull(p_mc * n, n, p_an)
    fig, (ax, axp) = residual_figure()
    nf = np.linspace(1.0, 1.63, 200)
    ax.plot(nf, [analytic.mirror_box_escape_fraction(N_SCINT, x) for x in nf],
            color=style.INK_SECONDARY, lw=1.5, label="1 − cos θc = 1 − √(1 − (n_det/n)²)")
    errbar(ax, n_dets, p_mc, n, "MC")
    ax.set_ylabel("detection fraction")
    ax.set_title(f"Escape cone of a box with perfect mirror walls\n({n:.0e} photons per point)")
    ax.legend(loc="upper left")
    style.pull_panel(axp, n_dets, pulls)
    axp.set_xlabel("readout-side refractive index n_det (n_scint = 1.63)")
    style.save(fig, os.path.join(outdir, "validation_escape_cone.png"))
    results["escape_cone"] = dict(n_det=n_dets.tolist(), mc=p_mc.tolist(),
                                  analytic=p_an.tolist(), pulls=pulls.tolist())


def validate_rayleigh(r, n, outdir, results):
    lengths = np.array([5, 10, 20, 50, 100, 300, 1000], dtype=float)
    no_scat = {k: v for k, v in BLACK_BOX.items() if k != "scat_length"}
    p_det, p_unsc, p_unsc_an = [], [], []
    for L in lengths:
        rec = r.run(n, scat_length=L, seed=int(4000 + L), **no_scat)
        det = rec["fate"] == 0
        p_det.append(np.mean(det))
        p_unsc.append(np.mean(det & (rec["n_scatter"] == 0)))
        p_unsc_an.append(analytic.attenuated_fraction(HALF, (0, 0, 0), L))
    p_det, p_unsc, p_unsc_an = map(np.array, (p_det, p_unsc, p_unsc_an))
    pulls_det = analytic.binomial_pull(p_det * n, n, 1 / 6)
    pulls_unsc = analytic.binomial_pull(p_unsc * n, n, p_unsc_an)
    fig, (ax, axp) = residual_figure()
    lf = np.geomspace(4, 1200, 60)
    ax.axhline(1 / 6, color=style.INK_SECONDARY, lw=1.5, ls="--",
               label="all detected: 1/6 (symmetry)")
    ax.plot(lf, [analytic.attenuated_fraction(HALF, (0, 0, 0), L) for L in lf],
            color=style.INK_SECONDARY, lw=1.5, label="unscattered: (1/4π)∫exp(−r/Ls)dΩ")
    errbar(ax, lengths, p_det, n, "MC, all detected", style.BLUE, "o")
    errbar(ax, lengths, p_unsc, n, "MC, detected without scattering", style.ORANGE, "s")
    ax.set_xscale("log")
    ax.set_ylabel("fraction of emitted photons")
    ax.set_title(f"Rayleigh scattering, source at the centre of a black cube\n"
                 f"({n:.0e} photons per point)")
    ax.legend(loc="center right")
    style.pull_panel(axp, lengths, pulls_det, style.BLUE, logx=True)
    axp.plot(lengths, pulls_unsc, "s", color=style.ORANGE, ms=5, ls="none")
    axp.set_xlabel("Rayleigh scattering length Ls (mm)")
    style.save(fig, os.path.join(outdir, "validation_rayleigh.png"))
    results["rayleigh"] = dict(L=lengths.tolist(), det=p_det.tolist(), unscattered=p_unsc.tolist(),
                               unscattered_analytic=p_unsc_an.tolist(),
                               pulls_det=pulls_det.tolist(), pulls_unscattered=pulls_unsc.tolist())


def validate_time(r, n, outdir, results):
    tau = 4.0
    src = (-7.0, 4.0, 2.0)
    rec = r.run(n, src_x=src[0], src_y=src[1], src_z=src[2], seed=5, **dict(BLACK_BOX, tau=tau))
    det = rec["fate"] == 0
    t_emit = rec["t"] - rec["path"] * N_SCINT / C_MM_NS
    ks = kstest(t_emit, "expon", args=(0, tau))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    bins = np.linspace(0, 30, 61)
    axs[0].hist(t_emit, bins=bins, histtype="step", color=style.BLUE, lw=2,
                label="MC: t − path·n/c")
    centres = 0.5 * (bins[1:] + bins[:-1])
    w = bins[1] - bins[0]
    axs[0].plot(centres, len(t_emit) * w * np.exp(-centres / tau) / tau,
                color=style.INK_SECONDARY, lw=1.5, ls="--", label=f"Exp(τ = {tau} ns)")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("reconstructed emission time (ns)")
    axs[0].set_ylabel("photons / bin")
    axs[0].set_title(f"Emission time: KS D = {ks.statistic:.4f}, p = {ks.pvalue:.2f}")
    axs[0].legend()

    # Time of flight of detected photons: straight lines from the source to the
    # readout face, bounded by the perpendicular distance and the farthest corner.
    tof = rec["path"][det] * N_SCINT / C_MM_NS
    h = HALF[2] - src[2]
    r_max = np.sqrt((HALF[0] + abs(src[0]))**2 + (HALF[1] + abs(src[1]))**2 + h**2)
    t_min, t_max = h * N_SCINT / C_MM_NS, r_max * N_SCINT / C_MM_NS
    inside = np.mean((tof >= t_min * (1 - 1e-6)) & (tof <= t_max * (1 + 1e-6)))
    axs[1].hist(tof, bins=np.linspace(0.9 * t_min, 1.05 * t_max, 80), histtype="step",
                color=style.BLUE, lw=2, label="MC: path·n/c of detected photons")
    for tv, name in [(t_min, "n·h/c (perpendicular)"), (t_max, "n·r/c (farthest corner)")]:
        axs[1].axvline(tv, color=style.INK_SECONDARY, lw=1.5, ls="--")
        axs[1].annotate(name, (tv, 0.55), xycoords=("data", "axes fraction"), rotation=90,
                        ha="right", va="center", fontsize=9, color=style.INK_SECONDARY)
    axs[1].set_xlabel("time of flight (ns)")
    axs[1].set_ylabel("photons / bin")
    axs[1].set_title(f"Time of flight of detected photons ({inside:.2%} within the bounds)")
    axs[1].legend(loc="upper center")
    style.save(fig, os.path.join(outdir, "validation_time.png"))
    results["emission_time"] = dict(ks_d=float(ks.statistic), ks_p=float(ks.pvalue),
                                    tof_inside_bounds=float(inside))


def plot_pull_summary(results, outdir):
    pulls = []
    for key in ("solid_angle_x0", "solid_angle_x15", "absorption", "escape_cone"):
        pulls += results[key]["pulls"]
    pulls += results["rayleigh"]["pulls_det"] + results["rayleigh"]["pulls_unscattered"]
    pulls = np.array(pulls)
    hit = np.array(results["hitmap"]["pulls"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    for ax, data, title in [(axs[0], pulls, f"all scan points ({len(pulls)})"),
                            (axs[1], hit, f"hit-map bins ({len(hit)})")]:
        bins = np.linspace(-4.5, 4.5, 19)
        ax.hist(data, bins=bins, histtype="stepfilled", color=style.BLUE, alpha=0.85,
                edgecolor=style.SURFACE, lw=2, label="pulls")
        x = np.linspace(-4.5, 4.5, 200)
        ax.plot(x, len(data) * (bins[1] - bins[0]) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi),
                color=style.INK_SECONDARY, lw=1.5, label="N(0, 1)")
        ax.set_title(f"{title}: mean {data.mean():+.2f}, rms {data.std():.2f}")
        ax.set_xlabel("pull (σ)")
        ax.legend()
    axs[0].set_ylabel("count")
    style.save(fig, os.path.join(outdir, "validation_pulls.png"))
    results["pull_summary"] = dict(n=len(pulls), mean=float(pulls.mean()), rms=float(pulls.std()),
                                   hit_mean=float(hit.mean()), hit_rms=float(hit.std()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True, help="path to the optphot executable")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    ap.add_argument("--n", type=float, default=1e6, help="photons per scan point")
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--prefix", default="", help="prefix for figure file names, e.g. 'gpu_'")
    ap.add_argument("--summary", default="results/validation_summary.json")
    args = ap.parse_args()
    style.apply()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    n = int(args.n)
    results = {"backend": args.backend, "photons_per_point": n}
    outdir = args.outdir
    if args.prefix:  # write e.g. docs/figures/gpu_validation_*.png
        style.FILE_PREFIX = args.prefix
    with tempfile.TemporaryDirectory() as tmp:
        r = Runner(args.exe, args.backend, args.threads, tmp)
        validate_solid_angle(r, n, outdir, results)
        validate_absorption(r, n, outdir, results)
        validate_hitmap(r, 4 * n, outdir, results)
        validate_escape_cone(r, max(n // 5, 100000), outdir, results)
        validate_rayleigh(r, n, outdir, results)
        validate_time(r, n, outdir, results)
    plot_pull_summary(results, outdir)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    ps = results["pull_summary"]
    print(f"{ps['n']} scan points: pull mean {ps['mean']:+.2f}, rms {ps['rms']:.2f}; "
          f"hit map chi2/ndf = {results['hitmap']['chi2']:.0f}/{results['hitmap']['ndf']} "
          f"(p = {results['hitmap']['p']:.2f}); "
          f"emission-time KS p = {results['emission_time']['ks_p']:.2f}")
    print("summary written to", args.summary)


if __name__ == "__main__":
    main()
