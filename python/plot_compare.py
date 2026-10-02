#!/usr/bin/env python3
"""Plots the output of optphot_compare (CPU vs GPU) for one or more setups.

Each input is LABEL=DIR where DIR was written by `optphot_compare --out DIR`
and contains cpu/, gpu/ (same seed) and gpu_indep/ (independent seed).

For every setup: arrival-time and hit-position distributions of CPU vs GPU
(independent seeds, so the statistical tests are meaningful), with ratio
panels. Across setups: fraction of photon histories that agree exactly.

usage: python python/plot_compare.py default=results/compare/default \
           lambertian=results/compare/lambertian [--outdir docs/figures]
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optphot_analysis import compare, io, style  # noqa: E402


def ratio_panel(ax, centres, h_cpu, h_gpu, color):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = h_gpu / h_cpu
        err = r * np.sqrt(1 / h_gpu + 1 / h_cpu)
    ok = (h_cpu >= 50) & (h_gpu >= 50)  # skip sparsely populated tail bins
    ax.axhline(1, color=style.INK_SECONDARY, lw=1)
    ax.errorbar(centres[ok], r[ok], yerr=err[ok], fmt="o", ms=3, color=color, elinewidth=1)
    ax.set_ylabel("GPU / CPU")
    ax.set_ylim(0.9, 1.1)


def plot_setup(label, d, outdir):
    cpu = io.load_records(os.path.join(d, "cpu"))
    gpu = io.load_records(os.path.join(d, "gpu_indep"))
    meta = cpu["meta"]
    hx = meta["size_x"] / 2
    stat = compare.distribution_tests(cpu, gpu, hx, meta["size_y"] / 2)
    dc, dg = cpu["fate"] == 0, gpu["fate"] == 0

    fig, axs = plt.subplots(2, 2, figsize=(12, 6.2), sharex="col", layout="constrained",
                            gridspec_kw=dict(height_ratios=[3, 1.2]))
    tb = np.linspace(0, 40, 81)
    tc = 0.5 * (tb[1:] + tb[:-1])
    h1, _ = np.histogram(cpu["t"][dc], tb)
    h2, _ = np.histogram(gpu["t"][dg], tb)
    axs[0, 0].stairs(h1, tb, color=style.BLUE, lw=2, label=f"CPU (seed {meta['seed']})")
    axs[0, 0].stairs(h2, tb, color=style.ORANGE, lw=2, ls="--",
                     label=f"GPU (seed {gpu['meta']['seed']})")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel("detected photons / bin")
    axs[0, 0].set_title(f"Arrival time: χ² p = {stat['time_chi2_p']:.2f}, "
                        f"KS p = {stat['time_ks_p']:.2f}")
    axs[0, 0].legend()
    ratio_panel(axs[1, 0], tc, h1.astype(float), h2.astype(float), style.ORANGE)
    axs[1, 0].set_xlabel("arrival time (ns)")

    xb = np.linspace(-hx, hx, 41)
    xc = 0.5 * (xb[1:] + xb[:-1])
    g1, _ = np.histogram(cpu["x"][dc], xb)
    g2, _ = np.histogram(gpu["x"][dg], xb)
    axs[0, 1].stairs(g1, xb, color=style.BLUE, lw=2, label="CPU")
    axs[0, 1].stairs(g2, xb, color=style.ORANGE, lw=2, ls="--", label="GPU")
    axs[0, 1].set_ylabel("detected photons / bin")
    axs[0, 1].set_title(f"Hit x on readout face: 2-D hit map χ² p = {stat['xy_chi2_p']:.2f}")
    axs[0, 1].legend(loc="lower center")
    ratio_panel(axs[1, 1], xc, g1.astype(float), g2.astype(float), style.ORANGE)
    axs[1, 1].set_xlabel("x (mm)")
    n = len(cpu["fate"])
    fig.suptitle(f"CPU vs GPU, independent seeds — {label} ({n:.0e} photons each): "
                 f"efficiency {stat['eff_a']:.4f} ± {stat['err_a']:.4f} vs "
                 f"{stat['eff_b']:.4f} ± {stat['err_b']:.4f} (z = {stat['eff_z']:+.2f})")
    style.save(fig, os.path.join(outdir, f"compare_{label}.png"))
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setups", nargs="+", help="LABEL=DIR")
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--summary", default="results/compare_summary.json")
    args = ap.parse_args()
    style.apply()
    os.makedirs(args.outdir, exist_ok=True)
    summary = {}
    for item in args.setups:
        label, d = item.split("=", 1)
        with open(os.path.join(d, "compare.json"), encoding="utf-8") as f:
            cj = json.load(f)
        stat = plot_setup(label, d, args.outdir)
        summary[label] = dict(history=cj["history"], independent=stat, device=cj["device"],
                              timing_ms=cj["timing_ms"])

    # History agreement across setups.
    labels = list(summary)
    bitwise = [summary[k]["history"]["frac_bitwise"] for k in labels]
    discrete = [summary[k]["history"]["frac_same_discrete"] for k in labels]
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3.5, 4.2), layout="constrained")
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, discrete, w, color=style.AQUA, label="same discrete history")
    b2 = ax.bar(x + w / 2 + 0.01, bitwise, w, color=style.BLUE, label="bitwise identical record")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.annotate(f"{v:.4%}" if v > 0.999 else f"{v:.1%}",
                        (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=style.INK_SECONDARY)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of photons")
    ax.set_title("CPU vs GPU with the same seed: photon-by-photon agreement")
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.12))
    style.save(fig, os.path.join(args.outdir, "compare_history_agreement.png"))

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("| setup | same discrete history | bitwise identical | eff. z (indep.) | time KS p | "
          "hit-map chi2 p |")
    print("|---|---|---|---|---|---|")
    for k in labels:
        h, s = summary[k]["history"], summary[k]["independent"]
        print(f"| {k} | {h['frac_same_discrete']:.5f} | {h['frac_bitwise']:.4f} | "
              f"{s['eff_z']:+.2f} | {s['time_ks_p']:.2f} | {s['xy_chi2_p']:.2f} |")


if __name__ == "__main__":
    main()
