#!/usr/bin/env python3
"""Plots optphot_bench CSV output: throughput vs number of photons per
backend/mode, and the GPU time breakdown (kernel vs device-to-host copy).

usage: python python/plot_benchmarks.py bench.csv [more.csv ...] --label "default cube"
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optphot_analysis import style  # noqa: E402

# Fixed identity per backend (colour follows the entity, never its rank).
SERIES = {
    "cpu1": ("CPU, 1 thread", style.BLUE, "o"),
    "cpu_omp": ("CPU, all threads (OpenMP)", style.AQUA, "^"),
    "gpu": ("GPU", style.ORANGE, "s"),
}


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def median_by(rows, key_fields, value):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(float(r[value]))
    return {k: (np.median(v), np.min(v), np.max(v)) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--label", default="", help="description of the benchmark setup")
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--prefix", default="benchmark")
    args = ap.parse_args()
    style.apply()
    rows = load(args.csv)
    devices = sorted({r["device"] for r in rows if r["backend"] == "gpu"})
    threads = sorted({int(r["threads"]) for r in rows if r["backend"] == "cpu_omp"})

    # 1. Throughput vs N, one panel per mode (same y scale -> comparable).
    modes = [m for m in ("tally", "records") if any(r["mode"] == m for r in rows)]
    fig, axs = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.6), sharey=True,
                            layout="constrained", squeeze=False)
    med = median_by(rows, ("backend", "mode", "n_photons"), "photons_per_s_kernel")
    med_total = median_by(rows, ("backend", "mode", "n_photons"), "photons_per_s_total")
    for ax, mode in zip(axs[0], modes):
        for backend, (label, color, marker) in SERIES.items():
            pts = sorted((int(k[2]), v) for k, v in med.items() if k[0] == backend and k[1] == mode)
            if not pts:
                continue
            n = np.array([p[0] for p in pts])
            y = np.array([p[1][0] for p in pts])
            lo = y - np.array([p[1][1] for p in pts])
            hi = np.array([p[1][2] for p in pts]) - y
            ax.errorbar(n, y, yerr=[lo, hi], color=color, marker=marker, ms=6, lw=2,
                        capsize=0, label=label + (" (transport only)" if backend == "gpu" else ""))
            if backend == "gpu" and mode == "records":
                yt = np.array([med_total[(backend, mode, str(k))][0] for k in n])
                ax.plot(n, yt, color=color, marker=marker, ms=6, lw=1.5, ls="--",
                        mfc=style.SURFACE, label="GPU incl. allocation + D2H copy")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("photons per run")
        ax.set_title(f"{mode} mode")
        ax.legend(loc="lower right")
    axs[0][0].set_ylabel("throughput (photons / s)")
    all_y = [v[1] for v in med.values()] + [v[2] for v in med.values()]
    axs[0][0].set_ylim(min(all_y) / 2, max(all_y) * 3)
    sub = []
    if args.label:
        sub.append(args.label)
    if devices and not any("GTX" in x or "RTX" in x or "Tesla" in x for x in sub):
        sub.append("GPU: " + ", ".join(devices))
    if threads:
        sub.append(f"CPU: {threads[-1]} threads")
    fig.suptitle("Throughput (median of repeats; bars = min/max)" +
                 ("\n" + "; ".join(sub) if sub else ""))
    style.save(fig, os.path.join(args.outdir, f"{args.prefix}_throughput.png"))

    # 2. GPU records-mode time breakdown: where does the time go?
    gpu_rec = [r for r in rows if r["backend"] == "gpu" and r["mode"] == "records"]
    if gpu_rec:
        ns = sorted({int(r["n_photons"]) for r in gpu_rec})

        def gpu_median(field):
            m = median_by(gpu_rec, ("n_photons",), field)
            return np.array([m[(str(n),)][0] for n in ns])

        total = gpu_median("total_ms")
        kernel, d2h, alloc = gpu_median("kernel_ms"), gpu_median("d2h_ms"), gpu_median("alloc_ms")
        # Whatever is not kernel, copy or device allocation happens on the host:
        # mostly resizing (zero-filling, first-touch page faults) the output vectors.
        host = np.clip(total - kernel - d2h - alloc, 0, None)
        parts = [(kernel, "transport kernel", style.ORANGE),
                 (d2h, "device → host copy", style.BLUE),
                 (alloc, "device allocation", style.AQUA),
                 (host, "host buffers (vector resize / zero-fill)", style.YELLOW)]
        fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
        x = np.arange(len(ns))
        bottom = np.zeros(len(ns))
        for vals, label, color in parts:
            frac = vals / total
            ax.bar(x, frac, bottom=bottom, color=color, width=0.6, label=label,
                   edgecolor=style.SURFACE, linewidth=2)
            bottom += frac
        for xi, t in zip(x, total):
            ax.annotate(f"{t:.1f} ms", (xi, 1.0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=style.INK_SECONDARY)
        ax.set_xticks(x, [f"{n:.0e}" for n in ns])
        ax.set_xlabel("photons per run")
        ax.set_ylabel("fraction of wall time")
        ax.set_ylim(0, 1.12)
        ax.set_title("GPU records mode: where the wall time goes (median total above bars)")
        ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.16))
        style.save(fig, os.path.join(args.outdir, f"{args.prefix}_gpu_breakdown.png"))

    # Markdown table for the README.
    print("| backend | mode | photons | median throughput (photons/s) |")
    print("|---|---|---|---|")
    for (backend, mode, n), (m, _, _) in sorted(med.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                              int(kv[0][2]))):
        print(f"| {backend} | {mode} | {int(n):.0e} | {m:.3g} |")


if __name__ == "__main__":
    main()
