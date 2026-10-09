#!/usr/bin/env python3
"""Event-based tracking (periodic compaction of the alive photons) versus the
history-based kernel: model prediction and, if available, measurement.

Per setup:
  1. survival curve: fraction of photons still alive after s steps (the heavy
     tail is why a single compaction is not enough);
  2. SIMT lane efficiency versus steps per launch k: model of
     divergence_estimate.py (from recorded history lengths) and the value
     counted on the GPU by the event-based kernel;
  3. throughput: the prediction "history-based throughput x efficiency(k) /
     efficiency(history-based)", an upper bound that ignores the cost of
     compaction, kernel launches and moving the photon state through global
     memory, against the measured event-based throughput.

Measurements come from apps/optphot_event_bench (one CSV per setup, named
event_<config>.csv in --event_dir). Without them, the prediction is scaled
from a history-based tally run.

usage: python python/plot_event_based.py --exe build/apps/optphot
           [--event_dir runs/event_based] [--n 1e6] [--outdir runs/event_based]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from divergence_estimate import periodic_compaction_efficiency, warp_efficiency  # noqa: E402
from optphot_analysis import io, style  # noqa: E402

SETUPS = {
    "polished cube (default)": ("default", style.BLUE, "o"),
    "Lambertian wrap": ("wrapped_lambertian", style.ORANGE, "s"),
}
INTERVALS = [1, 2, 4, 8, 16, 32, 64, 128]


def history_steps(exe, cfg, n, backend, out):
    subprocess.run([exe, "--config", f"configs/{cfg}.cfg", "--backend", backend, "--threads", "0",
                    "--n_photons", str(n), "--out", out], check=True, stdout=subprocess.DEVNULL)
    rec = io.load_records(out)
    # Same step definition as divergence_estimate.py.
    steps = (rec["n_boundary"] + rec["n_scatter"]).astype(np.int64)
    return steps + (rec["fate"] == 1)


def tally_throughput(exe, cfg, n, backend):
    res = subprocess.run([exe, "--config", f"configs/{cfg}.cfg", "--backend", backend,
                          "--threads", "0", "--mode", "tally", "--n_photons", str(n)],
                         check=True, capture_output=True, text=True)
    return float(re.search(r"throughput:\s*([0-9.eE+-]+)", res.stdout).group(1))


def load_measured(path):
    """Median over repeats: {('history'|'event', k): dict of floats}."""
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups[(r["method"], int(r["k"]))].append(r)
    med = {}
    for key, rows in groups.items():
        med[key] = {c: float(np.median([float(r[c]) for r in rows]))
                    for c in ("photons_per_s", "transport_ms", "kernels_ms", "launches", "simt_eff",
                              "bitwise_identical", "n_photons")}
        med[key]["lo"] = min(float(r["photons_per_s"]) for r in rows)
        med[key]["hi"] = max(float(r["photons_per_s"]) for r in rows)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to optphot")
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--n", type=float, default=1e6, help="photons for the history lengths")
    ap.add_argument("--event_dir", help="directory with event_<config>.csv from optphot_event_bench")
    ap.add_argument("--n_speed", type=float, default=1e7,
                    help="photons for the tally throughput (only without --event_dir)")
    ap.add_argument("--outdir", default="runs/event_based")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    style.apply()
    exe = os.path.abspath(args.exe)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.9), layout="constrained")
    print("| setup | k | model SIMT eff. | measured SIMT eff. | predicted photons/s | "
          "measured photons/s | measured / history | launches |")
    print("|---|---|---|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for label, (cfg, color, marker) in SETUPS.items():
            steps = history_steps(exe, cfg, int(args.n), args.backend, os.path.join(tmp, cfg))
            eff_h = warp_efficiency(steps)
            eff = np.array([periodic_compaction_efficiency(steps, k) for k in INTERVALS])
            meas = None
            if args.event_dir:
                path = os.path.join(args.event_dir, f"event_{cfg}.csv")
                if os.path.exists(path):
                    meas = load_measured(path)
            if meas:
                speed_h = meas[("history", 0)]["photons_per_s"]
                base_label = "history-based records kernel, measured"
            else:
                speed_h = tally_throughput(exe, cfg, int(args.n_speed), args.backend)
                base_label = "history-based tally kernel, measured"
            proj = speed_h * eff / eff_h

            # 1. Survival curve.
            s = np.arange(0, steps.max() + 1)
            alive = 1.0 - np.searchsorted(np.sort(steps), s, side="right") / len(steps)
            axs[0].step(s, np.where(alive > 0, alive, np.nan), where="post", color=color,
                        label=f"{label}: mean {steps.mean():.1f}, max {steps.max()}")

            # 2. SIMT efficiency: model line, measured markers.
            axs[1].plot(INTERVALS, eff, color=color, lw=1.5, alpha=0.6,
                        label=f"{label}: model")
            axs[1].axhline(eff_h, color=color, ls=":", lw=1.2)
            # 3. Throughput: prediction (upper bound) vs measurement.
            axs[2].plot(INTERVALS, proj, color=color, lw=1.5, ls="--", alpha=0.7,
                        label=f"{label}: predicted (upper bound)")
            axs[2].axhline(speed_h, color=color, ls=":", lw=1.2)
            axs[2].annotate(f"{base_label}: {speed_h:.2g}/s", (INTERVALS[-1], speed_h),
                            xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8,
                            color=color)
            if meas:
                ks = [k for k in INTERVALS if ("event", k) in meas]
                m_eff = [meas[("event", k)]["simt_eff"] for k in ks]
                axs[1].plot(ks, m_eff, color=color, marker=marker, ms=6, ls="none",
                            label=f"{label}: counted on the GPU")
                y = np.array([meas[("event", k)]["photons_per_s"] for k in ks])
                lo = y - np.array([meas[("event", k)]["lo"] for k in ks])
                hi = np.array([meas[("event", k)]["hi"] for k in ks]) - y
                axs[2].errorbar(ks, y, yerr=[lo, hi], color=color, marker=marker, ms=6, lw=2,
                                capsize=0, label=f"{label}: measured, event-based")
                kern = [meas[("event", k)]["n_photons"] / (meas[("event", k)]["kernels_ms"] * 1e-3)
                        for k in ks]
                axs[2].plot(ks, kern, color=color, marker=marker, ms=5, lw=1, mfc=style.SURFACE,
                            label=f"{label}: measured, kernels only (no launch gaps)")
                for k, p_eff, p_speed in zip(INTERVALS, eff, proj):
                    if ("event", k) not in meas:
                        continue
                    e = meas[("event", k)]
                    print(f"| {label} | {k} | {p_eff:.1%} | {e['simt_eff']:.1%} | {p_speed:.2e} | "
                          f"{e['photons_per_s']:.2e} | {e['photons_per_s'] / speed_h:.2f}x | "
                          f"{e['launches']:.0f} |")
                    if e["bitwise_identical"] != e["n_photons"]:
                        print(f"WARNING: {label} k={k}: event-based records differ from history-based")

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("transport steps s")
    axs[0].set_ylabel("fraction of photons still alive after s steps")
    axs[0].set_title("Heavy tail: few photons live very long")
    axs[0].legend(loc="lower left")
    for ax in axs[1:]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(INTERVALS, [str(k) for k in INTERVALS])
        ax.set_xlabel("steps per launch k (compact the alive photons after each launch)")
    axs[1].set_ylim(0, 1.05)
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axs[1].set_ylabel("lanes doing useful work")
    axs[1].set_title("SIMT efficiency (dotted: history-based)")
    axs[1].legend(loc="upper right")
    axs[2].set_yscale("log")
    axs[2].set_ylabel(f"{args.backend.upper()} transport throughput (photons / s)")
    axs[2].set_title("Throughput: prediction vs measurement (dotted: history-based)")
    axs[2].legend(loc="upper right", fontsize=7.5)
    title = f"Event-based tracking: periodic compaction of the alive photons ({args.n:.0e} photons)"
    if args.label:
        title += "\n" + args.label
    fig.suptitle(title)
    style.save(fig, os.path.join(args.outdir, "event_based.png"))


if __name__ == "__main__":
    main()
