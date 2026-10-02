#!/usr/bin/env python3
"""Estimates the SIMT efficiency lost to *history-length divergence* in the
history-based GPU kernel, using photon records (from any backend).

In transport_records_kernel thread i transports photon i, so a warp holds 32
consecutive photons and keeps running until its longest history is done;
lanes whose photon has finished idle. If every step costs roughly the same,

    warp efficiency = sum(steps) / sum_over_warps(32 * max steps in warp)

is the fraction of issued lane-steps that do useful work. The script also
evaluates periodic compaction: every k steps the alive photons are packed
into full warps (k = 1 is a fully event-based stepping loop).

usage: python python/divergence_estimate.py --exe build/apps/optphot [--n 1e6]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optphot_analysis import io, style  # noqa: E402

SETUPS = {
    "black walls": "configs/solid_angle.cfg",
    "polished (default)": "configs/default.cfg",
    "Lambertian wrap": "configs/wrapped_lambertian.cfg",
}
WARP = 32


def warp_efficiency(steps):
    n = len(steps) // WARP * WARP
    w = steps[:n].reshape(-1, WARP)
    return w.sum() / (WARP * w.max(axis=1)).sum()


def periodic_compaction_efficiency(steps, k):
    """Efficiency if every k steps the still-alive photons are compacted
    (keeping their order) into full warps: k = 1 is a fully event-based
    stepping loop, k = infinity the plain history-based kernel."""
    issued, remaining = 0, steps.copy()
    while len(remaining):
        pad = (-len(remaining)) % WARP
        w = np.concatenate([remaining, np.zeros(pad, np.int64)]).reshape(-1, WARP)
        issued += (WARP * np.minimum(w.max(axis=1), k)).sum()
        remaining = remaining[remaining > k] - k
    return steps.sum() / issued


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--n", type=float, default=1e6)
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()
    style.apply()
    colors = [style.BLUE, style.ORANGE, style.AQUA]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    print("| setup | mean steps | max steps | history-based | compact every 8 steps | "
          "every 4 | every 2 | every step |")
    print("|---|---|---|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for (label, cfg), color in zip(SETUPS.items(), colors):
            out = os.path.join(tmp, label.split()[0])
            subprocess.run([os.path.abspath(args.exe), "--config", cfg, "--threads", "0",
                            "--n_photons", str(int(args.n)), "--out", out],
                           check=True, stdout=subprocess.DEVNULL)
            rec = io.load_records(out)
            # One loop iteration of transport_photon = one boundary interaction
            # or one scattering (a bulk absorption ends the last iteration).
            steps = (rec["n_boundary"] + rec["n_scatter"]).astype(np.int64)
            steps += rec["fate"] == 1
            eff = warp_efficiency(steps)
            compacted = [periodic_compaction_efficiency(steps, k) for k in (8, 4, 2, 1)]
            print(f"| {label} | {steps.mean():.2f} | {steps.max()} | {eff:.0%} | "
                  + " | ".join(f"{c:.0%}" for c in compacted) + " |")
            bins = np.unique(np.geomspace(1, steps.max() + 1, 60).astype(int))
            ax.hist(steps, bins=bins, histtype="step", lw=2, color=color,
                    label=f"{label}: mean {steps.mean():.1f}, warp eff. {eff:.0%}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("transport steps per photon")
    ax.set_ylabel("photons")
    ax.set_title(f"History length distribution ({args.n:.0e} photons per setup)")
    ax.legend(loc="upper right")
    style.save(fig, os.path.join(args.outdir, "divergence_history_length.png"))


if __name__ == "__main__":
    main()
