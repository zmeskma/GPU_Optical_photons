#!/usr/bin/env python3
"""Compares two optphot record directories photon by photon and
statistically, e.g. two builds (library math vs portable math, CPU vs GPU)
run with the same seed.

usage: python python/compare_runs.py DIR_A DIR_B [--json out.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optphot_analysis import compare, io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    a, b = io.load_records(args.a), io.load_records(args.b)
    same_seed = a["meta"]["seed"] == b["meta"]["seed"]
    out = {"a": args.a, "b": args.b, "same_seed": same_seed}
    if same_seed:
        h = compare.history_agreement(a, b)
        out["history"] = h
        print(f"photon-by-photon ({h['n']} photons, same seed):")
        print(f"  bitwise identical     {h['frac_bitwise']:.6f}")
        print(f"  same discrete history {h['frac_same_discrete']:.6f}")
        print(f"  fate differs          {h['fate_differs']}")
        print(f"  max |dt| {h['max_abs_dt_ns']:.3g} ns, max |dpos| {h['max_abs_dpos_mm']:.3g} mm")
    half = (a["meta"]["size_x"] / 2, a["meta"]["size_y"] / 2)
    d = compare.distribution_tests(a, b, *half)
    out["distributions"] = d
    note = " (same seed: samples are correlated, tests pass trivially)" if same_seed else ""
    print(f"distributions{note}:")
    print(f"  efficiency {d['eff_a']:.6f} +- {d['err_a']:.6f} vs {d['eff_b']:.6f} +- "
          f"{d['err_b']:.6f}, z = {d['eff_z']:.2f}")
    print(f"  arrival time chi2/ndf {d['time_chi2']:.1f}/{d['time_ndf']} (p {d['time_chi2_p']:.3f}),"
          f" KS p {d['time_ks_p']:.3f}; hit map chi2/ndf {d['xy_chi2']:.1f}/{d['xy_ndf']} "
          f"(p {d['xy_chi2_p']:.3f})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
