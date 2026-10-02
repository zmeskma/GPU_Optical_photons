"""Reading optphot output directories (.npy arrays + meta.json)."""
import json
import os

import numpy as np

FATES = ["detected", "absorbed_bulk", "absorbed_surface", "escaped", "max_steps"]
RECORD_FIELDS = ["fate", "n_boundary", "n_scatter", "t", "x", "y", "z", "path"]


def load_meta(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        return json.load(f)


def load_records(run_dir):
    """Returns a dict of numpy arrays (one entry per photon) plus 'meta'."""
    rec = {k: np.load(os.path.join(run_dir, k + ".npy")) for k in RECORD_FIELDS}
    rec["meta"] = load_meta(run_dir)
    return rec


def load_tally(run_dir):
    tally = {k: np.load(os.path.join(run_dir, k + ".npy"))
             for k in ["fate_counts", "time_hist", "xy_hist"]}
    tally["meta"] = load_meta(run_dir)
    return tally


def fate_counts(rec):
    return np.bincount(rec["fate"], minlength=len(FATES))
