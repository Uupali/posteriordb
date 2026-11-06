#!/usr/bin/env python3
"""
Build posteriordb data files for the German credit models in
Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR 15, Section 4.1.2-4.1.3.

Outputs (in --outdir):
  german_credit.json                 N=1000, D=24  main effects        (for LR)
  german_credit_interactions.json    N=1000, D=300 main + pairwise     (for HLR)
  *.json.zip                         zipped copies (posteriordb stores data zipped)
  *.info.json                        DRAFT info files - check fields against
                                     existing entries in posterior_database/data/info

Each data JSON has the form
  {"N": int, "D": int, "X": [[...], ...] (N x D, row-major), "y": [0/1, ...]}
which matches the Stan data block:
  int N; int D; matrix[N, D] X; array[N] int<lower=0,upper=1> y;

Preprocessing conventions (documented here so the Stan and PyMC sides match):
  * Source: UCI Statlog German credit, file german.data-numeric
    (24 numeric attributes + class column; class 1 = good, 2 = bad).
  * y = 1 for good (grant credit), 0 for bad (deny). This is the paper's
    +1/-1 coding mapped to {0,1}.
  * Main effects are standardized to mean 0, variance 1, using the
    population SD (ddof=0).
  * Interactions are products of pairs of *standardized* main effects,
    all pairs j < k, in itertools.combinations order, appended after the
    24 main effects. No squared terms. 24 + 276 = 300 columns.
  * By default the interaction columns are then re-standardized
    (mean 0, variance 1) so that all 300 predictors are on the same scale,
    matching the paper's statement that predictors are normalized.
    Use --no-restandardize-interactions to keep raw products instead.

Usage:
  python make_german_credit_data.py                      # downloads from UCI
  python make_german_credit_data.py --input german.data-numeric
  python make_german_credit_data.py --outdir out/ --no-restandardize-interactions

Requires only numpy.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data-numeric"
)

N_EXPECTED = 1000
D_EXPECTED = 24


def load_raw(path=None):
    """Return the raw (1000, 25) array from a local file or from UCI."""
    if path is not None:
        with open(path, "r") as f:
            text = f.read()
        source = os.path.abspath(path)
    else:
        print(f"Downloading {UCI_URL} ...", file=sys.stderr)
        with urllib.request.urlopen(UCI_URL) as resp:
            text = resp.read().decode("ascii")
        source = UCI_URL

    rows = [line.split() for line in text.strip().splitlines() if line.strip()]
    raw = np.array(rows, dtype=float)

    if raw.shape != (N_EXPECTED, D_EXPECTED + 1):
        raise ValueError(
            f"Expected shape ({N_EXPECTED}, {D_EXPECTED + 1}), got {raw.shape}. "
            "Is this the german.data-numeric file (not german.data)?"
        )
    return raw, source


def standardize(M, what):
    """Column-wise standardization with population SD; fail on constant columns."""
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=0)
    bad = np.where(sd < 1e-12)[0]
    if bad.size:
        raise ValueError(f"Zero-variance {what} columns at indices {bad.tolist()}")
    return (M - mu) / sd


def build(raw, restandardize_interactions=True):
    X_raw = raw[:, :D_EXPECTED]
    cls = raw[:, D_EXPECTED].astype(int)

    if not set(np.unique(cls)) <= {1, 2}:
        raise ValueError(f"Unexpected class labels: {np.unique(cls)}")
    y = (cls == 1).astype(int)  # 1 = good -> 1, 2 = bad -> 0

    X_main = standardize(X_raw, "main-effect")

    pairs = list(itertools.combinations(range(D_EXPECTED), 2))
    X_int = np.column_stack([X_main[:, j] * X_main[:, k] for j, k in pairs])
    if restandardize_interactions:
        X_int = standardize(X_int, "interaction")

    X_full = np.hstack([X_main, X_int])
    assert X_full.shape == (N_EXPECTED, D_EXPECTED + len(pairs)) == (1000, 300)
    return X_main, X_full, y, pairs


def to_json_dict(X, y):
    return {
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "X": X.tolist(),  # full float64 precision via repr
        "y": y.tolist(),
    }


def write_data(outdir, name, data):
    json_path = os.path.join(outdir, f"{name}.json")
    payload = json.dumps(data, separators=(",", ":"))
    with open(json_path, "w") as f:
        f.write(payload)

    zip_path = json_path + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname=f"{name}.json")

    digest = hashlib.sha256(payload.encode()).hexdigest()
    return json_path, zip_path, digest


def write_info(outdir, name, title, description, source):
    # DRAFT: compare against an existing file in posterior_database/data/info/
    info = {
        "name": name,
        "keywords": ["german credit", "logistic regression", "NUTS paper"],
        "title": title,
        "description": description,
        "urls": [
            "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            "https://jmlr.org/papers/v15/hoffman14a.html",
        ],
        "references": ["hoffman2014nuts"],
        "added_date": datetime.date.today().isoformat(),
        "added_by": "TODO",
        "_source_file_used": source,
    }
    path = os.path.join(outdir, f"{name}.info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="local path to german.data-numeric (else download)")
    p.add_argument("--outdir", default=".", help="output directory")
    p.add_argument("--no-restandardize-interactions", action="store_true",
                   help="keep raw products of standardized main effects")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw, source = load_raw(args.input)
    restd = not args.no_restandardize_interactions
    X_main, X_full, y, pairs = build(raw, restandardize_interactions=restd)

    # Sanity checks
    print(f"N = {len(y)}, good (y=1) = {y.sum()}, bad (y=0) = {len(y) - y.sum()}")
    print(f"main effects: max |mean| = {np.abs(X_main.mean(0)).max():.2e}, "
          f"SD range = [{X_main.std(0).min():.6f}, {X_main.std(0).max():.6f}]")
    print(f"full matrix:  shape = {X_full.shape}, "
          f"SD range = [{X_full.std(0).min():.6f}, {X_full.std(0).max():.6f}]")

    int_note = ("re-standardized to mean 0 / variance 1" if restd
                else "left as raw products (not re-standardized)")

    outputs = [
        ("german_credit", X_main,
         "German credit data (24 standardized numeric predictors)",
         "UCI Statlog German credit (german.data-numeric), 1000 customers. "
         "24 numeric predictors standardized to mean 0, variance 1 (population SD). "
         "y = 1 good credit risk, 0 bad. As used in Hoffman & Gelman (2014), Sec. 4.1.2."),
        ("german_credit_interactions", X_full,
         "German credit data with pairwise interactions (300 predictors)",
         "UCI Statlog German credit (german.data-numeric), 1000 customers. "
         "24 standardized main effects followed by all 276 pairwise products "
         "(j < k, itertools.combinations order) of the standardized main effects, "
         f"{int_note}. No squared terms. y = 1 good, 0 bad. "
         "As used in Hoffman & Gelman (2014), Sec. 4.1.3."),
    ]

    print()
    for name, X, title, desc in outputs:
        json_path, zip_path, digest = write_data(args.outdir, name, to_json_dict(X, y))
        info_path = write_info(args.outdir, name, title, desc, source)
        print(f"{name}:")
        print(f"  {json_path}\n  {zip_path}\n  {info_path}")
        print(f"  sha256(json) = {digest}")

    # Column map, handy for the PyMC side and for debugging
    colmap = [f"x{j + 1}" for j in range(D_EXPECTED)] + \
             [f"x{j + 1}:x{k + 1}" for j, k in pairs]
    with open(os.path.join(args.outdir, "german_credit_interactions_columns.txt"), "w") as f:
        f.write("\n".join(colmap) + "\n")


if __name__ == "__main__":
    main()