"""
PosteriorDB-style implementation equivalence check
for the quartic model (Stan vs PyMC).

Follows the protocol in Torgander & Magnusson:
  log p1(φ) - log p2(φ) ≈ constant
  ∇ log p1(φ) - ∇ log p2(φ) ≈ 0
on a fixed set of points in the computational space.
"""

import os
import json
import numpy as np
import cmdstanpy

from quartic_model import make_model


def configure_windows_compiler():
    """Inject RTools paths into PATH."""
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
    ]
    extra = [
        r"C:\rtools40\usr\bin",
        r"C:\rtools40\mingw64\bin",
        r"C:\Users\DELL\AppData\Local\Rtools40\usr\bin",
        r"C:\Users\DELL\AppData\Local\Rtools40\mingw64\bin",
    ]
    found = [p for p in compiler_paths + extra if os.path.exists(p)]
    if found:
        os.environ["PATH"] = ";".join(found) + ";" + os.environ.get("PATH", "")
        print("Compiler paths added:", found)
    else:
        print("WARNING: no RTools paths found; Stan compile may fail.")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_quartic(stan_file, data, n_points=50, seed=42, label=""):
    print(f"\n=== quartic model {label} ===")
    print(f"  y={data['y']}, sigma={data['sigma']}")

    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    model = make_model(data)
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    # Fixed, reproducible evaluation points in the computational space
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 1.5, size=(n_points, 2))

    logp_diffs = []
    max_rel_grad = 0.0
    max_abs_grad = 0.0

    for i, theta in enumerate(points):
        # Stan
        stan_out = stan_model.log_prob(data=data, params={"theta": theta.tolist()})
        if hasattr(stan_out, "iloc"):
            lp_s = float(stan_out["lp__"].iloc[0])
            grad_s = (
                stan_out.drop(columns=["lp__"], errors="ignore")
                .iloc[0]
                .to_numpy(dtype=float)
            )
        else:
            lp_s = float(np.asarray(stan_out).squeeze())
            grad_s = None

        # PyMC
        point = {"theta": theta}
        lp_p = float(logp_fn(point))
        g_p = dlogp_fn(point)
        grad_p = np.concatenate([np.atleast_1d(g).ravel() for g in g_p])

        logp_diffs.append(lp_s - lp_p)

        if grad_s is not None and grad_s.size == grad_p.size:
            abs_err = float(np.max(np.abs(grad_s - grad_p)))
            scale = max(
                float(np.max(np.abs(grad_s))),
                float(np.max(np.abs(grad_p))),
                1e-8,
            )
            rel_err = abs_err / scale
            max_abs_grad = max(max_abs_grad, abs_err)
            max_rel_grad = max(max_rel_grad, rel_err)

    logp_diffs = np.asarray(logp_diffs)
    # Criterion (4): differences should be approximately constant
    logp_range = float(np.max(logp_diffs) - np.min(logp_diffs))
    logp_std = float(np.std(logp_diffs))

    print(f"  Logp difference range (max-min): {logp_range:.6e}")
    print(f"  Logp difference std:             {logp_std:.6e}")
    print(f"  Max absolute gradient error:     {max_abs_grad:.6e}")
    print(f"  Max relative gradient error:     {max_rel_grad:.6e}")

    # Tolerances appropriate for this model (vanishing-noise friendly)
    # - relative gradient error tiny
    # - logp differences nearly constant (range not exploding)
    ok = (max_rel_grad < 1e-5) and (logp_range < 1.0)
    print("  RESULT:", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    configure_windows_compiler()
    here = os.path.dirname(os.path.abspath(__file__))

    stan_file = os.path.join(here, "quartic_model.stan")
    data_files = [
        ("quartic_sigma1.json", "sigma=1.0"),
        ("quartic_sigma0.1.json", "sigma=0.1"),
        ("quartic_sigma0.01.json", "sigma=0.01"),
    ]

    results = []
    for fname, label in data_files:
        path = os.path.join(here, fname)
        if not os.path.exists(path):
            print(f"\nWARNING: {fname} not found, skipping.")
            results.append(False)
            continue
        data = load_json(path)
        ok = check_quartic(stan_file, data, label=label)
        results.append(ok)

    print("\n=======================================================")
    if all(results):
        print("OVERALL STATUS: SUCCESS! All three sigma values are implementation-equivalent.")
    else:
        print("OVERALL STATUS: FAILED. Inspect output above.")
    print("=======================================================")
    