"""
Compare Stan (CmdStanPy) and PyMC log-density and gradients
for both German credit models at random parameter points.
"""

import os
import sys
import json
import numpy as np
import cmdstanpy
import pymc as pm

from german_credit_lr import make_model as make_lr
from german_credit_hlr import make_model as make_hlr


def configure_windows_compiler():
    """Inject RTools paths into PATH (same as Rosenbrock)."""
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
    ]
    # fallbacks some installs use
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


def check_lr(stan_file, data, n_points=30, seed=42):
    print("\n=== german_credit_lr ===")
    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    model = make_lr(data)
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    rng = np.random.default_rng(seed)
    D = int(data["D"])
    stan_lps, pm_lps = [], []
    max_grad_err = 0.0

    for i in range(n_points):
        alpha = float(rng.normal(0, 2))
        beta = rng.normal(0, 2, size=D)

        # Stan
        stan_out = stan_model.log_prob(
            data=data,
            params={"alpha": alpha, "beta": beta.tolist()},
        )
        # CmdStanPy returns a DataFrame-like result
        if hasattr(stan_out, "iloc"):
            lp_s = float(stan_out["lp__"].iloc[0])
            grad_s = stan_out.drop(columns=["lp__"], errors="ignore").iloc[0].to_numpy(dtype=float)
        else:
            lp_s = float(np.asarray(stan_out).squeeze())
            grad_s = None

        # PyMC
        point = {"alpha": alpha, "beta": beta}
        lp_p = float(logp_fn(point))
        g_p = dlogp_fn(point)
        grad_p = np.concatenate([np.atleast_1d(g).ravel() for g in g_p])
        
        stan_lps.append(lp_s)
        pm_lps.append(lp_p)
        if grad_s is not None and grad_s.size == grad_p.size:
            max_grad_err = max(max_grad_err, float(np.max(np.abs(grad_s - grad_p))))

    diff = np.asarray(stan_lps) - np.asarray(pm_lps)
    var_logp = float(np.var(diff))
    print(f"  Log-density diff variance: {var_logp:.6e}")
    print(f"  Max gradient error:        {max_grad_err:.6e}")
    ok = var_logp < 1e-6 and (max_grad_err < 1e-4 or max_grad_err == 0.0)
    print("  RESULT:", "PASSED" if ok else "FAILED")
    return ok


def check_hlr(stan_file, data, n_points=20, seed=42):
    print("\n=== german_credit_hlr ===")
    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    model = make_hlr(data)
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    rng = np.random.default_rng(seed)
    D = int(data["D"])
    stan_lps, pm_lps = [], []
    max_grad_err = 0.0

    for i in range(n_points):
        alpha = float(rng.normal(0, 2))
        beta = rng.normal(0, 2, size=D)
        sigma2 = float(rng.uniform(20.0, 200.0))  # must be > 0

        stan_out = stan_model.log_prob(
            data=data,
            params={"alpha": alpha, "beta": beta.tolist(), "sigma2": sigma2},
        )
        if hasattr(stan_out, "iloc"):
            lp_s = float(stan_out["lp__"].iloc[0])
            grad_s = stan_out.drop(columns=["lp__"], errors="ignore").iloc[0].to_numpy(dtype=float)
        else:
            lp_s = float(np.asarray(stan_out).squeeze())
            grad_s = None

        
        point = {
            "alpha": alpha,
            "beta": beta,
            "sigma2_log__": np.log(sigma2),  # unconstrained name PyMC expects
        }
        lp_p = float(logp_fn(point))
        g_p = dlogp_fn(point)
        grad_p = np.concatenate([np.atleast_1d(g).ravel() for g in g_p])

        stan_lps.append(lp_s)
        pm_lps.append(lp_p)
        if grad_s is not None and grad_s.size == grad_p.size:
            max_grad_err = max(max_grad_err, float(np.max(np.abs(grad_s - grad_p))))

    diff = np.asarray(stan_lps) - np.asarray(pm_lps)
    var_logp = float(np.var(diff))
    print(f"  Log-density diff variance: {var_logp:.6e}")
    print(f"  Max gradient error:        {max_grad_err:.6e}")
    ok = var_logp < 1e-6 and (max_grad_err < 1e-4 or max_grad_err == 0.0)
    print("  RESULT:", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    configure_windows_compiler()
    here = os.path.dirname(os.path.abspath(__file__))

    data_lr = load_json(os.path.join(here, "german_credit.json"))
    data_hlr = load_json(os.path.join(here, "german_credit_interactions.json"))

    ok1 = check_lr(os.path.join(here, "german_credit_lr.stan"), data_lr)
    ok2 = check_hlr(os.path.join(here, "german_credit_hlr.stan"), data_hlr)

    print("\n=======================================================")
    if ok1 and ok2:
        print("OVERALL STATUS: SUCCESS! Both models match.")
    else:
        print("OVERALL STATUS: FAILED. Inspect output above.")
    print("=======================================================")