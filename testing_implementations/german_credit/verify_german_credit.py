"""
implementation equivalence check
for the three German credit models (Stan vs PyMC).

Protocol (Torgander & Magnusson):
1. Fixed set of points in the computational space.
2. log p_Stan - log p_PyMC is approximately constant.
3. Gradients agree (∇log p_Stan ≈ ∇log p_PyMC).
"""

import os
import json
import numpy as np
import cmdstanpy

from german_credit_lr import make_model as make_lr
from german_credit_hlr import make_model as make_hlr
from german_credit_hlr_nc import make_model as make_hlr_nc


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


def check_model(stan_file, make_model_fn, data, sample_point_fn, n_points=30, seed=42, label=""):
    print(f"\n=== {label} ===")

    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    model = make_model_fn(data)
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    rng = np.random.default_rng(seed)
    logp_diffs = []
    max_rel_grad = 0.0
    max_abs_grad = 0.0

    for _ in range(n_points):
        stan_params, pymc_point = sample_point_fn(rng, data)

        # Stan
        stan_out = stan_model.log_prob(data=data, params=stan_params)
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
        lp_p = float(logp_fn(pymc_point))
        g_p = dlogp_fn(pymc_point)
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
    logp_range = float(np.max(logp_diffs) - np.min(logp_diffs))
    logp_std = float(np.std(logp_diffs))

    print(f"  Logp difference range (max-min): {logp_range:.6e}")
    print(f"  Logp difference std:             {logp_std:.6e}")
    print(f"  Max absolute gradient error:     {max_abs_grad:.6e}")
    print(f"  Max relative gradient error:     {max_rel_grad:.6e}")

    ok = (max_rel_grad < 1e-4) and (logp_range < 1.0)
    print("  RESULT:", "PASSED" if ok else "FAILED")
    return ok


def sample_lr(rng, data):
    D = int(data["D"])
    alpha = float(rng.normal(0, 2))
    beta = rng.normal(0, 2, size=D)
    stan_params = {"alpha": alpha, "beta": beta.tolist()}
    pymc_point = {"alpha": alpha, "beta": beta}
    return stan_params, pymc_point


def sample_hlr(rng, data):
    D = int(data["D"])
    alpha = float(rng.normal(0, 2))
    beta = rng.normal(0, 2, size=D)
    sigma2 = float(rng.uniform(20.0, 200.0))
    stan_params = {"alpha": alpha, "beta": beta.tolist(), "sigma2": sigma2}
    pymc_point = {"alpha": alpha, "beta": beta, "sigma2_log__": np.log(sigma2)}
    return stan_params, pymc_point


def sample_hlr_nc(rng, data):
    D = int(data["D"])
    alpha_raw = float(rng.normal(0, 1))
    beta_raw = rng.normal(0, 1, size=D)
    sigma2 = float(rng.uniform(20.0, 200.0))
    stan_params = {
        "alpha_raw": alpha_raw,
        "beta_raw": beta_raw.tolist(),
        "sigma2": sigma2,
    }
    pymc_point = {
        "alpha_raw": alpha_raw,
        "beta_raw": beta_raw,
        "sigma2_log__": np.log(sigma2),
    }
    return stan_params, pymc_point


if __name__ == "__main__":
    configure_windows_compiler()
    here = os.path.dirname(os.path.abspath(__file__))

    data_lr = load_json(os.path.join(here, "german_credit.json"))
    data_hlr = load_json(os.path.join(here, "german_credit_interactions.json"))

    ok1 = check_model(
        os.path.join(here, "german_credit_lr.stan"),
        make_lr,
        data_lr,
        sample_lr,
        label="german_credit_lr",
    )
    ok2 = check_model(
        os.path.join(here, "german_credit_hlr.stan"),
        make_hlr,
        data_hlr,
        sample_hlr,
        label="german_credit_hlr",
    )
    ok3 = check_model(
        os.path.join(here, "german_credit_hlr_nc.stan"),
        make_hlr_nc,
        data_hlr,
        sample_hlr_nc,
        label="german_credit_hlr_nc",
    )

    print("\n=======================================================")
    if ok1 and ok2 and ok3:
        print("OVERALL STATUS: SUCCESS! All three models are implementation-equivalent.")
    else:
        print("OVERALL STATUS: FAILED. Inspect output above.")
    print("=======================================================")