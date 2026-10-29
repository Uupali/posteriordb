"""
mcmc_diagnostics_rosenbrock.py
==============================
Purpose
-------
Run NUTS sampling on the Hybrid Rosenbrock target with both CmdStanPy and PyMC,
convert both fits to ArviZ InferenceData, and produce a side-by-side diagnostic
report (divergences, R-hat, ESS, etc.).

This script fulfils the project requirements:
  - Construct a reproducible pipeline for posterior sampling
  - Perform diagnostic checks and compare convergence quality between PPLs

It is intentionally kept at a moderate length (4 chains X 1000 warm-up + 1000
sampling) so that it finishes quickly during development.  
"""

import os
import sys
import numpy as np
import pandas as pd
import arviz as az
import cmdstanpy
import pymc as pm
import pickle

# ---------------------------------------------------------------------------
# Import the PyMC model constructor (must live in the same folder)
# ---------------------------------------------------------------------------
try:
    from rosenbrock import make_model
except ImportError:
    print("CRITICAL ERROR: 'rosenbrock.py' must be in the same directory.")
    sys.exit(1)


def configure_windows_compiler():
    """
    On Windows, CmdStan needs the RTools toolchain on PATH.
    This helper injects the usual locations if they exist.
    On Linux / macOS the function is a no-op.
    """
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
    ]
    if all(os.path.exists(p) for p in compiler_paths):
        os.environ["PATH"] = (
            ";".join(compiler_paths) + ";" + os.environ.get("PATH", "")
        )


def run_mcmc_comparison(
    instance_name: str,
    data_dict: dict,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    seed: int = 42,
    adapt_delta: float = 0.95,
):
    """
    Sample the same Rosenbrock instance with Stan and with PyMC, then print
    a compact diagnostic comparison.

    Parameters
    ----------
    instance_name : str
        Human-readable label (e.g. "rosenbrock_n5_hybrid")
    data_dict : dict
        Must contain keys n1, n2, a, b, mu
    chains, warmup, draws, seed, adapt_delta
        Standard NUTS settings (kept moderate for development speed)
    """
    configure_windows_compiler()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stan_file = os.path.join(current_dir, "rosenbrock.stan")

    n1 = int(data_dict["n1"])
    n2 = int(data_dict["n2"])
    total_dim = 1 + n2 * (n1 - 1)

    print("=" * 70)
    print(f"  MCMC comparison for: {instance_name}")
    print(f"  Dimension D = {total_dim}   (n1={n1}, n2={n2})")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Stan / CmdStanPy sampling
    # ------------------------------------------------------------------
    print("\n[1/3] Compiling & sampling with CmdStanPy ...")
    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    stan_fit = stan_model.sample(
        data=data_dict,
        chains=chains,
        iter_warmup=warmup,
        iter_sampling=draws,
        adapt_delta=adapt_delta,
        seed=seed,
        show_progress=True,
        # Optional: save the CSV files if you want to inspect them later
        # output_dir="stan_output"
    )

    # Convert to ArviZ (this preserves the original parameter names x1, x_rest)
    idata_stan = az.from_cmdstanpy(posterior=stan_fit)

    # ------------------------------------------------------------------
    # 2. PyMC sampling
    # ------------------------------------------------------------------
    print("\n[2/3] Sampling with PyMC ...")
    pymc_model = make_model(data_dict)

    with pymc_model:
        idata_pymc = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            target_accept=adapt_delta,
            random_seed=seed,
            progressbar=True,
            # cores=1 keeps things sequential and avoids Windows multiprocessing issues
            cores=1,
        )

    # ------------------------------------------------------------------
    # 3. Diagnostic summary
    # ------------------------------------------------------------------
    print("\n[3/3] Computing diagnostics ...")

    # Divergences (total across all chains after warm-up)
    stan_div = int(idata_stan.sample_stats["diverging"].sum())
    pymc_div = int(idata_pymc.sample_stats["diverging"].sum())

    # R-hat / ESS summaries (ArviZ handles the nested matrix parameter nicely)
    summary_stan = az.summary(idata_stan, var_names=["x1", "x_rest"])
    summary_pymc = az.summary(idata_pymc, var_names=["x1", "x_rest"])

    print("\n" + "-" * 70)
    print("DIVERGENCES")
    print(f"  Stan : {stan_div}")
    print(f"  PyMC : {pymc_div}")
    print("-" * 70)

    print("\nSTAN summary (first few rows)")
    print(summary_stan.head(8).to_string())

    print("\nPYMC summary (first few rows)")
    print(summary_pymc.head(8).to_string())

    # Quick pass / fail style check useful during development
    max_rhat_stan = summary_stan["r_hat"].max()
    max_rhat_pymc = summary_pymc["r_hat"].max()
    min_ess_stan = summary_stan["ess_bulk"].min()
    min_ess_pymc = summary_pymc["ess_bulk"].min()

    print("\nQUICK HEALTH CHECK")
    print(f"  Max R-hat  Stan / PyMC : {max_rhat_stan:.4f} / {max_rhat_pymc:.4f}")
    print(f"  Min ESS    Stan / PyMC : {min_ess_stan:.0f} / {min_ess_pymc:.0f}")

    if stan_div == 0 and pymc_div == 0 and max_rhat_stan < 1.05 and max_rhat_pymc < 1.05:
        print("  STATUS: looks healthy for a short development run")
    else:
        print("  STATUS: investigate further (possible funnel / ridge problems)")

    # Return the two InferenceData objects so the plotting script can reuse them
    return idata_stan, idata_pymc


# ---------------------------------------------------------------------------
# Main entry point – run all three benchmark instances
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    instances = {
        "rosenbrock_2d": {"n1": 2, "n2": 1, "mu": 1.0, "a": 0.05, "b": 5.0},
        "rosenbrock_n3_chain": {"n1": 3, "n2": 1, "mu": 1.0, "a": 0.05, "b": 5.0},
        "rosenbrock_n5_hybrid": {"n1": 3, "n2": 2, "mu": 1.0, "a": 0.05, "b": 5.0},
    }

    results = {}

    for name, data in instances.items():
        idata_s, idata_p = run_mcmc_comparison(
            instance_name=name,
            data_dict=data,
            chains=4,
            warmup=1000,
            draws=1000,
            seed=42,
            adapt_delta=0.95,
        )
        results[name] = {"stan": idata_s, "pymc": idata_p}

        # ---- NEW: persist the results ----
        os.makedirs("results", exist_ok=True)

        with open(f"results/{name}_stan.pkl", "wb") as f:
            pickle.dump(idata_s, f)

        with open(f"results/{name}_pymc.pkl", "wb") as f:
            pickle.dump(idata_p, f)

        print(f"  Saved → results/{name}_stan.pkl  and  results/{name}_pymc.pkl")

