"""
generate_reference_draws_rosenbrock.py
=====================================
Generate high-quality reference posterior draws for the Hybrid Rosenbrock
targets, following PosteriorDB criteria as closely as possible.
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import arviz as az
import cmdstanpy


def configure_windows_compiler():
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
    ]
    if all(os.path.exists(p) for p in compiler_paths):
        os.environ["PATH"] = (
            ";".join(compiler_paths) + ";" + os.environ.get("PATH", "")
        )


def lag1_autocorr(x):
    """Simple lag-1 autocorrelation for a 1-D array."""
    x = np.asarray(x).ravel()
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def generate_reference(
    instance_name: str,
    data_dict: dict,
    chains: int = 10,
    warmup: int = 10_000,
    sampling: int = 20_000,
    thin: int = 30,
    adapt_delta: float = 0.99,
    max_treedepth: int = 14,
    seed: int = 42,
    output_dir: str = "reference_draws",
):
    configure_windows_compiler()

    current_dir = Path(__file__).parent
    stan_file = current_dir / "rosenbrock.stan"

    n1 = int(data_dict["n1"])
    n2 = int(data_dict["n2"])
    dim = 1 + n2 * (n1 - 1)

    print("\n" + "=" * 75)
    print(f"  REFERENCE DRAWS  →  {instance_name}")
    print(f"  Dimension D = {dim}   (n1={n1}, n2={n2})")
    print("=" * 75)

    # ------------------------------------------------------------------
    # 1. Sampling
    # ------------------------------------------------------------------
    print("\n[1/4] Compiling Stan model and running long NUTS ...")
    print(f"      chains={chains}, warmup={warmup}, sampling={sampling}, "
          f"thin={thin}, adapt_delta={adapt_delta}, max_treedepth={max_treedepth}")

    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=data_dict,
        chains=chains,
        iter_warmup=warmup,
        iter_sampling=sampling,
        thin=thin,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        show_progress=True,
    )

    idata = az.from_cmdstanpy(posterior=fit)

    # ------------------------------------------------------------------
    # 2. Basic diagnostics
    # ------------------------------------------------------------------
    print("\n[2/4] Computing diagnostics ...")

    n_divergent = int(idata.sample_stats["diverging"].sum())

    summary = az.summary(idata, var_names=["x1", "x_rest"])
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())

    n_draws = idata.posterior.sizes["draw"] * idata.posterior.sizes["chain"]
    print(f"      Total draws after thinning : {n_draws}")

    # ------------------------------------------------------------------
    # 3. Lag-1 autocorrelation
    # ------------------------------------------------------------------
    print("\n[3/4] Checking lag-1 autocorrelation ...")

    x1_draws = idata.posterior["x1"].values.ravel()
    lag1_x1 = lag1_autocorr(x1_draws)
    max_lag1 = abs(lag1_x1)

    print(f"      lag-1 autocorr of x1       : {lag1_x1:.4f}")

    # ------------------------------------------------------------------
    # 4. Energy diagnostic (E-BFMI)
    # ------------------------------------------------------------------
    print("\n[4/4] Energy diagnostic (E-BFMI) ...")
    try:
        bfmi = az.bfmi(idata)
        min_bfmi = float(np.min(bfmi))
        print(f"      Minimum E-BFMI             : {min_bfmi:.4f}")
    except Exception as e:
        min_bfmi = np.nan
        print(f"      Could not compute E-BFMI: {e}")

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    criteria = {
        "n_draws_ok": n_draws >= 10_000,
        "no_divergences": n_divergent == 0,
        "rhat_ok": max_rhat < 1.01,
        "lag1_ok": max_lag1 < 0.05,
        "bfmi_ok": (min_bfmi >= 0.2) if not np.isnan(min_bfmi) else False,
    }

    passed = all(criteria.values())

    print("\n" + "-" * 75)
    print("REFERENCE POSTERIOR CHECK")
    print("-" * 75)
    print(f"  Draws after thinning          : {n_draws:>8}   {'OK' if criteria['n_draws_ok'] else 'FAIL'}")
    print(f"  Divergent transitions         : {n_divergent:>8}   {'OK' if criteria['no_divergences'] else 'FAIL'}")
    print(f"  Max R-hat                     : {max_rhat:>8.4f}   {'OK' if criteria['rhat_ok'] else 'FAIL'}")
    print(f"  |lag-1 autocorr| (x1)         : {max_lag1:>8.4f}   {'OK' if criteria['lag1_ok'] else 'FAIL'}")
    print(f"  Min E-BFMI                    : {min_bfmi:>8.4f}   {'OK' if criteria['bfmi_ok'] else 'FAIL'}")
    print("-" * 75)
    print(f"  OVERALL                       : {'PASSED' if passed else 'FAILED'}")
    print("-" * 75)

    # ------------------------------------------------------------------
    # Save only if the checks passed
    # ------------------------------------------------------------------
    if passed:
        out_path = Path(output_dir) / instance_name
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            idata.to_netcdf(out_path / f"{instance_name}_full.nc")
        except Exception:
            with open(out_path / f"{instance_name}_full.pkl", "wb") as f:
                pickle.dump(idata, f)

        info = {
            "name": instance_name,
            "dimension": dim,
            "data": data_dict,
            "inference": {
                "method": "stan_sampling",
                "method_arguments": {
                    "chains": chains,
                    "iter_warmup": warmup,
                    "iter_sampling": sampling,
                    "thin": thin,
                    "adapt_delta": adapt_delta,
                    "max_treedepth": max_treedepth,
                    "seed": seed,
                    "algorithm": "NUTS",
                },
            },
            "diagnostics": {
                "n_draws": n_draws,
                "n_divergent": n_divergent,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_ess_bulk,
                "min_ess_tail": min_ess_tail,
                "lag1_autocorr_x1": lag1_x1,
                "min_bfmi": min_bfmi,
            },
            "checks_passed": True,
            "criteria": criteria,
        }

        with open(out_path / f"{instance_name}_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\nPASSED → Results saved to: {out_path}/")
    else:
        print("\nFAILED → Nothing was saved. Adjust settings and re-run.")

    return idata, passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # All three instances are defined here.
    # Currently only the easiest one (rosenbrock_2d) is active.
    # Later, just uncomment the other two blocks when you are ready.
    # ------------------------------------------------------------------
    instances = {
        "rosenbrock_2d": {
            "n1": 2, "n2": 1, "mu": 1.0, "a": 0.05, "b": 5.0
        },

        # ---------- Uncomment the next two when you want to run them ----------
        #"rosenbrock_n3_chain": {
        #     "n1": 3, "n2": 1, "mu": 1.0, "a": 0.05, "b": 5.0
        #},
        # "rosenbrock_n5_hybrid": {
        #     "n1": 3, "n2": 2, "mu": 1.0, "a": 0.05, "b": 5.0
        #},
    }

    for name, data in instances.items():
        generate_reference(
            instance_name=name,
            data_dict=data,
            chains=10,
            warmup=10_000,
            sampling=20_000,
            thin=30,
            adapt_delta=0.99,
            max_treedepth=14,
            seed=42,
            output_dir="reference_draws",
        )
