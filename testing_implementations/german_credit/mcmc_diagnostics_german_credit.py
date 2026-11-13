"""
mcmc_diagnostics_german_credit.py
=================================
Unified short/full diagnostics:
  - Stan (CmdStanPy) vs PyMC
  - Stricter NUTS settings (adapt_delta, max_treedepth)
  - Optional PosteriorDB-style checks on Stan draws
  - Models: lr, hlr_nc  (centered hlr optional)
"""

import json
import os
import pickle
import time

import arviz as az
import cmdstanpy
import numpy as np
import pymc as pm

from german_credit_lr import make_model as make_lr
from german_credit_hlr_nc import make_model as make_hlr_nc

# Optional centered (slow / hard) — set True only if you want it
RUN_CENTERED_HLR = False
if RUN_CENTERED_HLR:
    from german_credit_hlr import make_model as make_hlr

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
QUICK = True          # True: short trial; False: closer to reference length
SEED = 42
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 15
TARGET_ACCEPT_PYMC = 0.95

if QUICK:
    CHAINS, WARMUP, DRAWS, THIN = 4, 1000, 1000, 1
else:
    CHAINS, WARMUP, DRAWS, THIN = 10, 10_000, 10_000, 10


def configure_windows_compiler():
    paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
        r"C:\rtools40\usr\bin",
        r"C:\rtools40\mingw64\bin",
    ]
    found = [p for p in paths if os.path.exists(p)]
    if found:
        os.environ["PATH"] = ";".join(found) + ";" + os.environ.get("PATH", "")
        print("Compiler paths added:", found)


os.makedirs("results", exist_ok=True)


def load(name):
    with open(name) as f:
        return json.load(f)


def run_stan(stan_file, data):
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data=data,
        chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=DRAWS,
        thin=THIN,
        seed=SEED,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        show_progress=True,
    )
    return fit, az.from_cmdstanpy(fit)


def run_pymc(make_fn, data, initvals=None, init="jitter+adapt_diag"):
    model = make_fn(data)
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=WARMUP,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT_PYMC,
            random_seed=SEED,
            progressbar=True,
            initvals=initvals,
            init=init
        )
    return idata

def hlr_nc_initvals(D, chains, seed=SEED, scale=0.1):
    """Per-chain starting points: sigma2 = 1, small random raw coefficients."""
    rng = np.random.default_rng(seed)
    return [
        {
            "sigma2": 1.0,
            "alpha_raw": rng.normal(0.0, scale),
            "beta_raw": rng.normal(0.0, scale, size=D),
        }
        for _ in range(chains)
    ]

def summarize_pair(name, idata_s, idata_p, var_names):
    print("\n" + "=" * 70)
    print(f"  {name}   (QUICK={QUICK}, chains={CHAINS}, draws={DRAWS}, thin={THIN})")
    print("=" * 70)
    for label, idata in [("Stan", idata_s), ("PyMC", idata_p)]:
        s = az.summary(idata, var_names=var_names, kind="diagnostics")
        div = (
            int(idata.sample_stats.diverging.sum())
            if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats
            else -1
        )
        print(f"\n[{label}]")
        print(f"  divergences : {div}")
        print(f"  max R-hat   : {s['r_hat'].max():.4f}")
        print(f"  min ESS bulk: {s['ess_bulk'].min():.0f}")
        print(f"  min ESS tail: {s['ess_tail'].min():.0f}")
        print(s.head(6).to_string())


def stan_reference_style_checks(fit, idata):
    """Lightweight PosteriorDB-style gates on Stan only."""
    n_draws = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    n_div = int(fit.divergences.sum()) if hasattr(fit, "divergences") else -1
    summary = az.summary(idata, kind="diagnostics")
    max_rhat = float(summary["r_hat"].max())

    try:
        bfmi = az.bfmi(idata)
        min_bfmi = float(np.min(bfmi))
    except Exception:
        min_bfmi = float("nan")

    print("\n--- Stan PosteriorDB-style snapshot ---")
    print(f"  draws        : {n_draws}   {'OK' if n_draws >= 10000 else 'FAIL (ok if QUICK)'}")
    print(f"  divergences  : {n_div}   {'OK' if n_div == 0 else 'FAIL'}")
    print(f"  max R-hat    : {max_rhat:.4f}   {'OK' if max_rhat < 1.01 else 'FAIL'}")
    print(f"  min E-BFMI   : {min_bfmi:.4f}   {'OK' if min_bfmi >= 0.2 else 'FAIL'}")
    print("  (lag-1 full check: use check_german_credit.py for all parameters)")


def save_idata(idata, path_nc):
    try:
        idata.to_netcdf(path_nc)
        print(f"  saved {path_nc}")
    except Exception as e:
        path_pkl = path_nc.replace(".nc", ".pkl")
        with open(path_pkl, "wb") as f:
            pickle.dump(idata, f)
        print(f"  netCDF failed ({e}); saved {path_pkl}")


if __name__ == "__main__":
    configure_windows_compiler()
    t0 = time.time()

    # --- LR ---
    print("\n>>> german_credit_lr")
    data_lr = load("german_credit.json")
    fit_s_lr, idata_s_lr = run_stan("german_credit_lr.stan", data_lr)
    idata_p_lr = run_pymc(make_lr, data_lr)
    save_idata(idata_s_lr, "results/german_credit_lr_stan.nc")
    save_idata(idata_p_lr, "results/german_credit_lr_pymc.nc")
    summarize_pair("german_credit_lr", idata_s_lr, idata_p_lr, ["alpha", "beta"])
    stan_reference_style_checks(fit_s_lr, idata_s_lr)

    # --- HLR non-centered ---
    print("\n>>> german_credit_hlr_nc")
    data_hlr = load("german_credit_interactions.json")
    fit_s_nc, idata_s_nc = run_stan("german_credit_hlr_nc.stan", data_hlr)
    idata_p_nc = run_pymc(
        make_hlr_nc,
        data_hlr,
        initvals=hlr_nc_initvals(data_hlr["D"], CHAINS),
        init="adapt_diag",
    )
    save_idata(idata_s_nc, "results/german_credit_hlr_nc_stan.nc")
    save_idata(idata_p_nc, "results/german_credit_hlr_nc_pymc.nc")
    summarize_pair("german_credit_hlr_nc", idata_s_nc, idata_p_nc, ["alpha", "sigma2"])
    stan_reference_style_checks(fit_s_nc, idata_s_nc)

    # --- Optional centered ---
    if RUN_CENTERED_HLR:
        print("\n>>> german_credit_hlr (centered)")
        fit_s_c, idata_s_c = run_stan("german_credit_hlr.stan", data_hlr)
        idata_p_c = run_pymc(make_hlr, data_hlr)
        summarize_pair("german_credit_hlr", idata_s_c, idata_p_c, ["alpha", "sigma2"])

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.  QUICK={QUICK}")