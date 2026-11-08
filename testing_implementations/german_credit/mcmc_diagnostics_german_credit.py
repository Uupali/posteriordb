"""
mcmc_diagnostics_german_credit.py
=================================
Short NUTS diagnostics only (no plots).
Stan (CmdStanPy) vs PyMC for:
  - german_credit_lr   (D=24)
  - german_credit_hlr  (D=300)
"""

import json
import os
import pickle

import arviz as az
import cmdstanpy
import pymc as pm

from german_credit_lr import make_model as make_lr
from german_credit_hlr import make_model as make_hlr

CHAINS = 4
DRAWS = 1000
TUNE = 1000
SEED = 42


def configure_windows_compiler():
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
        r"C:\rtools40\usr\bin",
        r"C:\rtools40\mingw64\bin",
        r"C:\Users\DELL\AppData\Local\Rtools40\usr\bin",
        r"C:\Users\DELL\AppData\Local\Rtools40\mingw64\bin",
    ]
    found = [p for p in compiler_paths if os.path.exists(p)]
    if found:
        os.environ["PATH"] = ";".join(found) + ";" + os.environ.get("PATH", "")
        print("Compiler paths added:", found)
    else:
        print("WARNING: no RTools paths found; Stan compile may fail.")


os.makedirs("results", exist_ok=True)


def load(name):
    with open(name) as f:
        return json.load(f)


def run_stan(stan_file, data):
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data=data,
        chains=CHAINS,
        iter_sampling=DRAWS,
        iter_warmup=TUNE,
        seed=SEED,
        show_progress=True,
    )
    return az.from_cmdstanpy(fit)


def run_pymc(make_fn, data):
    model = make_fn(data)
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=0.9,
            random_seed=SEED,
            progressbar=True,
        )
    return idata


def summarize(name, idata_stan, idata_pymc, var_names):
    print("\n" + "=" * 70)
    print(f"  {name}")
    print("=" * 70)
    for label, idata in [("Stan", idata_stan), ("PyMC", idata_pymc)]:
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
        print(s.head(8).to_string())


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

    # --- LR ---
    print("\n>>> Sampling german_credit_lr ...")
    data_lr = load("german_credit.json")
    idata_s_lr = run_stan("german_credit_lr.stan", data_lr)
    idata_p_lr = run_pymc(make_lr, data_lr)
    save_idata(idata_s_lr, "results/german_credit_lr_stan.nc")
    save_idata(idata_p_lr, "results/german_credit_lr_pymc.nc")
    summarize("german_credit_lr", idata_s_lr, idata_p_lr, ["alpha", "beta"])

    # --- HLR ---
    print("\n>>> Sampling german_credit_hlr ...")
    data_hlr = load("german_credit_interactions.json")
    idata_s_hlr = run_stan("german_credit_hlr.stan", data_hlr)
    idata_p_hlr = run_pymc(make_hlr, data_hlr)
    save_idata(idata_s_hlr, "results/german_credit_hlr_stan.nc")
    save_idata(idata_p_hlr, "results/german_credit_hlr_pymc.nc")
    summarize("german_credit_hlr", idata_s_hlr, idata_p_hlr, ["alpha", "sigma2"])

    print("\nDone. Results in results/")
    