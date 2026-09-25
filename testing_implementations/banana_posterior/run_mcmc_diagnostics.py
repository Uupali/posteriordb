import os
import pymc as pm
import cmdstanpy
import arviz as az
import numpy as np
import pandas as pd

# 1. Setup Windows C++ compiler paths for Stan
compiler_paths = [
    r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
    r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin"
]
os.environ["PATH"] = os.pathsep.join(compiler_paths) + os.pathsep + os.environ.get("PATH", "")

current_dir = os.path.dirname(os.path.abspath(__file__))
stan_file_path = os.path.join(current_dir, "banana.stan")

# 2. Configuration Parameters
D = 10
data_dict = {"D": D, "v": 100.0, "b": 0.1}
SEED = 42
CHAINS = 4
WARMUP = 1000
DRAWS = 1000

print("=======================================================")
print(" 1. SAMPLING VIA CMDSTANPY (NUTS)                      ")
print("=======================================================")
stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
stan_fit = stan_model.sample(
    data=data_dict,
    chains=CHAINS,
    iter_warmup=WARMUP,
    iter_sampling=DRAWS,
    adapt_delta=0.95,
    seed=SEED,
    show_progress=True
)

# Convert CmdStanMCMC to ArviZ InferenceData format
# Note: Stan saves variables grouped as an array y[1], y[2]... 
idata_stan = az.from_cmdstanpy(posterior=stan_fit)

print("\n=======================================================")
print(" 2. SAMPLING VIA PYMC (NUTS)                           ")
print("=======================================================")
with pm.Model() as pymc_banana:
    sigma_x0 = np.sqrt(data_dict["v"])
    
    y1 = pm.Normal("y1", mu=0, sigma=sigma_x0)
    y2 = pm.Normal("y2", mu=-data_dict["b"] * (y1**2 - data_dict["v"]), sigma=1.0)
    
    if D > 2:
        y_rest = pm.Normal("y_rest", mu=0, sigma=1.0, shape=D - 2)
        
    idata_pymc = pm.sample(
        draws=DRAWS,
        tune=WARMUP,
        chains=CHAINS,
        target_accept=0.95,  # Improve convergence for banana-shaped posterior
        cores=1,  # Keep it sequential or adjust based on CPU cores
        random_seed=SEED,
        progressbar=True
    )

print("\n=======================================================")
print(" 3. DIAGNOSTICS & CONVERGENCE COMPARISON              ")
print("=======================================================")

# Extract Divergences
stan_divergences = idata_stan.sample_stats["diverging"].sum().item()
pymc_divergences = idata_pymc.sample_stats["diverging"].sum().item()

# Compute summaries via ArviZ
summary_stan = az.summary(idata_stan, var_names=["y"])
summary_pymc = az.summary(idata_pymc, var_names=["y1", "y2", "y_rest"])

# Print side-by-side diagnostic snapshot for key warped coordinates (Index 0 and 1)
print(f"Total Divergent Transitions -> Stan: {stan_divergences} | PyMC: {pymc_divergences}")
print("\n--- Stan MCMC Diagnostics Summary (First two coordinates) ---")
print(summary_stan.iloc[[0, 1]][["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])

print("\n--- PyMC MCMC Diagnostics Summary (First two coordinates) ---")
# Accessing y1 and y2 explicitly
print(summary_pymc.iloc[[0, 1]][["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])
