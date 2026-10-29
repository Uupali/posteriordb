"""
plot_diagnostics_rosenbrock.py
==============================
Purpose
-------
Produce the visual diagnostics that help assess sampling quality and
compare the two PPL implementations.

Typical plots generated:
  - Trace plots
  - Rank plots (for R-hat / mixing)
  - Pair plots (especially useful for the parabolic ridge of Rosenbrock)
  - Energy plots (HMC energy diagnostics)
  - Autocorrelation plots

Run this AFTER the MCMC script
"""

import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_all_diagnostics(idata_stan, idata_pymc, instance_name: str, out_dir: str = "plots"):
    """
    Create a standard battery of diagnostic figures for one Rosenbrock instance.
    Compatible with the newest ArviZ (arviz_plots backend).
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, instance_name)

    # ------------------------------------------------------------------
    # 1. Trace plots
    # ------------------------------------------------------------------
    print(f"[{instance_name}] Trace plots ...")
    az.plot_trace(idata_stan, var_names=["x1", "x_rest"])
    plt.suptitle(f"{instance_name} – Stan traces", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{prefix}_stan_trace.png", dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_trace(idata_pymc, var_names=["x1", "x_rest"])
    plt.suptitle(f"{instance_name} – PyMC traces", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{prefix}_pymc_trace.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 2. Rank plots
    # ------------------------------------------------------------------
    print(f"[{instance_name}] Rank plots ...")
    az.plot_rank(idata_stan, var_names=["x1", "x_rest"])
    plt.suptitle(f"{instance_name} – Stan rank plots")
    plt.tight_layout()
    plt.savefig(f"{prefix}_stan_rank.png", dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_rank(idata_pymc, var_names=["x1", "x_rest"])
    plt.suptitle(f"{instance_name} – PyMC rank plots")
    plt.tight_layout()
    plt.savefig(f"{prefix}_pymc_rank.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 3. Pair plots
    # ------------------------------------------------------------------
    print(f"[{instance_name}] Pair plots ...")
    try:
        az.plot_pair(
            idata_stan,
            var_names=["x1", "x_rest"],
            marginals=True,
        )
    except Exception:
        # ultra-safe fallback
        az.plot_pair(idata_stan, var_names=["x1"])
    plt.suptitle(f"{instance_name} – Stan pair plot")
    plt.tight_layout()
    plt.savefig(f"{prefix}_stan_pair.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 4. Energy plots
    # ------------------------------------------------------------------
    print(f"[{instance_name}] Energy plots ...")
    az.plot_energy(idata_stan)
    plt.title(f"{instance_name} – Stan energy")
    plt.tight_layout()
    plt.savefig(f"{prefix}_stan_energy.png", dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_energy(idata_pymc)
    plt.title(f"{instance_name} – PyMC energy")
    plt.tight_layout()
    plt.savefig(f"{prefix}_pymc_energy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 5. Autocorrelation
    # ------------------------------------------------------------------
    print(f"[{instance_name}] Autocorrelation plots ...")
    az.plot_autocorr(idata_stan, var_names=["x1"], max_lag=50)
    plt.suptitle(f"{instance_name} – Stan autocorrelation (x1)")
    plt.tight_layout()
    plt.savefig(f"{prefix}_stan_autocorr.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[{instance_name}] All plots written to folder: {out_dir}/")

# ---------------------------------------------------------------------------
# Example usage (assumes you already have the InferenceData objects)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    instances_to_plot = [
        "rosenbrock_2d",
        "rosenbrock_n3_chain",
        "rosenbrock_n5_hybrid",
    ]

    for name in instances_to_plot:
        stan_path = f"results/{name}_stan.pkl"
        pymc_path = f"results/{name}_pymc.pkl"

        if not (os.path.exists(stan_path) and os.path.exists(pymc_path)):
            print(f"Skipping {name}: pickle files not found. "
                  "Run mcmc_diagnostics_rosenbrock.py first.")
            continue

        print(f"\nLoading {name} ...")
        with open(stan_path, "rb") as f:
            idata_stan = pickle.load(f)
        with open(pymc_path, "rb") as f:
            idata_pymc = pickle.load(f)

        plot_all_diagnostics(
            idata_stan=idata_stan,
            idata_pymc=idata_pymc,
            instance_name=name,
            out_dir="plots"
        )

    print("\nAll requested plots have been written to the 'plots/' folder.")
    