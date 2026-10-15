import os
import sys
import cmdstanpy
import numpy as np
import pymc as pm
import pandas as pd

try:
    from rosenbrock import make_model
except ImportError:
    print("CRITICAL ERROR: 'rosenbrock.py' must be in same directory as verification.py")
    sys.exit(1)


def configure_windows_compiler():
    """Injects RTools paths into PATH environment variable."""
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
    ]
    if all(os.path.exists(p) for p in compiler_paths):
        os.environ["PATH"] = ";".join(compiler_paths) + ";" + os.environ.get("PATH", "")


def verify_live():
    """Runs live on-the-fly CmdStanPy vs PyMC verification across all 3 proposed benchmark instances."""
    configure_windows_compiler()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stan_file_path = os.path.join(current_dir, "rosenbrock.stan")

    print(f"Compiling Stan model from: {stan_file_path}")
    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)

    # All three proposed PosteriorDB benchmark instances
    benchmark_instances = {
        "rosenbrock_2d": {"n1": 2, "n2": 1, "mu": 1.0, "a": 0.05, "b": 5.0},
        "rosenbrock_n3_chain": {"n1": 3, "n2": 1, "mu": 1.0, "a": 0.05, "b": 5.0},
        "rosenbrock_n5_hybrid": {"n1": 3, "n2": 2, "mu": 1.0, "a": 0.05, "b": 5.0},
    }

    all_passed = True

    for name, data_dict in benchmark_instances.items():
        n1, n2 = int(data_dict["n1"]), int(data_dict["n2"])
        print(f"\nEvaluating instance: {name} (n1={n1}, n2={n2}, Dim={1 + n2*(n1-1)})...")

        pymc_model = make_model(data_dict)
        with pymc_model:
            pymc_logp_fn = pymc_model.compile_logp()
            pymc_dlogp_fn = pymc_model.compile_dlogp()

        np.random.seed(42)
        N_points = 100
        total_dim = 1 + n2 * (n1 - 1)
        test_points = np.random.uniform(-3, 3, size=(N_points, total_dim))

        stan_log_probs, stan_grads = [], []
        pymc_log_probs, pymc_grads = [], []

        for i in range(N_points):
            phi = test_points[i]
            x1_val = float(phi[0])
            x_rest_val = phi[1:].reshape((n2, n1 - 1), order="F")

            point_dict = {"x1": x1_val, "x_rest": x_rest_val.tolist()}
            stan_out = stan_model.log_prob(params=point_dict, data=data_dict)
            stan_log_probs.append(stan_out["lp__"].iloc[0])

            stan_grad_df = stan_out.drop(columns=["lp__"]).iloc[0]
            stan_grad_x1 = stan_grad_df.iloc[0]

            stan_grad_x_rest = np.zeros((n2, n1 - 1))
            for r in range(n2):
                for c in range(n1 - 1):
                    col_name = f"x_rest[{r+1},{c+1}]"
                    if col_name in stan_grad_df:
                        stan_grad_x_rest[r, c] = stan_grad_df[col_name]
                    else:
                        stan_grad_x_rest[r, c] = stan_grad_df.iloc[1 + r * (n1 - 1) + c]

            flat_stan_grad = np.concatenate([
                np.atleast_1d(stan_grad_x1),
                stan_grad_x_rest.ravel(order="F"),
            ])
            stan_grads.append(flat_stan_grad)

            pymc_point = {"x1": x1_val, "x_rest": x_rest_val}
            pymc_log_probs.append(pymc_logp_fn(pymc_point))
            pm_grads = pymc_dlogp_fn(pymc_point)
            flat_pm_grads = np.concatenate([np.atleast_1d(g).ravel(order="F") for g in pm_grads])
            pymc_grads.append(flat_pm_grads)

        logp_diff = np.array(stan_log_probs) - np.array(pymc_log_probs)
        grad_diff = np.array(stan_grads) - np.array(pymc_grads)

        var_logp = np.var(logp_diff)
        max_grad_err = np.max(np.abs(grad_diff))

        print("-------------------------------------------------------")
        print(f" Instance: {name}")
        print(f" Log-density diff variance: {var_logp:.6e}")
        print(f" Max gradient error:        {max_grad_err:.6e}")

        if var_logp < 1e-6 and max_grad_err < 1e-4:
            print(" RESULT: PASSED")
        else:
            print(" RESULT: FAILED")
            all_passed = False
        print("-------------------------------------------------------")

    print("\n=======================================================")
    if all_passed:
        print("OVERALL STATUS: SUCCESS! All 3 benchmark instances are mathematically equivalent.")
    else:
        print("OVERALL STATUS: FAILED. Check failed instances above.")
    print("=======================================================")


if __name__ == "__main__":
    verify_live()