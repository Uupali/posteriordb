import os
import sys
import numpy as np
import pymc as pm
import cmdstanpy

def configure_windows_compiler():
    """Safely injects RTools paths into the Windows terminal environment."""
    compiler_paths = [
        r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
        r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin"
    ]
    if all(os.path.exists(p) for p in compiler_paths):
        os.environ["PATH"] = ";".join(compiler_paths) + ";" + os.environ.get("PATH", "")


def run_verification():
    configure_windows_compiler()

    # 1. Resolve paths and compile Stan model safely
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stan_file_path = os.path.join(current_dir, "banana.stan")

    if not os.path.exists(stan_file_path):
        print(f"CRITICAL ERROR: 'banana.stan' not found in {current_dir}")
        sys.exit(1)

    print(f"Compiling Stan model from: {stan_file_path}")
    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
    print("Stan model compiled successfully!\n")

    # 2. Setup global model dimensions (Stan data contains *only* D)
    D = 8  # e.g., 2, 4, 8 from Haario et al., 1999
    stan_data = {"D": D}
    v_val = 100.0
    b_val = 0.1

    # 3. Define and compile the PyMC Model geometry
    print("Initializing PyMC model mapping configurations...")
    with pm.Model() as pymc_banana:
        sigma_x0 = np.sqrt(v_val)

        # First coordinate
        x = pm.Normal("x", mu=0, sigma=sigma_x0)

        # Twisting transformation for second coordinate
        mu_y = -b_val * (x**2 - v_val)
        y = pm.Normal("y", mu=mu_y, sigma=1.0)

        # Remaining dimensions for D > 2
        if D > 2:
            y_rest = pm.Normal("y_rest", mu=0, sigma=1.0, shape=D - 2)

        # Compile log-probability and gradient functions
        pymc_logp_fn = pymc_banana.compile_logp()
        pymc_dlogp_fn = pymc_banana.compile_dlogp()

    # 4. Generate fixed evaluation points for cross-PPL comparison
    np.random.seed(42)
    N_points = 100
    test_points = np.random.uniform(-5, 5, size=(N_points, D))

    stan_log_probs = []
    stan_grads = []
    pymc_log_probs = []
    pymc_grads = []

    print("Running cross-PPL evaluation loop...")
    for i in range(N_points):
        phi = test_points[i]
        
        # --- 1. STAN EVALUATION ---
        stan_point = {"y": phi.tolist()}
        stan_out = stan_model.log_prob(params=stan_point, data=stan_data)
        
        # Extract true scalar float from the "lp__" column row index 0
        stan_log_probs.append(stan_out["lp__"].iloc[0])

        # Extract everything EXCEPT "lp__" to isolate the raw gradient vector array
        grad_vector = stan_out.drop(columns=["lp__"]).iloc[0].values
        stan_grads.append(grad_vector)

        # --- 2. PYMC EVALUATION ---
        pymc_point = {
            "x": phi[0],  # Element 0 matches y in Stan
            "y": phi[1]   # Element 1 matches y in Stan
        }
        if D > 2:
            pymc_point["y_rest"] = phi[2:]  # Elements 2 to D match y[3:D] in Stan
            
        pymc_log_probs.append(pymc_logp_fn(pymc_point))
        
        # Evaluate PyMC's gradients (returns a flat, 10-element array directly)
        pm_grads = pymc_dlogp_fn(pymc_point)
        
        # PyMC naturally outputs them ordered by execution context matching Stan's vector layout perfectly.
        # We ensure it is cast tightly as a plain 1D float array.
        pymc_grads.append(np.asarray(pm_grads, dtype=float))

    # Convert collections to arrays for linear algebra error checks
    stan_log_probs = np.array(stan_log_probs)
    stan_grads = np.array(stan_grads)
    pymc_log_probs = np.array(pymc_log_probs)
    pymc_grads = np.array(pymc_grads)

    # 5. Equivalence Protocol Verification Calculations
    logp_diff = stan_log_probs - pymc_log_probs
    grad_diff = stan_grads - pymc_grads

    print("\n=======================================================")
    print("   HAARIO BANANA IMPLEMENTATION EQUIVALENCE REPORT     ")
    print("=======================================================")
    print(f"Log-density difference variance (should be ~0): {np.var(logp_diff):.6e}")
    print(f"Gradient difference max error (should be ~0):   {np.max(np.abs(grad_diff)):.6e}")
    print("=======================================================")
    
    # Tolerances are set keeping minor PPL float formatting variations in mind
    if np.var(logp_diff) < 1e-6 and np.max(np.abs(grad_diff)) < 1e-6:
        print("STATUS: SUCCESS! Model implementations are mathematically equivalent.")
    else:
        print("STATUS: FAILED. Check alignment formatting profiles.")
    print("=======================================================")


# Protects the Windows execution process pool
if __name__ == "__main__":
    run_verification()
