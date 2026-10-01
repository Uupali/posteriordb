import os
import sys
import cmdstanpy
import numpy as np
import pymc as pm

# Import the PyMC model constructor directly from banana_improve.py
try:
  from banana_improve import get_banana_model
except ImportError:
  print(
      "CRITICAL ERROR: 'banana_improve.py' must be in the same directory as"
      " verification.py"
  )
  sys.exit(1)


def configure_windows_compiler():
  """Safely injects RTools paths into the Windows terminal environment."""
  compiler_paths = [
      r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
      r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin",
  ]
  if all(os.path.exists(p) for p in compiler_paths):
    os.environ["PATH"] = (
        ";".join(compiler_paths) + ";" + os.environ.get("PATH", "")
    )


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

  # 2. Setup global model configuration (Matching Haario et al., 1999)
  data_dict = {"D": 8, "b": 0.1}
  D = int(data_dict["D"])

  # 3. Instantiate the PyMC Model using get_banana_model
  print("Initializing PyMC model from banana_improve.py...")
  pymc_banana = get_banana_model(data_dict)

  with pymc_banana:
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

  print("Running cross-PPL log-density & gradient evaluation loop...")
  for i in range(N_points):
    phi = test_points[i]

    # --- 1. STAN EVALUATION ---
    stan_point = {"y": phi.tolist()}
    stan_out = stan_model.log_prob(params=stan_point, data=data_dict)

    # Extract lp__ scalar float
    stan_log_probs.append(stan_out["lp__"].iloc[0])

    # Extract raw gradient vector
    grad_vector = stan_out.drop(columns=["lp__"]).iloc[0].values
    stan_grads.append(grad_vector)

    # --- 2. PYMC EVALUATION ---
    pymc_point = {"x": phi[0], "y": phi[1]}
    if D > 2:
      pymc_point["y_rest"] = phi[2:]

    pymc_log_probs.append(pymc_logp_fn(pymc_point))

    # Evaluate PyMC gradients
    pm_grads = pymc_dlogp_fn(pymc_point)
    pymc_grads.append(np.asarray(pm_grads, dtype=float))

  # Convert outputs to numpy arrays
  stan_log_probs = np.array(stan_log_probs)
  stan_grads = np.array(stan_grads)
  pymc_log_probs = np.array(pymc_log_probs)
  pymc_grads = np.array(pymc_grads)

  # 5. Calculate Equivalence Metrics
  logp_diff = stan_log_probs - pymc_log_probs
  grad_diff = stan_grads - pymc_grads

  print("\n=======================================================")
  print("    HAARIO BANANA IMPLEMENTATION EQUIVALENCE REPORT     ")
  print("=======================================================")
  print(
      "Log-density difference variance (should be ~0):"
      f" {np.var(logp_diff):.6e}"
  )
  print(
      "Gradient difference max error (should be ~0):  "
      f" {np.max(np.abs(grad_diff)):.6e}"
  )
  print("=======================================================")

  if np.var(logp_diff) < 1e-6 and np.max(np.abs(grad_diff)) < 1e-6:
    print("STATUS: SUCCESS! Model implementations are mathematically equivalent.")
  else:
    print("STATUS: FAILED. Check coordinate order and parameter values.")
  print("=======================================================")


if __name__ == "__main__":
  run_verification()