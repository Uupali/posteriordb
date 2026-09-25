import os
import numpy as np
import pymc as pm
import cmdstanpy

# 1. Force the terminal to see your Windows C++ compiler paths (matching your setup)
compiler_paths = [
    r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
    r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin"
]
os.environ["PATH"] = os.pathsep.join(compiler_paths) + os.pathsep + os.environ.get("PATH", "")

# 2. Get paths to your files in the same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
stan_file_path = os.path.join(current_dir, "banana.stan")

print(f"Compiling Stan model from: {stan_file_path}")
stan_model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
print("Stan model compiled successfully!\n")

# 3. Setup global model dimensions & data parameters
D = 10
data_dict = {"D": D, "v": 100.0, "b": 0.1}

# 4. Define and compile the PyMC Model geometry
with pm.Model() as pymc_banana:
    sigma_x0 = np.sqrt(data_dict["v"])

    # First coordinate
    x = pm.Normal("x", mu=0, sigma=sigma_x0)

    # Twisting transformation for second coordinate
    mu_y = -data_dict["b"] * (x**2 - data_dict["v"])
    y = pm.Normal("y", mu=mu_y, sigma=1.0)

    # Remaining dimensions for D > 2
    if D > 2:
        y_rest = pm.Normal("y_rest", mu=0, sigma=1.0, shape=D - 2)

    # Compile the log-probability and gradient functions for PyMC
    pymc_logp_fn = pymc_banana.compile_logp()
    pymc_dlogp_fn = pymc_banana.compile_dlogp()

# 5. Generate fixed evaluation points (Section 2.1 Protocol Step 1)
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
    
    # --- STAN EVALUATION ---
    stan_point = {"y": phi.tolist()}
    stan_out = stan_model.log_prob(params=stan_point, data=data_dict)
    # Extract the log probability value from the "lp__" column
    stan_log_probs.append(stan_out["lp__"].iloc[0])

    # Extract the remaining columns, which represent the gradient vector
    grad_vector = stan_out.drop(columns=["lp__"]).iloc[0].values
    stan_grads.append(grad_vector)

    
    # --- PYMC EVALUATION ---
    # Map the flat array coordinates to PyMC's specific parameter names
    pymc_point = {
        "x": phi[0],
        "y": phi[1]
    }
    if D > 2:
        pymc_point["y_rest"] = phi[2:]
        
    pymc_log_probs.append(pymc_logp_fn(pymc_point))
    
    # Flatten the PyMC gradients (Order: [grad_x, grad_y, grad_y_rest])
    pm_grads = pymc_dlogp_fn(pymc_point)
    flat_pm_grads = np.concatenate([np.atleast_1d(g) for g in pm_grads])
    pymc_grads.append(flat_pm_grads)

# Convert all results to numpy arrays for verification math
stan_log_probs = np.array(stan_log_probs)
stan_grads = np.array(stan_grads)
pymc_log_probs = np.array(pymc_log_probs)
pymc_grads = np.array(pymc_grads)

# 6. The Section 2.1 Equivalence Protocol Verification Checks
logp_diff = stan_log_probs - pymc_log_probs
grad_diff = stan_grads - pymc_grads

print("\n=======================================================")
print("   HAARIO BANANA IMPLEMENTATION EQUIVALENCE REPORT     ")
print("=======================================================")
print(f"Log-density difference variance (should be ~0): {np.var(logp_diff):.6f}")
print(f"Gradient difference max error (should be ~0):   {np.max(np.abs(grad_diff)):.6f}")
print("=======================================================")
