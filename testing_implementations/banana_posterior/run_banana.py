import os
import cmdstanpy

# 1. Force the terminal to see the Windows C++ compiler paths
compiler_paths = [
    r"C:\Users\DELL\.cmdstan\RTools40\usr\bin",
    r"C:\Users\DELL\.cmdstan\RTools40\mingw64\bin"
]
os.environ["PATH"] = os.pathsep.join(compiler_paths) + os.pathsep + os.environ.get("PATH", "")

# 2. Get the path to your stan file in the same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
stan_file_path = os.path.join(current_dir, "banana.stan")

print(f"Compiling Stan model from: {stan_file_path}")

# 3. Compile the model
model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
print("Model compiled successfully!")

# 4. Define the input data dictionary required by banana.stan
data_dict = {"D": 2, "v": 100.0, "b": 0.1}

# 5. Run the MCMC sampler
print("Running MCMC sampling chains...")
#fit = model.sample(data=data_dict, chains=4, iter_sampling=1000)
fit = model.sample( data=data_dict,
                    chains=4,
                    iter_warmup=2000, 
                    iter_sampling=1000,
                    adapt_delta=0.95,
                    seed=42 )

# 6. Print out the results summary table
print("\n--- Stan Sampling Summary ---")
print(fit.summary())
