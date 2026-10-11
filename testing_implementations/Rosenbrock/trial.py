from multiprocessing import freeze_support
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

def make_model(data: dict) -> pm.Model:
    """PyMC Hybrid Rosenbrock Posterior Model.

    Expects data dict with keys:
    - 'n1': block size (number of chained variables per block)
    - 'n2': number of parallel blocks
    - 'mu': root variable location/mean
    - 'a': scale parameter for root variable x
    - 'b': curvature parameter for chained coordinates
    """
    n1 = int(data.get("n1", 3))
    n2 = int(data.get("n2", 2))
    mu = float(data.get("mu", 1.0))
    a = float(data.get("a", 0.05))
    b = float(data.get("b", 5.0))

    # Total dimension: n = (n1 - 1) * n2 + 1
    n = (n1 - 1) * n2 + 1

    def hybrid_rosenbrock_logp(x):
        # x is a vector of length n
        # 1. Root variable x log-density: -a * (x - mu)^2
        logp_root = -a * pt.sqr(x - mu)

        # 2. Parallel block chains: x_{j,i} | x_{j,i-1} ~ N(x_{j,i-1}^2, sqrt(1 / (2b)))
        logp_blocks = 0.0
        idx = 1
        for j in range(n2):
            prev = x
            for i in range(1, n1):
                logp_blocks += -b * pt.sqr(x[idx] - pt.sqr(prev))
                prev = x[idx]
                idx += 1

        return logp_root + logp_blocks

    with pm.Model() as model:
        # Single vector parameter 'x' of shape (n,) matching PosteriorDB schema
        pm.CustomDist(
            "x", logp=hybrid_rosenbrock_logp, shape=(n,), initval=np.ones(n)
        )

    return model

if __name__ == "__main__":
    # Required for Windows/macOS multiprocessing safety
    freeze_support()
    print("--- Sampling Hybrid Rosenbrock Posterior via PyMC ---")

    # Benchmark configuration (n1=3, n2=2 gives n=5 dimensions)
    data_dict = {"n1": 3, "n2": 2, "mu": 1.0, "a": 0.05, "b": 5.0}

    # Initialize the model
    rosenbrock_model = make_model(data_dict)

    # Sample from the target posterior
    with rosenbrock_model:
        idata = pm.sample(
            draws=1000,
            tune=2000,
            chains=4,
            cores=2,  # Set to more than 1 if your OS environment allows stable multiprocessing
            target_accept=0.95,  # Higher target_accept helps navigate the narrow parabolic valley
            random_seed=42,
            progressbar=True,
        )

    print("\nSampling finished!")

    # Check for geometry issues (divergences)
    divergences = idata.sample_stats["diverging"].sum().item()
    print(f"Number of Divergent Transitions: {divergences}")

    print("\nSummary:")
    print(az.summary(idata))
