'''import pymc as pm
import arviz as az
from multiprocessing import freeze_support

freeze_support()

if __name__ == '__main__':
    print("--- Improved medium sampling: Haario banana ---")

    b = 0.1

    with pm.Model() as banana_model:
        x = pm.Normal("x", mu=0, sigma=10)
        y = pm.Normal("y", mu=b * (x**2 - 100), sigma=1)

        idata = pm.sample(
            draws=1000,
            tune=2000,               # more tuning
            chains=4,
            cores=1,
            target_accept=0.95,      # key improvement
            random_seed=42,
            progressbar=True
        )

    print("\nSampling finished!")
    divergences = idata.sample_stats["diverging"].sum().item()
    print(f"Number of Divergent Transitions: {divergences}")
    print("\nSummary:")
    print(az.summary(idata, var_names=["x", "y"]))
    '''

from multiprocessing import freeze_support
import arviz as az
import numpy as np
import pymc as pm

def get_banana_model(data_dict: dict) -> pm.Model:
  """Generalized PyMC Banana Posterior matching PosteriorDB standards.

  Expects data_dict with keys: 'D', 'v', 'b'
  """
  D = data_dict.get("D", 2)
  v = data_dict.get("v", 100.0)
  b = data_dict.get("b", 0.1)

  with pm.Model() as model:
    sigma_x0 = np.sqrt(v)

    # First coordinate
    x = pm.Normal("x", mu=0, sigma=sigma_x0)

    # Twisting transformation for second coordinate (aligned sign)
    mu_y = -b * (x**2 - v)
    y = pm.Normal("y", mu=mu_y, sigma=1.0)

    # Remaining dimensions for D > 2
    if D > 2:
      pm.Normal("y_rest", mu=0, sigma=1.0, shape=D - 2)

  return model

if __name__ == "__main__":
  freeze_support()
  print("--- Sampling Haario Banana Posterior via PyMC ---")

  data_dict = {"D": 10, "v": 100.0, "b": 0.1}
  banana_model = get_banana_model(data_dict)

  with banana_model:
    idata = pm.sample(
        draws=1000,
        tune=2000,
        chains=4,
        cores=1,
        target_accept=0.95,
        random_seed=42,
        progressbar=True,
    )

  print("\nSampling finished!")
  divergences = idata.sample_stats["diverging"].sum().item()
  print(f"Number of Divergent Transitions: {divergences}")
  print("\nSummary:")
  print(az.summary(idata))