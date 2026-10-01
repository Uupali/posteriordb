from multiprocessing import freeze_support
import arviz as az
import numpy as np
import pymc as pm

def get_banana_model(data_dict: dict) -> pm.Model:
  """PyMC Banana Posterior.

  Expects data_dict with key: 'D'
  """
  D = int(data_dict["D"])
  b = float(data_dict["b"])
  v = 100

  with pm.Model() as model:
    sigma_x0 = np.sqrt(v)

    # First coordinate
    x = pm.Normal("x", mu=0, sigma=sigma_x0)

    # Twisting transformation for second coordinate
    mu_y = -b * (x**2 - v)
    y = pm.Normal("y", mu=mu_y, sigma=1.0)

    # Remaining dimensions for D > 2
    if D > 2:
      pm.Normal("y_rest", mu=0, sigma=1.0, shape=D - 2)

  return model

if __name__ == "__main__":
  freeze_support()
  print("--- Sampling Haario Banana Posterior via PyMC ---")

  data_dict = {"D": 8} #2, 4, 8 Haario et al. (1999)
  banana_model = get_banana_model(data_dict)

  with banana_model:
    idata = pm.sample(
        draws=1000,
        tune=2000,
        chains=4,
        cores=1,
        target_accept=0.99,
        random_seed=42,
        progressbar=True,
    )

  print("\nSampling finished!")
  divergences = idata.sample_stats["diverging"].sum().item()
  print(f"Number of Divergent Transitions: {divergences}")
  print("\nSummary:")
  print(az.summary(idata))