# Plasma spectroscopy models for posteriordb
 
Source: Langmore, Dikovsky, Geraedts, Norgaard, von Behren, "Hamiltonian Monte
Carlo in Inverse Problems; Ill-Conditioning and Multi-Modality",
arXiv:2103.07515 (2022).
 
The paper publishes neither its data nor several constants, so every dataset
here is simulated from the prior with a fixed seed. Values taken from the paper
and values chosen by us are marked in each generator script.
 
## Three tiers
 
| Tier | Stan model(s) | Data | Posterior | Expected difficulty |
|---|---|---|---|---|
| 1. Linear Gaussian toy (paper eqs. 1-2, Fig. 4) | `plasma_linear_noncentered.stan`, `plasma_linear_centered.stan` | `plasma_linear.json` | Exactly Gaussian, analytic reference in `plasma_linear_reference.json` | Easy. kappa (eq. 8) is 3.2 non-centred, 117 centred |
| 2. Shell model, lite | `plasma_shell.stan` | `plasma_shell_lite_sim.json` | Non-Gaussian, no analytic form | Moderate; not yet sampled |
| 3. Shell model, full | `plasma_shell.stan` | `plasma_shell_sim.json` | Non-Gaussian, very tight | Expensive; one test chain only |
 
Tier 1 is the best candidate for a posteriordb reference posterior. Tiers 2 and
3 can be submitted as model + data, with the sampling status stated.
 
## Files
 
| File | What it is | Goes into posteriordb? |
|---|---|---|
| `*.stan` | Models | Yes |
| `plasma_linear.json`, `plasma_shell_lite_sim.json`, `plasma_shell_sim.json` | Model data; keys match each Stan `data` block | Yes |
| `*_truth.json` | Parameter values used to simulate the data (an answer key) | No; for recovery checks and Stan/PyMC log-density comparison |
| `plasma_linear_reference.json` | Exact posterior mean and sd of the linear toy | No, but useful in the PR as proof of correctness |
| `generate_linear.py`, `generate_data.py` | Data generators (numpy; matplotlib optional) | Mention or link in the PR so data are reproducible |
| `run_linear.R`, `run_plasma.R` | Sampling and diagnostics with cmdstanr | No |
| `*.png` | Plots of the simulated spectra | No |
 
## Usage
 
```
python3 generate_linear.py          # tier 1 data + analytic reference
python3 generate_data.py lite       # tier 2 data
python3 generate_data.py full       # tier 3 data (default)
 
Rscript run_linear.R                # tier 1: both parameterizations vs analytic
Rscript run_plasma.R lite           # tier 2
Rscript run_plasma.R full           # tier 3 (slow)
```
 
## Choices not fixed by the paper (PyMC version must match)
 
- Prior covariance `exp(-(r - r')^2 / (2 tau^2)) + delta * I` (delta on the diagonal only).
- Linear toy forward matrix: each row a Gaussian-weighted local average (width 0.2),
  rows summing to 1. The paper's A is not given.
- Shell model: one centre shared by A, T, V with prior N(0, sigma_c^2 I);
  piecewise-linear radial interpolation, constant beyond the grid;
  vertical and horizontal chords; sigma and sigma_p fixed;
  `w = sqrt(T) / nu_0` as printed in the paper.
- The centred and non-centred linear models have the same posterior over `x`,
  but their log densities differ by the constant log|det L_pr|.
