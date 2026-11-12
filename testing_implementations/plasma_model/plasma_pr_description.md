# [Draft] Add plasma spectroscopy inverse-problem models (Langmore et al. 2022)
 
> **Status:** draft for discussion. This PR adds **data, models and posteriors only**. No reference posterior draws are included. Several modelling choices are not fixed by the source paper; they are listed under [Open decisions](#open-decisions) and we would welcome feedback on those. 
 
## Summary
 
This PR adds a family of Bayesian inverse problems from fusion-plasma spectroscopy, based on
 
Langmore, I., Dikovsky, M., Geraedts, S., Norgaard, P., von Behren, R. (2022). *Hamiltonian Monte Carlo in Inverse Problems; Ill-Conditioning and Multi-Modality.* arXiv:2103.07515.
 
The paper uses this problem to study two difficulties HMC meets in inverse problems: **ill-conditioning** and **multi-modality**. It is a physics forward model with a low-rank integration operator, signal-dependent noise and a non-linear geometric parameter. The paper publishes neither its data nor several constants, so **all datasets are simulated from the prior** with fixed seeds, and the generator scripts are included.
 
We propose three tiers, from an exactly Gaussian toy to the full non-linear model:
 
| Posterior | Model | Data | Dim. | Character |
|---|---|---|---|---|
| `plasma_linear-plasma_linear_noncentered` | `plasma_linear_noncentered` | `plasma_linear` | 40 | Exactly Gaussian, well conditioned |
| `plasma_linear-plasma_linear_centered` | `plasma_linear_centered` | `plasma_linear` | 40 | Same posterior, badly conditioned |
| `plasma_shell_lite-plasma_shell` | `plasma_shell` | `plasma_shell_lite` | 26 | Non-linear, non-Gaussian, small |
| `plasma_shell_full-plasma_shell` | `plasma_shell` | `plasma_shell_full` | 38 | Non-linear, very concentrated, expensive |
 
## The model
 
### Intuition
 
A hot plasma (a cloud of ions) glows, but it cannot be observed directly. Detectors look through it along straight lines (*chords*) and record a spectrum: intensity as a function of frequency. Three spatial fields determine the light each point emits:
 
- **amplitude $\mathcal A$**: how brightly it emits, which sets the height of the spectral peak;
- **temperature $\mathcal T$**: thermal motion Doppler-broadens the emission line, which sets the width;
- **velocity $\mathcal V$**: bulk motion Doppler-shifts the line, which sets the position.
Each detector measures only the **sum along its chord**, so the 2-D fields must be reconstructed from line integrals. The inverse problem is under-determined, so the answer is a posterior over fields.
 
### Shell model (paper Sec. 2.3)
 
**Prior.** Each field is described by a radial profile on $R$ nodes $r_1 < \dots < r_R$, with a smooth Gaussian-process prior
 
$$
\tilde X_j = L_{pr} Z_j,\quad Z_j \sim \mathcal N(0, I_R),\quad
L_{pr} L_{pr}^\top = C_{pr},\quad
C_{pr}[i,k] = \exp\!\Big(-\frac{(r_i-r_k)^2}{2\tau^2}\Big) + \delta\,\mathbb 1_{i=k},
\qquad j \in \{A, T, V\}.
$$
 
The field is constant on circles around a random centre $c \sim \mathcal N(0, \sigma_c^2 I_2)$. For pixel $p$ at $x_p \in [-1,1]^2$, with $r_p = \lVert x_p - c \rVert$, the profiles are interpolated at $r_p$ and mapped to physical quantities:
 
$$
\mathcal A_p = c_A\,\mathcal S(\tilde X_A(r_p)),\qquad
\mathcal T_p = c_T\,\mathcal S(\tilde X_T(r_p)),\qquad
\mathcal V_p = c_V\,\tilde X_V(r_p),\qquad
\mathcal S(u) = \log(1+e^u).
$$
 
**Forward model.** Each pixel emits a Gaussian line in inverse frequency:
 
$$
E_{p}(\nu) = \frac{\mathcal A_p}{\sqrt{2\pi}\,w_p}
\exp\!\Big(-\frac{(\nu^{-1} - \tilde\nu_p^{-1})^2}{2 w_p^2}\Big),\qquad
\tilde\nu_p^{-1} = \frac{1-\mathcal V_p}{\nu_0},\qquad
w_p = \frac{\sqrt{\mathcal T_p}}{\nu_0}.
$$
 
A fixed integration matrix $I \in \mathbb R^{M\times P}$ sums emission along each chord: $\mu_{m}(\nu_k) = \sum_p I_{mp} E_p(\nu_k)$.
 
**Likelihood (paper eq. 4).** There is an additive noise floor plus proportional noise representing model error:
 
$$
y_{mk} \sim \mathcal N\big(\mu_{mk},\ \sigma^2 + \sigma_p^2\,\mu_{mk}^2\big).
$$
 
**Parameters:** $Z_A, Z_T, Z_V \in \mathbb R^R$ and $c_{raw} \in \mathbb R^2$ (with $c = \sigma_c c_{raw}$), for a dimension of $3R+2$. Everything else is data.
 
### Linear Gaussian toy (paper eqs. 1–2, Fig. 4)
 
The paper's own simplification keeps only the amplitude and drops the softplus and the proportional noise:
 
$$
x \sim \mathcal N(0, C_{pr}),\qquad y \sim \mathcal N(A x, \sigma^2 I_M).
$$
 
The posterior is exactly Gaussian. Two parameterizations are provided:
 
- **centred**, sampling $x$ directly;
- **non-centred**, sampling $Z$ with $x = L_{pr} Z$ (the paper's "prior reparameterization", Sec. 4.2).
With the included data, the paper's condition number (eq. 8) of the posterior is **117** (centred) against **3.2** (non-centred), which reproduces the effect shown in the paper's Fig. 4. The two models share one dataset and have the same posterior over $x$; their log densities differ by the constant $\log\lvert\det L_{pr}\rvert$.
 
## Why these posteriors are interesting
 
- **Ill-conditioning.** The integration operator is (numerically) low rank, so some directions are tightly constrained by the data and others only by the prior. The centred/non-centred pair isolates this effect in a case where the true posterior is known exactly.
- **Non-linearity and non-Gaussianity.** The shifted centre, the softplus transforms and the signal-dependent noise all contribute.
- **Potential multi-modality.** The paper shows that when the measured spectra have more peaks than the model can produce, several distinct explanations compete. The datasets here are simulated *from the model*, which should avoid that regime, but it is not yet verified across chains.
## Files
 
Following `doc/DATABASE_CONTENT.md`:
 
```
posterior_database/
├── models/stan/
│   ├── plasma_linear_noncentered.stan
│   ├── plasma_linear_centered.stan
│   └── plasma_shell.stan
├── models/pymc/                                # [if the PyMC versions land in this PR]
│   └── ...
├── models/info/
│   ├── plasma_linear_noncentered.info.json
│   ├── plasma_linear_centered.info.json
│   └── plasma_shell.info.json
├── data/data/
│   ├── plasma_linear.json.zip                  #  ~6 KB zipped
│   ├── plasma_shell_lite.json.zip              #  ~7 KB zipped
│   └── plasma_shell_full.json.zip              # ~26 KB zipped
├── data/data-raw/
│   ├── plasma_linear/plasma_linear.py
│   ├── plasma_shell_lite/plasma_shell_lite.py
│   └── plasma_shell_full/plasma_shell_full.py
├── data/info/
│   └── (one .info.json per dataset)
├── posteriors/
│   └── (four posterior .json files, reference_posterior_name: null)
└── bibliography/references.bib                 # + langmore2022hamiltonian
```
 
## Data: what comes from the paper and what is chosen
 
The paper does not publish its data. Each dataset is a single draw from the prior followed by simulated noisy measurements, with the seed recorded in the generator script.
 
<details>
<summary><b>Shell model constants</b></summary>

| Quantity | Full | Lite | Source |
|---|---|---|---|
| Domain | $[-1,1]^2$ | same | paper |
| Pixels $P$ | 20×20 | 10×10 | chosen |
| Chords $M$ | 20 vertical + 20 horizontal | 10 + 10 | paper uses 20 vertical; horizontal set added (see D6) |
| Frequencies $K$ | 60 | 30 | chosen (paper: ≈200) |
| Window | 229.5–229.9 | same | read off paper Fig. 3 |
| $\nu_0$ | 229.7 | same | inferred (C III line, Fig. 3) |
| $\tau$ | 0.2 | same | paper (one of the two values used) |
| $\delta$ | 0.001 | same | paper |
| Radial nodes $R$, range | 12 on [0, 3] | 8 on [0, 2] | chosen to cover all pixel distances |
| $\sigma_c$ | 0.1 | same | chosen |
| $\sigma$ | 5 | 15 | within the paper's range 1.25–15 |
| $\sigma_p$ | 0.05 | same | chosen |
| $c_A$ | 7e-4 | 2e-4 | chosen: peak signal ≈10³ (full), ≈3·10² (lite) |
| $c_T$ | 2e-8 | same | chosen: line width ≈1.5–4.5 frequency spacings |
| $c_V$ | 1e-4 | same | chosen: shift ≈ one line width per unit of $\tilde X_V$ |
| Peak signal-to-noise | ≈20 | ≈15 | resulting |
 
</details>

<details>

<summary><b>Linear toy constants</b></summary>
| Quantity | Value | Source |
|---|---|---|
| $N$, $M$ | 40, 20 | paper (Fig. 1/4) |
| $\tau$, $\delta$, $\sigma$ | 0.2, 0.001, 1 | paper (Fig. 4) |
| Grid | 40 equispaced points on [−1, 1] | chosen |
| Forward matrix $A$ | Each row is a Gaussian-weighted local average (width 0.2), rows sum to 1 | chosen; the paper's $A$ is not given |
 
</details>

## Open decisions
 
These are points where the paper is ambiguous or silent. For each, we state what we implemented and the alternative. **Feedback welcome**, particularly on D1–D3.
 
| # | Question | Implemented | Alternative / note |
|---|---|---|---|
| **D1** | Meaning of $\delta$ in the prior covariance (paper eq. 5 prints "$+\delta$") | $\delta$ added on the **diagonal** ($C + \delta I$) | Literal reading adds $\delta$ to every entry, a rank-one shift that does not do what the paper says it is for (preventing small eigenvalues) |
| **D2** | Line width $w$ | $w = \sqrt{\mathcal T}/\nu_0$, as printed | $\sqrt{\mathcal T/\nu_0}$ is also a possible reading of the PDF; it would change the scale of $c_T$ |
| **D3** | Shell or slab spatial structure | **Shell** only (the paper's main case) | Slab (fields constant on chord-aligned strips) is simpler and could be added later as another model |
| D4 | One centre for all three fields, or one each | One **shared** centre | Prior scale $\sigma_c$ and mean not given in the paper; possibly in its ref. [5], Dikovsky et al. (2021), *Phys. Plasmas* 28, 062503 |
| D5 | Interpolation of radial profiles | **Piecewise linear**, constant beyond the last node | The log density has kinks where $r_p$ crosses a node. A smooth scheme (e.g. cubic spline) would be cleaner for HMC but harder to match exactly across Stan/PyMC |
| D6 | Chord geometry | Vertical **and horizontal** chords | With vertical chords only (as in Fig. 2), the likelihood is exactly symmetric under $c_y \to -c_y$, which builds in a bimodality |
| D7 | Noise parameters | $\sigma$, $\sigma_p$ **fixed** (data) | They could be inferred instead, which adds two parameters |
| D8 | Physical constants $c_A, c_T, c_V$, grids | Chosen so the simulated spectra resemble paper Fig. 3 | Not given in the paper |
| D9 | Linear-toy forward matrix | Gaussian local-average rows | Not given in the paper |
| D10 | Include both linear parameterizations? | **Yes**, as a centred/non-centred pair (like `eight_schools`) | Only the non-centred one could be added |
| D11 | Full shell dataset | Included | Expensive to sample |
 
A purely numerical reformulation: the code uses $\tilde\nu^{-1} = (1-\mathcal V)/\nu_0$ directly instead of $\tilde\nu = \nu_0/(1-\mathcal V)$. This is mathematically identical and avoids a division by zero at $\mathcal V = 1$.


## Questions for maintainers
 
1. Are posteriors without reference draws welcome for models of this cost, in particular `plasma_shell_full`?
2. Naming: are `plasma_linear`, `plasma_shell_lite` and `plasma_shell_full` acceptable, or is a different convention preferred for simulated datasets?
3. Both shell datasets come from one generator script with a `lite`/`full` switch. Is it fine to have one script per `data-raw` folder that calls the shared generator, or should the code be duplicated?
4. Which keywords should we use? Our suggestions are `inverse problem`, `ill-conditioned`, `simulated data`.

## Checklist
 
- [ ] Models, data, info files and posterior files added following `doc/DATABASE_CONTENT.md`
- [ ] Data zipped; matrices stored as lists of rows
- [ ] Generator scripts in `data/data-raw/`
- [ ] BibTeX entry `langmore2022hamiltonian` added to `bibliography/references.bib`
- [ ] `added_by` / `added_date` filled in: *[name]*, *[date]*
- [ ] Open decisions D1–D3 agreed with supervisor
- [ ] Stan/PyMC log-density equivalence check completed

<details>
<summary>BibTeX</summary>
```bibtex
@misc{langmore2022hamiltonian,
  title         = {Hamiltonian {M}onte {C}arlo in Inverse Problems; Ill-Conditioning and Multi-Modality},
  author        = {Langmore, Ian and Dikovsky, Michael and Geraedts, Scott and Norgaard, Peter and von Behren, Rob},
  year          = {2022},
  eprint        = {2103.07515},
  archivePrefix = {arXiv},
  primaryClass  = {stat.CO}
}
```
 
</details>
