# Proposed Addition: Hybrid Rosenbrock Distribution Benchmark

## Background & Rationale

Following the recommendations in Pagani et al. (2020), *"An n-dimensional Rosenbrock distribution for MCMC testing"*, we propose adding the **Hybrid Rosenbrock distribution** to `posteriordb`.

While the classic 2D Rosenbrock kernel is a standard MCMC test case (Goodman & Weare, 2010), extending it to \(n\) dimensions via simple chaining (*Full Rosenbrock*) compresses lower-dimensional marginals toward the mode and leads to exponential variance growth (Pagani et al., 2020). Alternatively, uncoupled pair constructions (*Even Rosenbrock*) lack the complex dependency structure desired for challenging high-dimensional sampling problems (Dixon, 1994; Pagani et al., 2020).

The **Hybrid Rosenbrock distribution** addresses these limitations through several desirable properties:

1. **Complex 2D Marginals**  
   Every 2D marginal pair remains a non-linear curved or straight narrow ridge, preserving challenging geometry across dimensions.

2. **Analytically Known Normalizing Constant**  
   The normalizing constant is available in closed form:

   $$
   \frac{\sqrt{a}\prod \sqrt{b_{j,i}}}{\pi^{n/2}}
   $$

   enabling exact marginal likelihood benchmarking.

3. **Direct Ground-Truth Sampling**  
   Exact i.i.d. reference samples can be generated sequentially without MCMC autocorrelation.

4. **Controlled Scale Mismatch**  
   The construction permits severe coordinate variance differences (\(n_1 \geq 3\)) while remaining numerically stable through parallel dependency blocks.

These properties make the Hybrid Rosenbrock distribution a useful addition to PosteriorDB's collection of benchmark targets for evaluating MCMC and variational inference methods.

---

## Proposed Benchmark Posteriors

Rather than duplicating model implementations, we propose a **single parameterized model** implemented in both Stan and PyMC. The model is controlled by the data hyperparameters

$$
(n_1, n_2, \mu, a, b)
$$

with overall dimension

$$
n = (n_1 - 1)n_2 + 1.
$$

The following benchmark instances are proposed:

### 1. `rosenbrock_2d`

- Parameters: \(n_1 = 2,\; n_2 = 1\)
- Dimension: \(D = 2\)
- Corresponds to the minimal parabolic ridge benchmark.
- Equivalent to Model 1 in Section 6.2 of Pagani et al. (2020).

### 2. `rosenbrock_n3_chain`

- Parameters: \(n_1 = 3,\; n_2 = 1\)
- Dimension: \(D = 3\)
- A single three-variable chain.
- Isolates the onset of coordinate scale mismatch.
- Equivalent to Model 3 in Section 6.2 of Pagani et al. (2020).

### 3. `rosenbrock_n5_hybrid`

- Parameters: \(n_1 = 3,\; n_2 = 2\)
- Dimension: \(D = 5\)
- A multi-block hybrid topology in which a shared root variable branches into two parallel chains.
- Equivalent to Model 6 (Equation 14) in Section 6.1 of Pagani et al. (2020).

---

## Proposed Directory Structure

```text
posterior_database/
├── bibliography/
│   └── references.bib
│
├── data/
│   ├── data-raw/
│   │   ├── rosenbrock_2d/
│   │   │   └── rosenbrock_2d.R
│   │   ├── rosenbrock_n3_chain/
│   │   │   └── rosenbrock_n3_chain.R
│   │   └── rosenbrock_n5_hybrid/
│   │       └── rosenbrock_n5_hybrid.R
│   │
│   ├── data/
│   │   ├── rosenbrock_2d.json.zip
│   │   ├── rosenbrock_n3_chain.json.zip
│   │   └── rosenbrock_n5_hybrid.json.zip
│   │
│   └── info/
│       ├── rosenbrock_2d.info.json
│       ├── rosenbrock_n3_chain.info.json
│       └── rosenbrock_n5_hybrid.info.json
│
├── models/
│   ├── info/
│   │   └── rosenbrock.info.json
│   │
│   ├── pymc/
│   │   └── rosenbrock.py
│   │
│   └── stan/
│       └── rosenbrock.stan
│
└── posteriors/
    ├── rosenbrock_2d-rosenbrock.json
    ├── rosenbrock_n3_chain-rosenbrock.json
    └── rosenbrock_n5_hybrid-rosenbrock.json
```

### Raw-data definitions

```r
# data/data-raw/rosenbrock_2d/rosenbrock_2d.R
rosenbrock_2d <- list(
  n1 = 2,
  n2 = 1,
  mu = 1.0,
  a = 0.05,
  b = 5.0
)
```

```r
# data/data-raw/rosenbrock_n3_chain/rosenbrock_n3_chain.R
rosenbrock_n3_chain <- list(
  n1 = 3,
  n2 = 1,
  mu = 1.0,
  a = 0.05,
  b = 5.0
)
```

```r
# data/data-raw/rosenbrock_n5_hybrid/rosenbrock_n5_hybrid.R
rosenbrock_n5_hybrid <- list(
  n1 = 3,
  n2 = 2,
  mu = 1.0,
  a = 0.05,
  b = 5.0
)
```