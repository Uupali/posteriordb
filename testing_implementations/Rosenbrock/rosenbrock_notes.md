# Rosenbrock distribution notes

Key feature of the Rosenbrock function - its minuimum  is located at the bottom of a long and narrow parabolic valley.

Cited paper proposes the **Hybrid Rosenbrock** distribution which is supposed to be a good benchamrk:
- easily extensible to higher dimensions
- has complex marginal density structure
- normalising constant is known
- effect of parameters is known
- MC samples are easy to obtain

## Simplest non-trivial case of Rosenbrock distribution (2D):

$$
\pi(x_1, x_2) \propto \exp\left\{- [100(x_2 - x_1^2)^2 + (1-x_1)^2] / 20\right\}
$$


## n-dimensional extension


$$
\pi(x) \propto \exp\left\{ - \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1-x_i)^2] / 20 \right\}
$$



Note: as n increases, the variance of xn increases steeply, as each new random variable is directly dependent on the squared value of the previous one. This makes the distribution increasingly difficult to sample from as n grows.

The paper demonstrates that the Full Rosenbrock extension (chaining $x_{i+1}$ on $x_i^2$) suffers from collapsing marginals as dimensions increase, causing numerical instability when $n$ grows.  

The Hybrid Rosenbrock extension (Figure 7) solves this using a block structure ($n_1$ variables per block across $n_2$ independent blocks). This maintains challenging non-linear correlations locally while keeping global variances computationally stable as total dimension $n = (n_1 - 1)n_2 + 1$ scales.

## What to add to the database?

1. Standard 2D Rosenbrock distribution 

Data: a: 0.05, b: 5, mu:1
Form: $\pi(x_1, x_2) \propto \exp\left\{-a(x_1 - \mu)^2 - b(x_2 - x_1^2)^2\right\}$
Rationale: Serves as teh standard baseline in MCMC 

2. Full chained rosebrock 

Data: 
- rosenbrock_full_3d N: 3, a: 0.05, b: 5, mu:1
- rosenbrock_full_5d N: 4, a: 0.05, b: 5, mu:1

Rationale: Demosntarates the traditional N-dimensional sequential extension
Formulation: $\pi(x) \propto \exp\left\{-a(x_1 - \mu)^2 - b \sum_{i=2}^{N} (x_i - x_{i-1}^2)^2\right\}$

Note: Both of these distributions will be one stan or pymc model `rosenbrock_full` with different datasets

3. Hybrid rosenbrock

Name: `rosenbrock_hybrid`
Form: $\pi(x) \propto \exp\left\{-a(x_1 - \mu)^2 - \sum_{j=1}^{n_2}\sum_{i=2}^{n_1} b_{j,i}(x_{j,i} - x_{j,i-1}^2)^2\right\}$ equation (12) in the paper
Rationale: Primary contribution of the paper, keeps tail variance tractable in higher dimensions while testing algorithms against nonplinear 2D marginal ridges
Data:
- rosenbrock_hybrid_3d N: 3, n1: 2, n2: 1, a: 0.05, b: 5, mu:1
- rosenbrock_hybrid_5d N: 5, n1: 3, n2: 2, a: 0.05, b: 5, mu:1 (already implemented in `trial.py` I think)