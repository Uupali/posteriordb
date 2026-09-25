# Mathematical Construction of the Banana-Shaped Target Distributions

gathered some info on the banana-shaped target distributions used as test cases in the paper. 

details taken from **Section 5.1** (“The test target distributions”).

---

## 1. Base (Untwisted) Distribution

Let $x = (x_1, x_2, \dots, x_D)^\top \in \mathbb{R}^D$.

The base distribution is the centred multivariate normal

$$
x \sim \mathcal{N}(0, C_1),
\qquad
C_1 = \operatorname{diag}(v, 1, \dots, 1),
\qquad
v = 100.
$$

Its unnormalised log-density is

$$
\log f(x)
= -\frac{1}{2} \left(
\frac{x_1^2}{v}
+ \sum_{i=2}^{D} x_i^2
\right).
$$

*(Source: Section 5.1)*

---

## 2. Twisting Transformation $\phi_b$

The observed (twisted) variable $y$ is obtained from $x$ by the map

$$
\phi_b(x)
=
\left(
x_1,\ 
x_2 + b(x_1^2 - v),\ 
x_3,\ 
\dots,\ 
x_D
\right),
\qquad
v = 100.
$$

The inverse map (required when expressing the density in terms of $y$) is

$$
\begin{aligned}
x_1 &= y_1, \\
x_2 &= y_2 - b(y_1^2 - v), \\
x_i &= y_i \quad \text{for } i = 3,\dots,D.
\end{aligned}
$$

*(Source: Section 5.1, page 9)*

---

## 3. Jacobian Determinant

The transformation $\phi_b$ modifies only the second coordinate and does so conditionally on the first. The Jacobian matrix is therefore lower-triangular with ones on the diagonal:

$$
\det(J_{\phi_b}) = 1
\qquad \Rightarrow \qquad
\log|\det J_{\phi_b}| = 0.
$$

Consequently, no volume-correction term appears in the log-density.

*(Source: Section 5.1, page 9 — “the determinant of the Jacobian of $\phi_b$ is identically equal to 1”)*

---

## 4. Target (Banana) Log-Density

Substituting the inverse map into the base log-density yields the density of the twisted banana distribution:

$$
\log f_b(y)
= -\frac{1}{2} \left(
\frac{y_1^2}{v}
+ \bigl(y_2 - b(y_1^2 - v)\bigr)^2
+ \sum_{i=3}^{D} y_i^2
\right),
\qquad
v = 100.
$$

This is the exact expression that must be implemented as the target log-density.



---

## 5. Parameter Settings Used in the Experiments

| Quantity          | Value(s)          | Meaning / Label in Paper       | Source              |
|-------------------|-------------------|--------------------------------|---------------------|
| Variance $v$      | $100$             | Variance of first coordinate   | Section 5.1, p. 8   |
| Curvature $b$     | $0.03$            | Moderately twisted ($\pi_3$)   | Section 5.1, p. 9   |
|                   | $0.10$            | Strongly twisted ($\pi_4$)     | Section 5.1, p. 9   |
| Dimension $D$     | $\{2, 4, 8\}$      | Dimensions for non-linear targets | Section 5.1, p. 9 |



trying to modify some details referring to these

commented out the earlier one, there was a slight sign mismatch which I've tried to fix in both the versions and run the verifications accordingly (still in progress, will continue from here again)