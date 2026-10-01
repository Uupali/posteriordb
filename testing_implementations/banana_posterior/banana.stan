data {
  int<lower=2> D;        // Target dimension (e.g., 2, 4, 8 from Haario et al., 1999)
  real<lower=0> b;         // Curvature parameter for parabolic transformation
}

transformed data {
  real v = 100.0;        // Variance of first coordinate y[1] (Haario et al., 1999)
}

parameters {
  vector[D] y;
}

model {
  // First coordinate: N(0, sqrt(v))
  y[1] ~ normal(0, sqrt(v));

  // Second coordinate twisted by parabola
  y[2] ~ normal(-b * (square(y[1]) - v), 1.0);

  // Remaining dimensions (if D > 2)
  if (D > 2) {
    y[3:D] ~ normal(0, 1.0);
  }
}
