/*data {
    int<lower=2> D; // dimensions
    real<lower=0> v; // variance of the first dimension
    real<lower=0> b; //curvature parameter
}
parameters {
   vector[D] y; //
}
model {
    target += -0.5 * (y[1]^2)/(v) - 0.5 * (y[2] + b * (y[1]^2 - v))^2 - 0.5 * dot_self(y[3:D]);
}
*/
data {
  int<lower=2> D;        // Target dimension (e.g., 2, 4, 8 from Haario et al., 1999)
}

transformed data {
  real v = 100.0;        // Variance of first coordinate y[1] (Haario et al., 1999)
  real b = 0.1;         // Curvature parameter for parabolic transformation
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