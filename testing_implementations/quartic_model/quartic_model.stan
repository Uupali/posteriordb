data {
  real y;   
  real<lower=0> sigma;
}
parameters {
  vector[2] theta;   // theta[1] = θ₀ (quartic term), theta[2] = θ₁
}
model {
  real F = square(theta[2]) + square(theta[1]) * (square(theta[1]) - 0.5);
  target += std_normal_lpdf(theta);
  target += normal_lpdf(y | F, sigma);
}