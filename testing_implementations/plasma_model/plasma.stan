functions {
  vector softplus(vector u) {
    return log1p_exp(u);
  }
}

data {
  // Dimensions
  int<lower=1> N; // Number of spatial grid points
  int<lower=1> M; // Number of measurement chords
  int<lower=1> K; // Number of observed frequencies
  
  // Model Constants & Inputs
  vector[K] nu;       // Array of observed frequencies
  real nu_0;          // Base rest frequency
  matrix[M, N] I_mat; // Integration matrix projecting spatial grid to measurement chords
  matrix[M, K] Y;     // Measured line-integrated emission data
  
  // Prior Covariance Structure
  matrix[N, N] L_pr;  // Cholesky factor of the prior covariance matrix for spatial correlation
  
  // Scaling Factors
  real c_A; // Amplitude scaling factor
  real c_T; // Temperature scaling factor
  real c_V; // Velocity scaling factor
  
  // Noise Hyperparameters
  real<lower=0> sigma;   // Additive noise baseline
  real<lower=0> sigma_p; // Proportional noise scaling representing model error
}

parameters {
  // Unconstrained, independent standard normal variables. 
  // This utilizes prior-reparameterization (a non-centered parameterization) 
  // to prevent ill-conditioning of the posterior
  vector[N] Z_A; 
  vector[N] Z_T;
  vector[N] Z_V;
}

transformed parameters {
  // 1. Impart spatial correlation using the Cholesky factor of the prior
  vector[N] X_tilde_A = L_pr * Z_A;
  vector[N] X_tilde_T = L_pr * Z_T;
  vector[N] X_tilde_V = L_pr * Z_V;
  
  // 2. Map to physical state variables, using softplus for strictly positive values
  vector[N] A = c_A * softplus(X_tilde_A);
  vector[N] T = c_T * softplus(X_tilde_T);
  vector[N] V = c_V * X_tilde_V;
  
  // 3. Compute Doppler physics variables
  vector[N] nu_tilde = nu_0 ./ (1.0 - V);  // './' computes elementwise division
  vector[N] w = sqrt(T) / nu_0;
  
  // 4. Calculate projected expectations and standard deviations for the likelihood
  matrix[M, K] expected_Y;
  matrix[M, K] sigma_Y;
  
  for (k in 1:K) {
    vector[N] emissivity;
    
    // Dimensionless discrete emissivity spectrum for each grid point
    for (n in 1:N) {
      emissivity[n] = (A[n] / (sqrt(2 * pi()) * w[n])) * 
                      exp(-pow(inv(nu[k]) - inv(nu_tilde[n]), 2) / (2 * pow(w[n], 2)));
    }
    
    // Project emissivity onto the M measurement chords
    expected_Y[, k] = I_mat * emissivity;
    
    // Calculate the signal-dependent standard deviation for each chord
    for (m in 1:M) {
      sigma_Y[m, k] = sqrt(square(sigma) + square(sigma_p * expected_Y[m, k]));
    }
  }
}

model {
  // Assign standard normal priors to the raw parameters
  Z_A ~ std_normal();
  Z_T ~ std_normal();
  Z_V ~ std_normal();
  
  // Evaluate the custom likelihood using the signal-proportional noise model
  for (k in 1:K) {
    Y[, k] ~ normal(expected_Y[, k], sigma_Y[, k]);
  }
}