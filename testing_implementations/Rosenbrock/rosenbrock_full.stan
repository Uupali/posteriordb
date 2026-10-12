data {
    int<lower=2> N;  // number of dimensions
    real<lower=0> a; // scale for the first variable's marginal variance
    real<lower=0> b; // scale for the conditional variance
    real mu;         // mean for the first variable
}
parameters {
    vector[N] x; 
}
model {
    target += -a * square(x[1] - mu);

    for (i in 2:N) {
        target += -b * square(x[i] - square(x[i-1]));
    }
}
