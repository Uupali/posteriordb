library(rstan)
library(posterior)
library(jsonlite)

options(mc.cores = 4)
rstan_options(auto_write = TRUE)   # cache compiled models
setwd("C:/Users/mkami/OneDrive/Pulpit/UNI/posteriordb/testing_implementations/plasma_model")
dat <- read_json("plasma_linear.json", simplifyVector = TRUE)
ref <- read_json("plasma_linear_reference.json", simplifyVector = TRUE)
N <- length(ref$x_mean)
x_names <- sprintf("x[%d]", seq_len(N))

check_model <- function(stan_file, n_chains = 10) {
  fit <- stan(stan_file, data = dat, chains = n_chains,
              iter = 2000, warmup = 1000, seed = 2024, refresh = 0)

  draws <- as_draws_array(as.array(fit, pars = "x"))
  s <- summarise_draws(draws, "mean", "sd", "rhat", "ess_bulk", "ess_tail")
  s <- s[match(x_names, s$variable), ]

  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  div  <- sum(sapply(sp, function(x) sum(x[, "divergent__"])))
  tdep <- max(sapply(sp, function(x) max(x[, "treedepth__"])))
  nlf  <- mean(sapply(sp, function(x) mean(x[, "n_leapfrog__"])))

  z_mean   <- (s$mean - ref$x_mean) / (ref$x_sd / sqrt(s$ess_bulk))
  sd_ratio <- s$sd / ref$x_sd

  cat("==", stan_file, "==\n")
  cat(sprintf("  divergences: %d   max treedepth: %d   mean leapfrog steps: %.0f\n",
              div, tdep, nlf))
  cat(sprintf("  max R-hat: %.4f   min ESS bulk: %.0f   min ESS tail: %.0f\n",
              max(s$rhat), min(s$ess_bulk), min(s$ess_tail)))
  cat(sprintf("  posterior mean vs analytic: max |z| = %.2f (MC-error units)\n",
              max(abs(z_mean))))
  cat(sprintf("  posterior sd / analytic sd: %.3f - %.3f\n\n",
              min(sd_ratio), max(sd_ratio)))
  invisible(fit)
}

fit_nc <- check_model("plasma_linear_noncentered.stan")
fit_c  <- check_model("plasma_linear_centered.stan")
