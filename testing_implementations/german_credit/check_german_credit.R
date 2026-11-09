library(posterior)
library(rstan)
library(jsonlite)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)  # cache compiled models between runs

# ==============================================================================
# 0. CHOOSE WHAT TO RUN
# ==============================================================================
# Everything is expected in the working directory:
#   german_credit.json, german_credit_interactions.json   (from make_german_credit_data.py)

configs <- list(
  lr = list(
    data_file  = "german_credit.json",
    stan_file  = "german_credit_lr.stan",
    name       = "german_credit-logistic_regression"
  ),
  hlr = list(
    data_file  = "german_credit_interactions.json",
    stan_file  = "german_credit_hlr.stan",
    name       = "german_credit_interactions-hierarchical_logistic_regression"
  ),
  hlr_nc = list(
    data_file  = "german_credit_interactions.json",
    stan_file  = "german_credit_hlr_nc.stan",
    name       = "german_credit_interactions-hierarchical_logistic_regression_noncentered"
  )
)

which_model <- "hlr_nc"   # "lr", "hlr" or "hlr_nc"

# Set TRUE for a fast trial run (fewer iterations) before committing to the
# full reference settings. The draw-count check will FAIL in quick mode by design.
quick <- TRUE

cfg <- configs[[which_model]]
setwd("C:/Users/mkami/OneDrive/Pulpit/UNI/posteriordb/testing_implementations/german_credit")

# ==============================================================================
# 1. SETUP & SAMPLING
# ==============================================================================
data <- jsonlite::read_json(cfg$data_file, simplifyVector = TRUE)
data$X <- as.matrix(data$X)
data$y <- as.integer(data$y)
stopifnot(nrow(data$X) == data$N, ncol(data$X) == data$D, length(data$y) == data$N)

mod <- stan_model(file = cfg$stan_file, model_name = cfg$name)

sampling_args <- if (quick) {
  list(chains = 4, iter = 2000, warmup = 1000, thin = 1)
} else {
  list(chains = 10, iter = 20000, warmup = 10000, thin = 10)
}

t0 <- Sys.time()
fit <- sampling(
  mod,
  data = data,
  chains = sampling_args$chains,
  iter = sampling_args$iter,
  warmup = sampling_args$warmup,
  thin = sampling_args$thin,
  seed = 42,
  control = list(
    adapt_delta = 0.99,
    max_treedepth = 15
  )
)
cat(sprintf("\nSampling time for %s: %.1f min\n",
            cfg$name, as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# ==============================================================================
# 2. QUALITY CHECKS (unchanged from check_posterior.R)
# ==============================================================================
check_total_draws <- function(draws, threshold = 10000) {
  total_draws <- posterior::ndraws(draws)
  list(total_draws = total_draws, pass_draws = total_draws >= threshold)
}
check_divergences <- function(fit) {
  divergences <- rstan::get_num_divergent(fit)
  list(divergences = divergences, pass_div = divergences == 0)
}
check_rhat <- function(draws_summary, threshold = 1.01) {
  max_rhat <- max(draws_summary$rhat, na.rm = TRUE)
  list(max_rhat = max_rhat, pass_rhat = max_rhat < threshold)
}
check_bfmi <- function(fit, threshold = 0.2) {
  bfmi_per_chain <- rstan::get_bfmi(fit)
  min_bfmi <- min(bfmi_per_chain)
  list(chain_values = as.list(bfmi_per_chain), min_bfmi = min_bfmi, pass_bfmi = min_bfmi >= threshold)
}
calc_mean_lag1_ac <- function(var_matrix) {
  acs <- apply(var_matrix, 2, function(chain_vals) {
    acf(chain_vals, lag.max = 1, plot = FALSE)$acf[2]
  })
  mean(acs)
}
check_lag1_ac <- function(draws_arr, threshold = 0.05) {
  lag1_acs <- apply(draws_arr, 3, calc_mean_lag1_ac)
  max_abs_ac <- max(abs(lag1_acs), na.rm = TRUE)
  list(max_abs_ac = max_abs_ac, pass_ac = max_abs_ac < threshold)
}

check_posterior_quality <- function(fit, verbose = TRUE) {
  draws <- posterior::as_draws_df(fit)
  draws_summary <- posterior::summarize_draws(draws, mean, sd, rhat, ess_bulk, ess_tail)
  draws_arr <- as.array(fit)
  results <- list()

  results$total_draws <- check_total_draws(draws)
  results$rhat <- check_rhat(draws_summary)
  results$divergences <- check_divergences(fit)
  results$bfmi <- check_bfmi(fit)
  results$lag1_ac <- check_lag1_ac(draws_arr)

  if (verbose) {
    cat("\n=== PosteriorDB Reference Quality Report ===\n")
    cat(sprintf("[ %s ] Total Draws: %d (Required: >= 10000)\n", ifelse(results$total_draws$pass_draws, "PASS", "FAIL"), results$total_draws$total_draws))
    cat(sprintf("[ %s ] Max R-hat: %.4f (Required: < 1.01)\n", ifelse(results$rhat$pass_rhat, "PASS", "FAIL"), results$rhat$max_rhat))
    cat(sprintf("[ %s ] Divergent Transitions: %d (Required: == 0)\n", ifelse(results$divergences$pass_div, "PASS", "FAIL"), results$divergences$divergences))
    cat(sprintf("[ %s ] Min E-FMI: %.4f (Required: >= 0.2)\n", ifelse(results$bfmi$pass_bfmi, "PASS", "FAIL"), results$bfmi$min_bfmi))
    cat(sprintf("[ %s ] Max Abs Lag-1 Autocorrelation: %.4f (Required: < 0.05)\n", ifelse(results$lag1_ac$pass_ac, "PASS", "FAIL"), results$lag1_ac$max_abs_ac))
    cat("============================================\n")
  }
  return(results)
}

cat(sprintf("\nPosterior: %s\n", cfg$name))
results <- check_posterior_quality(fit, verbose = TRUE)

# ==============================================================================
# 3. EXTRA DIAGNOSTICS (to see *what* fails, not just whether)
# ==============================================================================
sp <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
treedepth <- unlist(lapply(sp, function(x) x[, "treedepth__"]))
stepsize  <- sapply(sp, function(x) x[1, "stepsize__"])
cat(sprintf("Max treedepth hit: %d / %d iterations (max_treedepth = 15)\n",
            sum(treedepth >= 15), length(treedepth)))
cat(sprintf("Mean treedepth: %.2f | step size per chain: %s\n",
            mean(treedepth), paste(signif(stepsize, 3), collapse = ", ")))

# Parameters with the worst R-hat and lag-1 autocorrelation
draws_summary <- summarize_draws(as_draws_df(fit), mean, sd, rhat, ess_bulk, ess_tail)
cat("\nWorst 5 parameters by R-hat:\n")
print(head(draws_summary[order(-draws_summary$rhat), ], 5))

draws_arr <- as.array(fit)
lag1 <- apply(draws_arr, 3, calc_mean_lag1_ac)
cat("\nWorst 5 parameters by |lag-1 autocorrelation|:\n")
print(head(sort(abs(lag1), decreasing = TRUE), 5))
