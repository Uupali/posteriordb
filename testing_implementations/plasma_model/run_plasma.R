# Fit the plasma shell model with rstan and run basic diagnostics.
# Needs: rstan, posterior, jsonlite. Optional: bayesplot.
# Run from the folder containing plasma_shell.stan and the data/truth JSON files.
#   Rscript run_plasma.R                 # full dataset (plasma_shell_sim.json)
#   Rscript run_plasma.R lite            # lite dataset (plasma_shell_lite_sim.json)
# In an interactive session, set `preset` by hand instead of using commandArgs.

library(rstan)
library(posterior)
library(jsonlite)

options(mc.cores = 4)
rstan_options(auto_write = TRUE)   # cache the compiled model next to the .stan file

args   <- commandArgs(trailingOnly = TRUE)
preset <- if (length(args) > 0) args[1] else "full"
stopifnot(preset %in% c("full", "lite"))
data_file  <- if (preset == "full") "plasma_shell_sim.json"   else "plasma_shell_lite_sim.json"
truth_file <- if (preset == "full") "plasma_shell_truth.json" else "plasma_shell_lite_truth.json"
n_chains   <- 4        # raise to 10 for the posteriordb reference run

# simplifyVector turns nested JSON arrays into R matrices (I_mat, Y, xy),
# which is what rstan expects for Stan `matrix` data.
dat   <- read_json(data_file,  simplifyVector = TRUE)
truth <- read_json(truth_file, simplifyVector = TRUE)
R <- dat$R

# Initialise from the prior (as in the paper); chains started far from each other
# are what reveals multi-modality via R-hat.
set.seed(2024)
inits <- lapply(seq_len(n_chains), function(i)
  list(Z_A = rnorm(R), Z_T = rnorm(R), Z_V = rnorm(R), c_raw = rnorm(2)))

fit <- stan(
  file    = "plasma_shell.stan",
  data    = dat,
  chains  = n_chains,
  iter    = 2000,          # includes warmup
  warmup  = 1000,
  seed    = 2024,
  init    = inits,
  control = list(adapt_delta = 0.95),
  refresh = 100
)

# ---- diagnostics -------------------------------------------------------------
cat("\n== Sampler diagnostics (post-warmup) ==\n")
sp <- get_sampler_params(fit, inc_warmup = FALSE)
diag_tab <- data.frame(
  chain          = seq_along(sp),
  divergences    = sapply(sp, function(x) sum(x[, "divergent__"])),
  max_treedepth  = sapply(sp, function(x) max(x[, "treedepth__"])),
  mean_leapfrog  = sapply(sp, function(x) mean(x[, "n_leapfrog__"])),
  stepsize       = sapply(sp, function(x) x[1, "stepsize__"])
)
print(diag_tab)
print(get_bfmi(fit))       # energy diagnostic per chain; values below ~0.3 are a warning sign

draws <- as_draws_array(as.array(fit, pars = c("ctr", "Z_A", "Z_T", "Z_V", "lp__")))

cat("\n== Summary (centre, lp__) ==\n")
print(summarise_draws(subset_draws(draws, variable = c("ctr", "lp__")),
                      "mean", "sd", "rhat", "ess_bulk", "ess_tail"))

all_sum <- summarise_draws(subset_draws(draws, variable = c("ctr", "Z_A", "Z_T", "Z_V")),
                           "rhat", "ess_bulk", "ess_tail")
cat("\nmax R-hat over all parameters:", max(all_sum$rhat, na.rm = TRUE), "\n")
cat("min bulk ESS over all parameters:", min(all_sum$ess_bulk, na.rm = TRUE), "\n")
cat("min tail ESS over all parameters:", min(all_sum$ess_tail, na.rm = TRUE), "\n")

# Per-chain means: chains sitting in different modes show up here.
cat("\n== Per-chain means ==\n")
arr <- as.array(fit, pars = c("ctr", "lp__"))          # iterations x chains x parameters
print(round(apply(arr, c(2, 3), mean), 4))

cat("\n== True centre used in the simulation ==\n")
print(truth$ctr)

# ---- optional plots ----------------------------------------------------------
if (requireNamespace("bayesplot", quietly = TRUE)) {
  library(bayesplot)
  print(mcmc_trace(as.array(fit), pars = c("ctr[1]", "ctr[2]", "lp__")))
  print(mcmc_pairs(as.array(fit), pars = c("ctr[1]", "ctr[2]")))
}

saveRDS(fit, paste0("plasma_fit_", preset, ".rds"))   # reload later with readRDS()
