library(posterior)
library(rstan)
library(posteriordb)
library(jsonlite)
options(mc.cores = parallel::detectCores())

# ==============================================================================
# 1. SETUP & SAMPLING
# ==============================================================================
# rosenbrock_2d-rosenbrock, rosenbrock_n3_chain-rosenbrock, rosenbrock_n5_hybrid-rosenbrock
posterior_name <- "rosenbrock_2d-rosenbrock" # Change this as needed
my_pdb <- pdb_local(".") # Or "posterior_database" depending on your working directory
po <- posterior(posterior_name, my_pdb)

data <- pdb_data(po)
code <- stan_code(po)

mod <- stan_model(model_code = code, model_name = po$model_info$name)

fit <- sampling(
  mod,
  data = data,
  chains = 10,
  iter = 20000,
  warmup = 10000,
  thin = 10,
  seed = 42,
  control = list(
    adapt_delta = 0.99,
    max_treedepth = 15
  )
)

# ==============================================================================
# 2. QUALITY CHECKS
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

results <- check_posterior_quality(fit, verbose = TRUE)

# ==============================================================================
# 3. PREPARE DRAWS & DIAGNOSTICS
# ==============================================================================
draws <- as_draws_df(fit)
draws_list <- as.list(draws)

# Remove internal metadata, lp__, and auxiliary parameters like x_raw
draws_list <- draws_list[!grepl("^\\.", names(draws_list))]
draws_list <- draws_list[!grepl("^lp__$", names(draws_list))]
draws_list <- draws_list[!grepl("^x_raw", names(draws_list))] # Adjust if other aux variables exist

draws_summary <- summarize_draws(draws, mean, sd, rhat, ess_bulk, ess_tail)
summary_clean <- draws_summary[!grepl("^lp__$|^x_raw", draws_summary$variable), ]

# ==============================================================================
# 4. SAVE DRAWS TO ZIPPED JSON
# ==============================================================================
zip_path <- "posterior_database/reference_posteriors/draws/draws"
json_draws_filename <- paste0(posterior_name, ".json")
zip_draws_filename <- paste0(posterior_name, ".zip")

write_json(draws_list, file.path(zip_path, json_draws_filename), pretty = TRUE, auto_unbox = TRUE, digits = 10)
zip(zipfile = file.path(zip_path, zip_draws_filename), files = file.path(zip_path, json_draws_filename))
file.remove(file.path(zip_path, json_draws_filename))

# ==============================================================================
# 5. ENVIRONMENT & METADATA 
# ==============================================================================
r_session_string <- paste(capture.output(sessionInfo()), collapse = "\n")
makevars_path <- file.path(Sys.getenv("HOME"), ".R", "Makevars")
r_makevars_string <- if (file.exists(makevars_path)) paste(readLines(makevars_path), collapse = "\n") else ""

info_metadata <- list(
  name = posterior_name,
  inference = list(
    method = "stan_sampling",
    method_arguments = list(
      chains = 10,
      iter = 20000,
      warmup = 10000,
      thin = 10,
      seed = 42,
      control = list(
        adapt_delta = 0.99, 
        max_treedepth = 15
      )
    )
  ),
  diagnostics = list(
    diagnostic_information = list(
      names = summary_clean$variable
    ),
    ndraws = posterior::ndraws(draws),
    nchains = posterior::nchains(draws),
    effective_sample_size_bulk = summary_clean$ess_bulk,
    effective_sample_size_tail = summary_clean$ess_tail,
    r_hat = summary_clean$rhat,
    expected_fraction_of_missing_information = as.list(rstan::get_bfmi(fit)),
    divergent_transitions = rstan::get_num_divergent(fit)
  ),
  checks_made = list(
    ndraws_is_10k = TRUE,
    nchains_is_gte_4 = TRUE,
    ess_within_bounds = TRUE,
    r_hat_below_1_01 = TRUE, # Manually verify based on printout
    efmi_above_0_2 = TRUE,
    abs_mean_lag1_ac_below_0_05 = FALSE
  ),
  comments = NULL,
  added_by = "Michal Kaminski",
  added_date = as.character(Sys.Date()),
  versions = list(
    rstan_version = paste("rstan", as.character(packageVersion("rstan"))),
    r_Makevars = r_makevars_string,
    r_version = R.version.string,
    r_session = r_session_string
  )
)

info_filename <- paste0(posterior_name, ".info.json")
info_path <- file.path("posterior_database/reference_posteriors/draws/info", info_filename) 

write_json(info_metadata, info_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
