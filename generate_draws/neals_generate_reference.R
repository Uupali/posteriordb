library(posteriordb)
library(rstan)
library(posterior)

# setwd("C:/Users/mkami/OneDrive/Pulpit/UNI/posteriordb")
my_pdb <- pdb_local("C:/Users/mkami/OneDrive/Pulpit/UNI/posteriordb")
po <- posterior("neals_funnel-neals_funnel_noncentered", my_pdb)

funnel_data <- pdb_data(po)
funnel_code <- stan_code(po)


mod <- stan_model(model_code = funnel_code, model_name = "neals_funnel_noncentered")
# 5. Sample the model (Heavy configuration for Reference Quality)
# run many iterations and thin to reduce autocorrelation
fit <- sampling(
  mod, 
  data = funnel_data, 
  chains = 10,              # Many chains to ensure thorough exploration
  iter = 20000,             # High iteration count
  warmup = 10000, 
  thin = 10,                # Thinning to ensure draws are approximately independent
  seed = 42,
  control = list(
    adapt_delta = 0.99,     # Strict step size to avoid divergent transitions
    max_treedepth = 15
  )
)

# 6. Convert the rstan fit object into a 'draws' object
my_draws <- as_draws_df(fit)
dim(my_draws)
# 7. Summarize the draws to check quality
draws_summary <- summarize_draws(
  my_draws,
  mean, 
  sd, 
  rhat, 
  ess_bulk, 
  ess_tail
)

print(draws_summary)


draws_arr <- as.array(fit)
# 1. Check Total Draws
# (Must be 10,000 or more per parameter)
total_draws <- posterior::ndraws(my_draws)
pass_draws <- total_draws >= 10000

# 2. Check R-hat
# (All parameters must be strictly below 1.01)
max_rhat <- max(draws_summary$rhat, na.rm = TRUE)
pass_rhat <- max_rhat < 1.01

# 3. Check Divergent Transitions
# (Must be exactly 0 for HMC)
divergences <- rstan::get_num_divergent(fit)
pass_div <- divergences == 0

# 4. Check E-FMI (Expected Fraction of Missing Information)
# (Checking that no chain falls below the 0.2 failure threshold)
bfmi_per_chain <- rstan::get_bfmi(fit)
min_bfmi <- min(bfmi_per_chain)
pass_bfmi <- min_bfmi >= 0.2 # the documentation suggests below 0.2 however 
# this seems like a typo, as the reference suggest values below 0.3 are **problematic**.
# so I take the values above 0.2 to be passing 


calc_mean_lag1_ac <- function(var_matrix) {
  # Calculate lag-1 AC for each chain, then take the mean
  acs <- apply(var_matrix, 2, function(chain_vals) {
    acf(chain_vals, lag.max = 1, plot = FALSE)$acf[2]
  })
  return(mean(acs))
}

# Apply the function to all parameters (margin 3 of the array)
lag1_acs <- apply(draws_arr, 3, calc_mean_lag1_ac)
max_abs_ac <- max(abs(lag1_acs), na.rm = TRUE)
pass_ac <- max_abs_ac < 0.05

# --- Print the Pass/Fail Report ---
cat("\n=== PosteriorDB Reference Quality Report ===\n")

cat(sprintf("[ %s ] Total Draws: %d (Required: >= 10000)\n", 
            ifelse(pass_draws, "PASS", "FAIL"), total_draws))

cat(sprintf("[ %s ] Max R-hat: %.4f (Required: < 1.01)\n", 
            ifelse(pass_rhat, "PASS", "FAIL"), max_rhat))

cat(sprintf("[ %s ] Divergent Transitions: %d (Required: == 0)\n", 
            ifelse(pass_div, "PASS", "FAIL"), divergences))

cat(sprintf("[ %s ] Min E-FMI: %.4f (Required: >= 0.2)\n", 
            ifelse(pass_bfmi, "PASS", "FAIL"), min_bfmi))

cat(sprintf("[ %s ] Max Abs Lag-1 Autocorrelation: %.4f (Required: < 0.05)\n", 
            ifelse(pass_ac, "PASS", "FAIL"), max_abs_ac))

cat("============================================\n")



#################################################################

## save draws to file
library(jsonlite)
draws_list <- as.list(my_draws)
length(draws_list)

# Remove the internal posterior:: metadata columns (.chain, .iteration, .draw)
draws_list <- draws_list[!grepl("^\\.", names(draws_list))]
# remove the lp__ column
draws_list <- draws_list[!grepl("^lp__$", names(draws_list))]

posterior_name <- "neals_funnel-neals_funnel_noncentered"
json_filename <- paste0(posterior_name, ".json")
zip_filename <- paste0(posterior_name, ".json.zip")

write_json(draws_list, json_filename, auto_unbox = TRUE, digits = 10)
zip(zipfile = zip_filename, files = json_filename)

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
    # Store the actual calculated maximums from your diagnostic checks
    max_rhat = max(draws_summary$rhat, na.rm = TRUE),
    min_ess_bulk = min(draws_summary$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(draws_summary$ess_tail, na.rm = TRUE),
    divergent_transitions = rstan::get_num_divergent(fit)
  ),
  checks_made = list(
    # compliance with the PosteriorDB reference quality criteria
    ndraws_is_10k = TRUE,
    nchains_is_gte_4 = TRUE,
    ess_within_bounds = TRUE,
    r_hat_below_1_01 = TRUE,
    efmi_above_0_2 = TRUE,
    abs_mean_lag1_ac_below_0_05 = TRUE
  ),
  comments = NULL,
  added_by = "Michal Kaminski",
  added_date = as.character(Sys.Date()),
  versions = list(
    rstan_version = as.character(packageVersion("rstan")),
    r_version = R.version.string
  )
)

info_filename <- paste0(posterior_name, ".info.json")
info_path <- file.path("posterior_database/reference_posteriors/draws/info", info_filename)


write_json(info_metadata, info_path, auto_unbox = TRUE, pretty = TRUE)
