check_total_draws <- function(draws, threshold = 10000) {
    total_draws <- posterior::ndraws(draws)
    pass_draws <- total_draws >= threshold
    return(list(total_draws = total_draws, pass_draws = pass_draws))
}

check_divergences <- function(fit) {
    divergences <- rstan::get_num_divergent(fit)
    pass_div <- divergences == 0
    return(list(divergences = divergences, pass_div = pass_div))
}

check_rhat <- function(draws_summary, threshold = 1.01) {
    max_rhat <- max(draws_summary$rhat, na.rm = TRUE)
    pass_rhat <- max_rhat < threshold
    return(list(max_rhat = max_rhat, pass_rhat = pass_rhat))
}

check_bfmi <- function(fit, threshold = 0.2) {
    bfmi_per_chain <- rstan::get_bfmi(fit)
    min_bfmi <- min(bfmi_per_chain)
    pass_bfmi <- min_bfmi >= threshold
    return(list(min_bfmi = min_bfmi, pass_bfmi = pass_bfmi))
}

calc_mean_lag1_ac <- function(var_matrix) {
    # Calculate lag-1 AC for each chain, then take the mean
    acs <- apply(var_matrix, 2, function(chain_vals) {
        acf(chain_vals, lag.max = 1, plot = FALSE)$acf[2]
    })
    return(mean(acs))
}

check_lag1_ac <- function(draws_arr, threshold = 0.05) {
    lag1_acs <- apply(draws_arr, 3, calc_mean_lag1_ac)
    max_abs_ac <- max(abs(lag1_acs), na.rm = TRUE)
    pass_ac <- max_abs_ac < threshold
    return(list(max_abs_ac = max_abs_ac, pass_ac = pass_ac))
}

#### UTILITY FUNCTION FOR CHECKING POSTERIOR QUALITY
#### THROWN IN YOUR FIT OBJECT AND IT WILL RETURN A LIST OF 
#### CHECK FOR POSTERIORDB REFERENCE DRAWS QUALITY CONTROL
#### EXAMPLE BELOW

check_posterior_quality <- function(fit, verbose = TRUE) {
    draws <- posterior::as_draws_df(fit)
    draws_summary <- posterior::summarize_draws(
        draws,
        mean, 
        sd, 
        rhat, 
        ess_bulk, 
        ess_tail
    )
    draws_arr <- as.array(fit)
    results <- list()

    # Check Total Draws
    results$total_draws <- check_total_draws(draws)

    # Check R-hat
    results$rhat <- check_rhat(draws_summary)

    # Check Divergent Transitions
    results$divergences <- check_divergences(fit)

    # Check E-FMI
    results$bfmi <- check_bfmi(fit)

    # Check Lag-1 Autocorrelation
    results$lag1_ac <- check_lag1_ac(draws_arr)

    if (verbose) {
        cat("\n=== PosteriorDB Reference Quality Report ===\n")

        cat(sprintf(
            "[ %s ] Total Draws: %d (Required: >= 10000)\n",
            ifelse(results$total_draws$pass_draws, "PASS", "FAIL"), results$total_draws$total_draws
        ))

        cat(sprintf(
            "[ %s ] Max R-hat: %.4f (Required: < 1.01)\n",
            ifelse(results$rhat$pass_rhat, "PASS", "FAIL"), results$rhat$max_rhat
        ))

        cat(sprintf(
            "[ %s ] Divergent Transitions: %d (Required: == 0)\n",
            ifelse(results$divergences$pass_div, "PASS", "FAIL"), results$divergences$divergences
        ))

        cat(sprintf(
            "[ %s ] Min E-FMI: %.4f (Required: >= 0.2)\n",
            ifelse(results$bfmi$pass_bfmi, "PASS", "FAIL"), results$bfmi$min_bfmi
        ))

        cat(sprintf(
            "[ %s ] Max Abs Lag-1 Autocorrelation: %.4f (Required: < 0.05)\n",
            ifelse(results$lag1_ac$pass_ac, "PASS", "FAIL"), results$lag1_ac$max_abs_ac
        ))

        cat("============================================\n")
    }

    return(results)
}


library(posterior)
library(rstan)
library(posteriordb)
options(mc.cores = parallel::detectCores())


my_pdb <- pdb_local(".")
po <- posterior("banana_2d_strong-banana", my_pdb)

data <- pdb_data(po)
code <- stan_code(po)
print(code)
print(data)

# load from stan file 

mod <- stan_model(model_code = code, model_name = "banana_2d_strong")

fit <- sampling(
  mod, 
  data = data, 
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

results <- check_posterior_quality(fit, verbose = TRUE)
# autocorrelation fails for 2d_moderate 
# divergence and autocorrelation fails for 2d_strong
# 8d strong passes all checks


#### true posterior draws for banana distribution (8d, b=0.1)
N <- 10000
v <- 100.0
b <- 0.03

x_draws <- rnorm(N, mean = 0, sd = sqrt(v))
y_draws <- rnorm(N, mean = -b * (x_draws^2 - v), sd = 1)

# Generate a matrix of 6 independent N(0,1) vectors
# y_rest_matrix <- matrix(rnorm(N * 6, mean = 0, sd = 1), nrow = N, ncol = 6)

# Combine into a dataframe
perfect_draws <- data.frame(
  x = x_draws,
  y = y_draws
)
# Add the y_rest columns dynamically (y_rest.1, y_rest.2, etc.)
# for (i in 1:6) {
#   perfect_draws_8d[[paste0("y_rest.", i)]] <- y_rest_matrix[, i]
# }

library(ggplot2)

# Assuming 'perfect_draws' is the dataframe from the analytical script
ggplot(perfect_draws, aes(x = x, y = y)) +
  # Use high transparency (alpha) to show density in the center vs the tails
  geom_point(alpha = 0.1, color = "darkblue", size = 0.5) +
  # Overlay density contours to highlight the geometry
  geom_density_2d(color = "orange", linewidth = 0.5) +
  theme_minimal() +
  labs(
    title = "Analytical Reference Draws: 2D Banana Target",
    subtitle = "Notice the extreme scale difference between the x and y axes",
    x = "x (Base Dimension: wide variance)",
    y = "y (Twisted Dimension: tightly constrained)"
  ) 
  # Force the axes to show the true scale disparity

# save the plot to a file
ggsave("banana_2d_03_reference_draws.png", width = 8, height = 6, dpi = 300)
