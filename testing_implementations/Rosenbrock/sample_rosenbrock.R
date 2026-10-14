# library(ggplot2)
# library(MASS)

# N <- 100000     
# a <- 1/20         # a = 0.05
# b <- 100/20       # b = 5.0
# mu <- 1.0

# # x1 ~ N(mu, 1 / (2*a))
# x1 <- rnorm(N, mean = mu, sd = sqrt(1 / (2 * a)))
# # x2 | x1 ~ N(x1^2, 1 / (2*b))
# x2 <- rnorm(N, mean = x1^2, sd = sqrt(1 / (2 * b)))

# df <- data.frame(x1 = x1, x2 = x2)

# ggplot(df, aes(x = x1, y = x2)) +
#   geom_density_2d_filled(bins = 10, alpha = 0.8) +
#   geom_density_2d(color = "white", size = 0.2) +
#   coord_cartesian(xlim = c(-15, 15), ylim = c(-20, 100)) +
#   labs(
#     title = "2D Rosenbrock Distribution (Direct Sampling)",
#     subtitle = paste0("a = ", a, ", b = ", b, ", mu = ", mu),
#     x = expression(x[1]),
#     y = expression(x[2])
#   ) +
#   theme_minimal() +
#   theme(legend.position = "right")


library(rstan)
options(mc.cores = parallel::detectCores())



setwd("C:\\Users\\mkami\\OneDrive\\Pulpit\\UNI\\posteriordb")
code_path <- 'testing_implementations\\Rosenbrock\\rosenbrock.stan'
rosenbrock_2d <- list(
  n1 = 2,
  n2 = 1,
  mu = 1.0,
  a = 0.05,
  b = 5.0
)


fit <- stan(code_path, data = rosenbrock_2d, 
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
draws <- as.data.frame(fit)


save_path <- 'testing_implementations\\Rosenbrock\\Rdraws'

write.csv(draws, file = paste0(save_path, "\\draws_rosenbrock.csv"), row.names = FALSE)

# Generate fixed evaluation points for implementation check

set.seed(23)
eval_indices <- sample(1:nrow(draws), 1000, replace = FALSE)

eval_points <- eval_points[, !grepl("^lp__$", colnames(eval_points))]


# extract log-densities and gradients for those points
log_probs <- numeric(1000)

grads <- matrix(NA, nrow=1000, ncol=ncol(eval_points))
colnames(grads) <- colnames(eval_points)

for (i in 1:1000) {
  pt <- as.numeric(eval_points[i, ])
  log_probs[i] <- log_prob(fit, pt)
  grads[i, ] <- grad_log_prob(fit, pt)
}

write.csv(eval_points, file.path(save_path, "eval_points_rosenbrock.csv"), row.names = FALSE)
write.csv(log_probs, file.path(save_path, "log_probs_rosenbrock.csv"), row.names = FALSE)
write.csv(grads, file.path(save_path, "grads_rosenbrock.csv"), row.names = FALSE)
