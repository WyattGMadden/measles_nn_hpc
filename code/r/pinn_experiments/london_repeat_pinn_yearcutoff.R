library(tidyverse)
library(patchwork)
library(kableExtra)
theme_set(theme_classic())
save_dir <- "../../../output/figures/"

dirs <- list.files("../../../output/models/pinn_experiments/london_repeat_pinn_yearcutoff", full.names = TRUE)
get_model_name <- function(text) {
  # Check if 'tsirpinn' is in the string
  if (grepl("tsirpinn", text)) {
    return("tsirpinn")
  } else if (grepl("naivepinn", text)) {
    # Check if 'naivepinn' is in the string
    return("naivepinn")
  } else {
    # Return 'neither' if none of the conditions are met
    return("neither")
  }
}
fit_info <- lapply(grep("_fit_info", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat <- dat[, c("ode_loss", "I_loss", 
                                      "I_test_loss", 
                                      "vert", "amp1", "amp2")]
                       dat$vert <- unlist(dat$vert)
                       dat$amp1 <- unlist(dat$amp1)
                       dat$amp2 <- unlist(dat$amp2)
                       dat$model <- get_model_name(x)
                       dat$city <- sub(".*city([^_]+)_.*", "\\1", x)
                       dat$k <- gsub(".*?_k([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$tlag <- gsub(".*?_tlag([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$run <- gsub("^.*?_run_([0-9]+)_.*$", "\\1", x)
                       dat$iter <- 1:nrow(dat)
                       return(dat)
                   }) |>
    (function(x) {Reduce(rbind, x)})() |>
    mutate(model = factor(model, 
                          levels = c("naivepinn", "tsirpinn"),
                          labels = c("Naive-PINN Model", "TSIR-PINN Model")))

S_params <- lapply(grep("_fit_info", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       #if there are columns that start with "S_"
                       if (any(grepl("^S_1", colnames(dat)))) {
                           dat <- dat[, names(dat)[grepl("^S_[0-9]+", colnames(dat))]]
                           dat$model <- get_model_name(x)
                           dat$k <- gsub(".*?_k([0-9]+)_.*", "\\1", x, perl = TRUE)
                           dat$city <- sub(".*city([^_]+)_.*", "\\1", x)
                           dat$tlag <- gsub(".*?_tlag([0-9]+)_.*", "\\1", x, perl = TRUE)
                           dat$run <- gsub("^.*?_run_([0-9]+)_.*$", "\\1", x)
                           dat$iter <- 1:nrow(dat)
                           return(dat)
                       } else {
                           return(NULL)
                       }
                   }) |>
    #remove NULLs
    (function(x) {Reduce(rbind, x)})()


# tsir data
tsir_preds <- read_csv("../../../output/data/basic_nn_optimal/tsir_preds_processed.csv")


test_preds <- lapply(grep("_test_predictions", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat$model <- get_model_name(x)
                       dat$city <- sub(".*city([^_]+)_.*", "\\1", x)
                       dat$k <- gsub(".*?_k([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$tlag <- gsub(".*?_tlag([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$run <- gsub("^.*?_run_([0-9]+)_.*$", "\\1", x)
                       return(dat)
                   }) |>
    (function(x) {Reduce(rbind, x)})() |>
    mutate(time_original = time_original + 1900) |>
    mutate(model = factor(model, 
                          levels = c("naivepinn", "tsirpinn"),
                          labels = c("Naive-PINN Model", "TSIR-PINN Model")))

train_preds <- lapply(grep("_train_predictions", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat$model <- get_model_name(x)
                       dat$city <- sub(".*city([^_]+)_.*", "\\1", x)
                       dat$k <- gsub(".*?_k([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$tlag <- gsub(".*?_tlag([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$run <- gsub("^.*?_run_([0-9]+)_.*$", "\\1", x)
                       return(dat)
                   }) |>
    (function(x) {Reduce(rbind, x)})() |>
    mutate(model = factor(model, 
                          levels = c("naivepinn", "tsirpinn"),
                          labels = c("Naive-PINN Model", "TSIR-PINN Model")))



tf_I_p_all_dat <- test_preds |>
    filter(k == "52",
           tlag == "104") |>
    filter(city == "London") |>
    mutate("Predicted Incidence" = I_pred,
           "Observed Incidence" = I,
           time = (time - 1) / 26 + 1951) 
tf_I_p_all <- tf_I_p_all_dat |>
    ggplot(aes(x = time)) +
    geom_line(aes(y = `Observed Incidence`, group = run), color = "black") +
    geom_line(aes(y = `Predicted Incidence`, group = run), color = "blue3", alpha = 0.05) +
    geom_line(data = (tf_I_p_all_dat |>
                group_by(time, model) |>
                summarize(mean_pred = mean(`Predicted Incidence`))),
                aes(y = mean_pred), 
                color = "blue3", 
                linetype = "dashed") +
    facet_grid(model~., labeller = labeller(model = label_value)) +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Year",
         y = "Incidence",
         color = "",
         linetype = "")
tf_I_p_all
ggplot2::ggsave(paste0(save_dir, "pinn_ab_test_pred_all_runs.png"),
                tf_I_p_all,
                width = 8,
                height = 6,
                dpi = 600)

tf_I_p <- test_preds |>
    filter(run == min(run)) |>
    pivot_longer(cols = c("I_pred", "I"),
                 names_to = "pred_actual",
                 values_to = "I") |>
    mutate(time = (time - 1) / 26 + 1951,
           pred_actual = case_when(pred_actual == "I_pred" ~ "Predicted Incidence", 
                                   pred_actual == "I" ~ "Observed Incidence")) |>
    ggplot(aes(x = time, y = I, color = pred_actual, linetype = pred_actual)) +
    geom_line() +
    facet_wrap(~model, labeller = labeller(model = label_value), ncol = 1) +
    scale_color_manual(values = c("Observed Incidence" = "black", 
                                  "Predicted Incidence" = "blue3", 
                                  "TSIR" = "yellow4")) +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Year",
         y = "Incidence",
         color = "",
         linetype = "")
tf_I_p
ggplot2::ggsave(paste0(save_dir, "pinn_ab_test_pred.png"),
                tf_I_p,
                width = 8,
                height = 6,
                dpi = 600)

fit_info_p2 <- fit_info |>
    filter(run == min(run)) |>
    pivot_longer(cols = c("vert", "amp1", "amp2"),
                 names_to = "param",
                 values_to = "value") |>
    ggplot(aes(x = iter, y = value, color = param, linetype = param)) +
    geom_line() +
    scale_color_manual(values = c("vert" = "black", 
                                  "amp1" = "blue3", 
                                  "amp2" = "yellow4"), 
                       labels = c("vert" = expression(nu), 
                                  "amp1" = expression(alpha[phantom(0) * 1]), 
                                  "amp2" = expression(alpha[phantom(0) * 2]))) +
    scale_linetype_manual(values = c("vert" = "solid", 
                                  "amp1" = "dashed", 
                                  "amp2" = "dotted"), 
                       labels = c("vert" = expression(nu), 
                                  "amp1" = expression(alpha[phantom(0) * 1]), 
                                  "amp2" = expression(alpha[phantom(0) * 2]))) +
    facet_wrap(~model, labeller = labeller(model = label_value), ncol = 1) +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Epoch",
         y = "Parameter Value",
         color = "",
         linetype = "")
fig2 <- tf_I_p / fit_info_p2 +
    plot_layout(widths = c(1), 
                heights = c(1, 1),
                ncol = 1, 
                nrow = 2) +
    plot_annotation(tag_levels = 'A')

scale_factor <- 3
ggsave(file.path(save_dir, "pinn_ab_test_pred_seasonal_param_fit.png"),
       fig2, 
       width = 2.5 * scale_factor,
       height = 3 * scale_factor,
       dpi = 600)

run_losses <- fit_info |>
    filter(k == "52") |>
#    filter(tlag == "130") |>
    filter(city == "London") |>
#    filter(tlag == "26") |>
    # normalize losses
    ggplot(aes(x = iter, y = I_test_loss, color = run)) +
    geom_line() +
    facet_grid(tlag~model, labeller = labeller(model = label_value)) +
    geom_hline(aes(yintercept = 457), linetype = "dashed") +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Epoch",
         y = "Test I MAE",
         color = "Run")



#mse by model
#output as latex table
#write to txt file

save_dir <- "../../../output/tables"
test_preds |>
    filter(k == "52") |>
    filter(city == "London") |>
    filter(tlag == "104") |>
    select(I_pred, I, model, run) |>
    group_by(model, run) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    group_by(model) |>
    summarize(mean_mae_I = round(mean(mae_I), 2),
              mean_cor_I = round(mean(cor_I), 2)) |>
    rename("Test $\\widehat{MAE(\\hat{I}, I)}$" = mean_mae_I,
           "Test $\\widehat{Cor(\\hat{I}, I)}$" = mean_cor_I,
           Model = model) |>
    kable(caption = "MAE by model on test set", 
          format = "latex", 
          escape = F) |>
    kable_styling(latex_options = c("hold_position")) |>
    writeLines(file.path(save_dir, "mae_test.txt"))
 
tsir_summ <- tsir_preds |>
    filter(city == "London",
           time >= 1961) |>
    mutate(k = as.character(k)) |>
    left_join(test_preds[, c("city", "time_original", "k", "I")], 
              by = c("city" = "city", "time" = "time_original", "k" = "k")) |>
    mutate(I_pred = tsir,
           model = "TSIR Model") |>
    select(time, k, I_pred, city, I, model) |>
    filter(k == "52") |>
    group_by(model) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    filter(!is.na(mae_I))

test_preds |>
    filter(k == "34") |>
    select(time, k, I_pred, city, I, model) |>
    rbind(tsir_in_test) |>
    group_by(model) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    rename("Test $MAE(\\hat{I}, I)$" = mae_I,
           Model = model) |>
    kable(caption = "MAE by model on test set", 
          format = "latex", 
          escape = F) |>
    kable_styling(latex_options = c("hold_position")) |>
    writeLines(file.path(save_dir, "mae_test.txt"))
    
tsir_in_test |>
    filter(k == "52") |>
    filter(city == "London") |>
    group_by(model, k, city) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    filter(k == "52") |>
    write_csv("~/resubmission_nn_temp/tsir_test_mae_cor_k52_london.csv")

temp <- test_preds |>
    group_by(model, run) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    group_by(model) |>
    summarize(mean_mae_I = mean(mae_I),
              sd_mae_I = sd(mae_I),
              mean_cor_I = mean(cor_I),
              sd_cor_I = sd(cor_I))
temp |>
    write_csv("~/resubmission_nn_temp/pinn_test_mean_mae_cor_by_model_k52_london.csv")

