library(tidyverse)
library(patchwork)
library(kableExtra)
theme_set(theme_classic())
save_dir <- "../../../output/figures"

dirs <- list.files("../../../output/models/pinn_experiments/final_onecity_pinn_yearcutoff", full.names = TRUE)
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
                           dat$iter <- 1:nrow(dat)
                           return(dat)
                       } else {
                           return(NULL)
                       }
                   }) |>
    #remove NULLs
    (function(x) {Reduce(rbind, x)})()


# tsir data
tsir_preds <- read_csv("../../../output/data/basic_nn_yearcutoff_optimal/tsir_preds_processed.csv")


test_temp <- arrow::read_parquet("../../../output/models/pinn_experiments/final_onecity_pinn_yearcutoff/tsirpinn_k52_tlag130_cityLondon_test_predictions.parquet")
test_preds <- lapply(grep("_test_predictions", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat$model <- get_model_name(x)
                       dat$city <- sub(".*city([^_]+)_.*", "\\1", x)
                       dat$k <- gsub(".*?_k([0-9]+)_.*", "\\1", x, perl = TRUE)
                       dat$tlag <- gsub(".*?_tlag([0-9]+)_.*", "\\1", x, perl = TRUE)
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
                       return(dat)
                   }) |>
    (function(x) {Reduce(rbind, x)})() |>
    mutate(model = factor(model, 
                          levels = c("naivepinn", "tsirpinn"),
                          labels = c("Naive-PINN Model", "TSIR-PINN Model")))


#read in obs data
train_fit <- arrow::read_parquet("../../../output/models/pinn_experiments/final_london_pinn_yearcutoff/naivepinn_train_predictions.parquet")

unique(test_preds$tlag)
unique(test_preds$k)

tf_I_p_train <- train_preds |>
    filter(k == "52") |>
    filter(city == "Birmingham") |>
    pivot_longer(cols = c("I_pred", "I"),
                 names_to = "pred_actual",
                 values_to = "I") |>
    mutate(time = (time - 1) / 26 + 1951,
           pred_actual = case_when(pred_actual == "I_pred" ~ "Predicted Incidence", 
                                   pred_actual == "I" ~ "Observed Incidence")) |>
    ggplot(aes(x = time, y = I, color = pred_actual, linetype = pred_actual)) +
    geom_line() +
    facet_wrap(tlag~model, labeller = labeller(model = label_value), ncol = 1) +
    scale_color_manual(values = c("Observed Incidence" = "black", 
                                  "Predicted Incidence" = "blue3", 
                                  "TSIR" = "yellow4")) +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Year",
         y = "Incidence",
         color = "",
         linetype = "")
tf_I_p_train
tf_I_p <- test_preds |>
    filter(k == "52") |>
#    filter(tlag == "78") |>
    filter(city == "London") |>
    pivot_longer(cols = c("I_pred", "I"),
                 names_to = "pred_actual",
                 values_to = "I") |>
    mutate(time = (time - 1) / 26 + 1951,
           pred_actual = case_when(pred_actual == "I_pred" ~ "Predicted Incidence", 
                                   pred_actual == "I" ~ "Observed Incidence")) |>
    ggplot(aes(x = time, y = I, color = pred_actual, linetype = pred_actual)) +
    geom_line() +
    facet_wrap(tlag~model, labeller = labeller(model = label_value), ncol = 1) +
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

unique(test_preds$k)
fit_info_p2 <- fit_info |>
    filter(k == "52") |>
#    filter(tlag == "130") |>
    filter(city == "Liverpool") |>
#    filter(tlag == "26") |>
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
    facet_grid(tlag~model, labeller = labeller(model = label_value)) +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Epoch",
         y = "Parameter Value",
         color = "",
         linetype = "")

unique(test_preds$k)
losses <- fit_info |>
    filter(k == "52") |>
#    filter(tlag == "130") |>
    filter(city == "Liverpool") |>
#    filter(tlag == "26") |>
    # normalize losses
    mutate(ode_loss = ode_loss / max(ode_loss),
           I_loss = I_loss / max(I_loss),
           I_test_loss = I_test_loss / max(I_test_loss)) |>
    pivot_longer(cols = c("I_loss", "I_test_loss", "ode_loss"),
                 names_to = "loss_type",
                 values_to = "loss") |>
    ggplot(aes(x = iter, y = loss, color = loss_type, linetype = loss_type)) +
    geom_line() +
    facet_grid(tlag~model, labeller = labeller(model = label_value)) +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom") +
    labs(x = "Epoch",
         y = "Parameter Value",
         color = "",
         linetype = "")
#save losses fig
losses
ggplot2::ggsave("~/resubmission_nn_temp/liverpool_losses.png",
                losses,
                width = 6,
                height = 8,
                dpi = 600)

#AB test-pred/seasonal param fit plots


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


#mse by model
#output as latex table
#write to txt file

save_dir <- "../../../output/tables"
test_preds |>
    select(time, k, I_pred, city, I, model, tlag) |>
    group_by(model, tlag, k) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    rename("Test $MAE(\\hat{I}, I)$" = mae_I,
           Model = model) |>
    kable(caption = "MAE by model on test set", 
          format = "latex", 
          escape = F) |>
    kable_styling(latex_options = c("hold_position")) |>
    writeLines(file.path(save_dir, "mae_test.txt"))
 
test_preds
temp$I
tsir_in_test <- tsir_preds |>
    filter(city %in% c("London", "Manchester", "Birmingham", "Liverpool"),
           time >= 1961) |>
    mutate(k = as.character(k)) |>
    left_join(test_preds[, c("city", "time_original", "k", "I")], 
              by = c("city" = "city", "time" = "time_original", "k" = "k")) |>
    mutate(I_pred = tsir,
           model = "TSIR Model") |>
    select(time, k, I_pred, city, I, model)

unique(test_preds$city)
k_temp <- "52"
test_mae_temp <- test_preds |>
    filter(k == k_temp) |>
    select(time, k, I_pred, city, I, model, tlag) |>
    group_by(city, tlag, model) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) 
print(test_mae_temp, n=100)
test_mae_temp |>
    write_csv("~/resubmission_nn_temp/pinn_test_mae_cor_by_city_tlag.csv")

temp_tsir <- tsir_in_test |>
    filter(k == k_temp) |>
    group_by(model, k, city) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    filter(!is.na(mae_I))
print(temp_tsir, n=100)

temp_tsir |>
    write_csv("~/resubmission_nn_temp/tsir_test_mae_cor_by_city_tlag.csv")

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
    group_by(model, k, city) |>
    summarize(mae_I = mean(abs(I_pred - I)),
              cor_I = cor(I_pred, I)) |>
    filter(k == "52")

temp <- test_preds |>
    group_by(city, tlag, k, model) |>
    summarize(mae_S = mean(abs(S_pred - S)),
              mae_I = mean(abs(I_pred - I)),
              cor_S = cor(S_pred, S),
              cor_I = cor(I_pred, I)) |>
    mutate(tlag = as.numeric(tlag),
           k = as.numeric(k))
print(temp, n=100)
temp |>
    filter(city == "London") |>
    filter(k == 52) |>
    filter(tlag == 130) |>
    ggplot(aes(x = model, y = mae_I, group = tlag)) +
    facet_grid(k ~ .) +
    geom_line()

