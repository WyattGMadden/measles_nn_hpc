library(tidyverse)
library(patchwork)
library(kableExtra)
theme_set(theme_classic())
save_dir <- "~/resubmission_nn_temp/"


#filter dirs that include "_fit_info"
dirs <- list.files("../../../output/models/pinn_experiments/final_manchester_pinn_yearcutoff", full.names = TRUE)

fit_info <- lapply(grep("_fit_info", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat <- dat[, c("ode_loss", "I_loss", 
                                      "I_test_loss", 
                                      "vert", "amp1", "amp2")]
                       dat$vert <- unlist(dat$vert)
                       dat$amp1 <- unlist(dat$amp1)
                       dat$amp2 <- unlist(dat$amp2)
                       dat$model <- substr(x, 74, nchar(x) - 17)
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
                           dat$model <- substr(x, 74, nchar(x) - 17)
                           dat$iter <- 1:nrow(dat)
                           return(dat)
                       } else {
                           return(NULL)
                       }
                   }) |>
    (function(x) {Reduce(rbind, x)})()



test_preds <- lapply(grep("_test_predictions", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat$model <- substr(x, 74, nchar(x) - nchar("_test_predictions.parquet"))
                       return(dat)
                   }) |>
    (function(x) {Reduce(rbind, x)})() |>
    mutate(model = factor(model, 
                          levels = c("naivepinn", "tsirpinn"),
                          labels = c("Naive-PINN Model", "TSIR-PINN Model")))

train_preds <- lapply(grep("_train_predictions", dirs, value = TRUE),
                   function(x) {
                       dat <- arrow::read_parquet(x)
                       dat$model <- substr(x, 74, nchar(x) - nchar("_train_predictions.parquet"))
                       return(dat)
                   }) |>
    (function(x) {Reduce(rbind, x)})() |>
    mutate(model = factor(model, 
                          levels = c("naivepinn", "tsirpinn"),
                          labels = c("Naive-PINN Model", "TSIR-PINN Model")))


#read in obs data
train_fit <- arrow::read_parquet("../../../output/models/pinn_experiments/final_manchester_pinn_yearcutoff/naivepinn_train_predictions.parquet")






tf_I_p <- test_preds |>
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


fit_info_p2 <- fit_info |>
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

save_dir <- "~/resubmission_nn_temp/manchester_pinn"
test_preds |>
    group_by(model) |>
    summarize(mae_S = mean(abs(S_pred - S)),
              mae_I = mean(abs(I_pred - I)),
              cor_S = cor(S_pred, S),
              cor_I = cor(I_pred, I)) |>
    rename("Test $MAE(\\hat{S}, S)$" = mae_S,
           "Test $MAE(\\hat{I}, I)$" = mae_I,
           Model = model) |>
    kable(caption = "MAE by model on test set", 
          format = "latex", 
          escape = F) |>
    kable_styling(latex_options = c("hold_position")) |>
    writeLines(file.path(save_dir, "mae_test.txt"))
    

