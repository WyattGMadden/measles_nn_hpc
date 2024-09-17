library(tidyverse)
library(patchwork)
theme_set(theme_classic())
set.seed(42)
save_dir <- "~/resubmission_nn_temp/"
read_dir <- "../../../data/"

#all original cases/birth/etc data from max's github
all_cities <- read_csv("../../../output/data/basic_nn/all_cases_from_max_gh.csv")

#tsir data predictions
tsir_dat <- read_csv("../../../output/data/basic_nn_yearcutoff/tsir_preds_processed.csv") |>
    mutate(tsir = ifelse(tsir < 0, 0, tsir)) |>
    filter(!is.na(time))

#neural net predictions
nn_dat <- read_csv("../../../output/data/basic_nn_yearcutoff_optimal/basic_nn_preds.csv")

#gbboost predictions
rf100_dat <- read_csv("../../../output/data/random_forest_100/random_forest_preds.csv")


full_dat_temp <- nn_dat[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    mutate(k = substr(k, nchar(k) - 1, nchar(k)),
           k = as.integer(k)) |>
    filter(train_test == "test") |>
#     left_join(nn_dat_no_susc[, c("time", "nn_no_susc", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(tsir_dat[, c("time", "tsir", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(all_cities, by = c("time", "city")) %>%
    left_join(rf100_dat[, c("time", "rf100", "city", "k")], by = c("time", "city", "k"))

full_dat <- full_dat_temp |>
    mutate(k = factor(k, levels = 1:52)) 


k_corrmse <- c(1, 4, 12, 20, 34, 52)


prmse <- full_dat %>% 
    filter(k %in% k_corrmse) %>%
    group_by(city, k) %>% 
    mutate(nn_standardized = (log(nn + 1) - mean(log(cases + 1))) / sd(log(cases + 1)), 
           rf100_standardized = (log(rf100 + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           tsir_standardized = (log(tsir + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           cases_standardized = (log(cases + 1) - mean(log(cases + 1))) / sd(log(cases + 1))) %>% 
    summarize(min_pop = unique(min_pop),
              tsir_rmse = sqrt(mean((cases_standardized - tsir_standardized) ^ 2, na.rm = T)), 
              rf100_rmse = sqrt(mean((cases_standardized - rf100_standardized) ^ 2, na.rm = T)), 
              nn_rmse = sqrt(mean((cases_standardized - nn_standardized) ^ 2, na.rm = T)),
              case_sum = sum(cases)) %>% 
    mutate(k = paste0("k = ", k), 
           k = factor(k, levels = paste0("k = ", k_corrmse))) %>% 
    ggplot() + 
    geom_point(aes(x = rf100_rmse, y = nn_rmse, colour = log(min_pop)), 
               size = 0.1) +
    geom_abline(color = "black") + 
    cividis::scale_color_cividis(direction = -1) +
    facet_wrap(~ k, ncol = 3) +
    labs(x = expression(RMSE[rf100boost]),
         y = expression(RMSE[SFNN]),
         colour = "Log(Population)") +
#    theme(axis.text.x = element_text(angle = -90, vjust = 0.5)) +
    scale_x_continuous(breaks = seq(0.0, 1.5, 0.5), limits = c(0, 2)) +
    scale_y_continuous(breaks = seq(0, 1.5, 0.5), limits = c(0, 2)) +
    theme(panel.border = element_rect(colour = "black", fill = NA, size = 1)) +
    theme(legend.position = "bottom")
ggplot2::ggsave(paste0(save_dir, "rmse_nn_rf100_facet.png"),
                prmse, width = 3 * 1.75, height = 4 * 1.75, dpi = 600)


#rf100 rmse by k
rmse_by_k_over_models <- full_dat |>
    group_by(k) |>
    summarize(rf100_rmse = sqrt(mean((log(rf100 + 1) - log(cases + 1)) ^ 2, na.rm = T)),
              nn_rmse = sqrt(mean((log(nn + 1) - log(cases + 1)) ^ 2, na.rm = T)),
              tsir_rmse = sqrt(mean((log(tsir + 1) - log(cases + 1)) ^ 2, na.rm = T)))

# save as csv
write_csv(rmse_by_k_over_models, paste0(save_dir, "rf100_rmse_by_k_over_models.csv"))

#like above but rmse gain
prmsegain <- full_dat %>% 
    filter(k %in% k_corrmse) %>%
    group_by(city, k) %>% 
    mutate(nn_standardized = (log(nn + 1) - mean(log(cases + 1))) / sd(log(cases + 1)), 
           tsir_standardized = (log(tsir + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           cases_standardized = (log(cases + 1) - mean(log(cases + 1))) / sd(log(cases + 1))) %>% 
    summarize(min_pop = unique(min_pop),
              tsir_rmse = sqrt(mean((cases_standardized - tsir_standardized) ^ 2, na.rm = T)), 
              nn_rmse = sqrt(mean((cases_standardized - nn_standardized) ^ 2, na.rm = T)),
              case_sum = sum(cases)) %>% 
    mutate(rmse_gain = tsir_rmse - nn_rmse,
           k = paste0("k = ", k), 
           k = factor(k, levels = paste0("k = ", k_corrmse))) %>% 
    ggplot(aes(x = log(min_pop), y = rmse_gain)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(shape = 21,
               size = 0.7, 
               alpha = 0.2, 
               fill = "black",
               colour = "transparent") +
    geom_smooth(colour = "blue3", linewidth = 0.5) +
    geom_abline(color = "black") + 
    cividis::scale_color_cividis(direction = -1) +
    facet_wrap(~ k, ncol = 3) +
    labs(x = "Log(Population)",
         y = expression(RMSE[TSIR] - RMSE[SFNN])) +
#    theme(axis.text.x = element_text(angle = -90, vjust = 0.5)) +
#    scale_x_continuous(breaks = seq(0.0, 1.5, 0.5), limits = c(0, 2)) +
    scale_y_continuous(breaks = seq(0, 1.5, 0.5), limits = c(-1, 1)) +
    theme(panel.border = element_rect(colour = "black", fill = NA, size = 1)) +
    theme(legend.position = "bottom")


figrmseall <- prmse / guide_area() / prmsegain +
    plot_layout(widths = c(1), 
                heights = c(3, 0.5, 3),
                guides = "collect") +
    plot_annotation(tag_levels = 'A')

scale_factor <- 2
ggsave(paste0(save_dir, "rf100_rmse_reg_and_gain_nn_tsir_k_facet.png"),
       figrmseall, 
       width = 3 * scale_factor,
       height = 4 * scale_factor,
       dpi = 600)




