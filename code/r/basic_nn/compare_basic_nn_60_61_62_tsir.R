library(tidyverse)
library(patchwork)
theme_set(theme_classic())
set.seed(42)
save_dir <- "../../../output/figures/"
read_dir <- "../../../data/"

#all original cases/birth/etc data from max's github
all_cities <- read_csv("../../../data/created/all_cases.csv")


#tsir data predictions
tsir_dat <- read_csv("../../../output/data/basic_nn_optimal/tsir_preds_processed.csv") |>
    mutate(tsir = ifelse(tsir < 0, 0, tsir)) |>
    filter(!is.na(time))

#neural net predictions - 60
nn_dat_60 <- read_csv("../../../output/data/basic_nn_optimal_60/basic_nn_preds.csv")


full_dat_temp_60 <- nn_dat_60[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    mutate(k = substr(k, nchar(k) - 1, nchar(k)),
           k = as.integer(k)) |>
    filter(train_test == "test") %>% 
#     left_join(nn_dat_no_susc[, c("time", "nn_no_susc", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(tsir_dat[, c("time", "tsir", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(all_cities, by = c("time", "city"))

full_dat_60 <- full_dat_temp_60 |>
    mutate(k = factor(k, levels = 1:52)) 

#neural net predictions - 61
nn_dat_61 <- read_csv("../../../output/data/basic_nn_optimal/basic_nn_preds.csv")


full_dat_temp_61 <- nn_dat_61[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    mutate(k = substr(k, nchar(k) - 1, nchar(k)),
           k = as.integer(k)) |>
    filter(train_test == "test") %>% 
#     left_join(nn_dat_no_susc[, c("time", "nn_no_susc", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(tsir_dat[, c("time", "tsir", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(all_cities, by = c("time", "city"))

full_dat_61 <- full_dat_temp_61 |>
    mutate(k = factor(k, levels = 1:52)) 

#neural net predictions - 60
nn_dat_62 <- read_csv("../../../output/data/basic_nn_optimal_62/basic_nn_preds.csv")


full_dat_temp_62 <- nn_dat_62[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    mutate(k = substr(k, nchar(k) - 1, nchar(k)),
           k = as.integer(k)) |>
    filter(train_test == "test") %>% 
    left_join(tsir_dat[, c("time", "tsir", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(all_cities, by = c("time", "city"))

full_dat_62 <- full_dat_temp_62 |>
    mutate(k = factor(k, levels = 1:52)) 

k_corrmse <- c(1, 4, 12, 20, 34, 52)
full_dat_60$train_test_cutoff <- "1960"
full_dat_61$train_test_cutoff <- "1961"
full_dat_62$train_test_cutoff <- "1962"

prmse <- full_dat_60 %>% 
    rbind(full_dat_61) %>%
    rbind(full_dat_62) %>%
    filter(k %in% k_corrmse) %>%
    group_by(city, k, train_test_cutoff) %>% 
    mutate(nn_standardized = (log(nn + 1) - mean(log(cases + 1))) / sd(log(cases + 1)), 
           tsir_standardized = (log(tsir + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           cases_standardized = (log(cases + 1) - mean(log(cases + 1))) / sd(log(cases + 1))) %>% 
    summarize(min_pop = unique(min_pop),
              tsir_rmse = sqrt(mean((cases_standardized - tsir_standardized) ^ 2, na.rm = T)), 
              nn_rmse = sqrt(mean((cases_standardized - nn_standardized) ^ 2, na.rm = T)),
              case_sum = sum(cases)) |>
    group_by(k, train_test_cutoff) |> 
    summarize(tsir_rmse = round(mean(tsir_rmse, na.rm = T), 4),
              nn_rmse = round(mean(nn_rmse, na.rm = T), 4))

write_csv(prmse, "../../../output/tables/test_train_cutoff_basic_tsir_rmse.csv")
