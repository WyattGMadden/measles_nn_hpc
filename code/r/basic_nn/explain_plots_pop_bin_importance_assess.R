library(tidyverse)
theme_set(theme_bw())
set.seed(42)
theme_better <- function(){
    theme_classic() +
        theme(panel.border = element_rect(fill = NA, color = "black", size = 1))
}
theme_set(theme_better())
save_dir <- "../../../output/figures/"
read_dir <- "../../../data/"

#all original cases/birth/etc data from max's github
all_cities <- read_csv("../../../output/data/basic_nn/all_cases_from_max_gh.csv")

#tsir data predictions
tsir_dat <- read_csv("../../../output/data/basic_nn/tsir_preds_processed.csv")

#neural net predictions
nn_dat <- read_csv("../../../output/data/basic_nn_yearcutoff/basic_nn_preds.csv")

coords <- read_csv("https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/formatted/prevac/coordinates_urban.csv") |>
    t() 
latlon <- coords |>
    #as tibble but rownames to column, its a matrix
    as_tibble(rownames = "city", colnames = c("lat", "lon")) |>
    rename(lon = V1, lat = V2) |>
    slice(-1) |>
    mutate(lat = as.numeric(lat),
           lon = as.numeric(lon))




full_dat <- nn_dat[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    filter(train_test == "test") %>% 
#     left_join(nn_dat_no_susc[, c("time", "nn_no_susc", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(tsir_dat[, c("time", "tsir", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(all_cities, by = c("time", "city")) %>% 
    filter(!is.na(tsir),
           !is.na(nn)) %>% 
    mutate(k = factor(k, levels = 1:52))


nbc <- read_csv("../../../output/data/basic_nn/nearest_big_city.csv") |>
    select(city, nearest_big_city) %>% 
    distinct() 


####################
###explainability###
####################

read_dir <- "../../../output/data/basic_nn_yearcutoff/explain/"


read_in_cap <- function(x) {
    x %>%
        select(2:ncol(.)) %>% 
        rename_all(~ gsub("_lag_", "", .)) %>%
        rename_all(~ gsub("[[:digit:]]+", "", .)) %>%
        rename(cases_nc = cases_nc_) |>
        left_join(nbc,
                  by = c("city")) |>
        mutate(time_temp = round(time, 2) + 1900) |>
        left_join(all_cities[, c("city", "time", "min_pop")] |>
                  mutate(time = round(time, 2)) |>
                  rename(city_pop = min_pop),
              by = c("city", "time_temp" = "time")) |>
        select(-time_temp) |>
        rename(`Incidence Lags` = cases,
               Population = pop,
               Births = births,
               `Nearby City Incidence Lags` = cases_nc)
}

k <- "52"

svs_exp <- read_csv(paste0(read_dir, k, "_svs_explain_sep_high_pop_groups.csv")) |>
    read_in_cap() 


svs_p1_abs <- svs_exp %>% 
    pivot_longer(-c(city, nearest_big_city, time, city_pop)) %>%
    group_by(name) %>% 
    summarize(svs = mean(abs(value))) %>%
    ggplot(aes(x = svs, y = name)) + 
    geom_vline(aes(xintercept = 0), linetype = "dashed") + 
    geom_point(color = "red2", size = 5) +
    labs(x = "Mean Absolute SHAP Value",
         y = "Feature")


process <- function(x) {

    abs_bycity <- x %>%
        select(-time) %>% 
        group_by(city, nearest_big_city, city_pop) %>%
        summarize_all(~ mean(abs(.))) 


    mean_bycity <- x %>%
        select(-time) %>% 
        group_by(city, city_pop, nearest_big_city) |>
        summarize_all(~ mean(.))  


    abs_long <- abs_bycity |>
        pivot_longer(-c(city, nearest_big_city, city_pop)) |>
        group_by(city, nearest_big_city, city_pop) |>
        mutate(rel_value = value / sum(value)) |>
        ungroup()

    #compute relative importance within each obs
    rel_within_obs <- x |>
        pivot_longer(-c(time, city, nearest_big_city, city_pop)) |>
        group_by(time, city) |>
        mutate(rel_value = abs(value) / sum(abs(value))) |>
        group_by(city, nearest_big_city, city_pop, name) |>
        summarize(rel_value = mean(rel_value))

    rel_by_time <- x |>
        pivot_longer(-c(time, city, nearest_big_city, city_pop)) |>
        group_by(time, city) |>
        mutate(rel_value = abs(value) / sum(abs(value))) |>
        mutate(time = round(time, 1) %%1) |>
        group_by(time, city, nearest_big_city, city_pop, name) |>
        summarize(rel_value = mean(rel_value))

    return(list(abs_bycity = abs_bycity,
                mean_bycity = mean_bycity,
                abs_long = abs_long,
                rel_within_obs = rel_within_obs,
                rel_by_time = rel_by_time))
}


svs_processed <- process(svs_exp)
svs_abs_bycity <- svs_processed$abs_bycity
svs_mean_bycity <- svs_processed$mean_bycity
svs_abs_long <- svs_processed$abs_long
svs_rel_within_obs <- svs_processed$rel_within_obs
svs_rel_by_time <- svs_processed$rel_by_time

big_city_sep_high <- svs_rel_within_obs |>
    ungroup() |>
    #filter if name column contains "lag"
    filter(grepl("cases", name)) |>
    mutate(name = gsub("cases_", "", name),
           #capitalize first letter
           name = paste0(toupper(substr(name, 1, 1)), substr(name, 2, nchar(name))))
        
spherical_dist <- function(lat1, lon1, lat2, lon2, r = 3958.75) {
    lat1 <- lat1 * pi / 180
    lat2 <- lat2 * pi / 180
    lon1 <- lon1 * pi / 180
    lon2 <- lon2 * pi / 180
    cos_lat1 <- cos(lat1)
    cos_lat2 <- cos(lat2)
    cos_lat_d <- cos(lat1 - lat2)
    cos_lon_d <- cos(lon1 - lon2)
    return(r * acos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d)))
}

big_city_sep_high |>
    ggplot() +
    geom_point(aes(x = log(city_pop), y = rel_value)) +
    geom_smooth(aes(x = log(city_pop), y = rel_value), method = "loess") +
    #geom_point(aes(x = log(city_pop), y = rel_value, colour = log(distance), alpha = 0.3) +
    #geom_smooth(aes(x = log(city_pop), y = rel_value), method = "loess", span = 0.5) +
    facet_grid(. ~ name, scales = "free")

big_cities <- big_city_sep_high |>
    group_by(city) |>
    summarize(city_pop = max(city_pop)) |>
    arrange(desc(city_pop)) |>
    slice(1:4) |>
    pull(city)

big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    ggplot() +
    geom_point(aes(x = log(city_pop), y = rel_value), alpha = 0.2) +
    geom_smooth(aes(x = log(city_pop), y = rel_value), method = "loess") +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_loess.png"), width = 9, height = 5)


big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    mutate(pop_group = cut_number(log(city_pop), 5)) |>
    ggplot() +
    geom_boxplot(aes(x = pop_group, y = rel_value), outlier.size = .3) +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size (5-Quantile)")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_5quantile.png"), width = 9, height = 5)


big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    mutate(pop_group = cut_number(log(city_pop), 8)) |>
    ggplot() +
    geom_boxplot(aes(x = pop_group, y = rel_value), outlier.size = .3) +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size (8-Quantile)")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_8quantile.png"), width = 9, height = 5)

big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    mutate(pop_group = cut_number(log(city_pop), 10)) |>
    ggplot() +
    geom_boxplot(aes(x = pop_group, y = rel_value), outlier.size = .3) +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size (10-Quantile)")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_10quantile.png"), width = 9, height = 5)

big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    mutate(pop_group = cut_number(log(city_pop), 15)) |>
    ggplot() +
    geom_boxplot(aes(x = pop_group, y = rel_value), outlier.size = .3) +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 5)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size (15-Quantile)")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_15quantile.png"), width = 9, height = 5)

big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    mutate(pop_group = cut_number(log(city_pop), 20)) |>
    ggplot() +
    geom_boxplot(aes(x = pop_group, y = rel_value), outlier.size = .3) +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 5)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size (20-Quantile)")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_20quantile.png"), width = 9, height = 5)

big_city_sep_high |>
    filter(name %in% big_cities) |>
    #set city factor levels by population
    mutate(name = paste0(name, " Incidence Lags"),
           name = factor(name,
                         levels = paste0(big_cities, " Incidence Lags"))) |>
    group_by(name) |>
    mutate(rel_value = (rel_value - min(rel_value)) / (max(rel_value) - min(rel_value))) |>
#    mutate(pop_group = cut(log(city_pop), breaks = 10)) |>
    #breaks on quantile
    mutate(pop_group = cut_number(log(city_pop), 25)) |>
    ggplot() +
    geom_boxplot(aes(x = pop_group, y = rel_value), outlier.size = .3) +
    facet_wrap( ~ name, nrow = 1) +
    #vertical x 
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 5)) + 
    labs(y = "Relative Contribution of Core Cities \nto Local Transmission", 
         x = "Log Population Size (25-Quantile)")

    
ggsave(paste0("~/resubmission_nn_temp/", "k", k, "sep_high_pop_groups_over_pop_cut_25quantile.png"), width = 9, height = 5)
