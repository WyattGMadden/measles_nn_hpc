library(tidyverse)
theme_set(theme_bw())
set.seed(42)
theme_better <- function(){
    theme_classic() +
        theme(panel.border = element_rect(fill = NA, color = "black", size = 1))
}
theme_set(theme_better())
save_dir <- "../../../output/figures/"

all_cities <- read_csv("../../../data/created/all_cases.csv")

nbc <- read_csv("../../../data/created/nearest_big_city.csv") |>
    select(city, nearest_big_city) %>% 
    distinct() 


####################
###explainability###
####################

read_dir <- "../../../output/data/basic_nn_optimal/explain/"


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


process <- function(x) {
    #compute relative importance within each obs
    rel_within_obs <- x |>
        pivot_longer(-c(time, city, nearest_big_city, city_pop)) |>
        group_by(time, city) |>
        mutate(rel_value = abs(value) / sum(abs(value))) |>
        group_by(city, nearest_big_city, city_pop, name) |>
        summarize(rel_value = mean(rel_value))

    return(list(rel_within_obs = rel_within_obs))
}


svs_processed <- process(svs_exp)
svs_rel_within_obs <- svs_processed$rel_within_obs

big_city_sep_high <- svs_rel_within_obs |>
    ungroup() |>
    #filter if name column contains "lag"
    filter(grepl("cases", name)) |>
    mutate(name = gsub("cases_", "", name),
           #capitalize first letter
           name = paste0(toupper(substr(name, 1, 1)), substr(name, 2, nchar(name))))
        

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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_loess.png"), width = 9, height = 5)


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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_5quantile.png"), width = 9, height = 5)


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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_8quantile.png"), width = 9, height = 5)

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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_10quantile.png"), width = 9, height = 5)

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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_15quantile.png"), width = 9, height = 5)

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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_20quantile.png"), width = 9, height = 5)

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

    
ggsave(paste0(save_dir, "k", k, "sep_high_pop_groups_over_pop_cut_25quantile.png"), width = 9, height = 5)
