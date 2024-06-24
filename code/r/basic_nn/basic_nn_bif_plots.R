library(tidyverse)
theme_set(theme_classic())
set.seed(42)
save_dir <- "../../../output/figures/"
read_dir <- "../../../data/"

#all original cases/birth/etc data from max's github
all_cities <- read_csv("../../../output/data/basic_nn/all_cases_from_max_gh.csv")

#neural net predictions
nn_dat <- read_csv("../../../output/data/basic_nn_bif/basic_nn_bif_preds.csv")

full_dat_temp <- nn_dat[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    left_join(all_cities, by = c("time", "city"))

full_dat <- full_dat_temp |>
    filter(!is.na(nn)) |> 
    mutate(k = factor(k, levels = 1:52))  |>
    filter(floor(time) < 1963)






pfacet <- full_dat %>% 
    filter(city == "London",
           k %in% 1:4) %>%
    mutate(city_pop = factor(paste0(city, " - ", round(min_pop / 1000), "k")),
           city_pop = reorder(city_pop, min_pop),
           city_pop = reorder(city_pop, -as.numeric(city_pop))) %>%
    pivot_longer(c("cases", "nn"),
                 names_to = "model",
                 values_to = "cases") %>%
    mutate(model = case_when(model == "cases" ~ "Observed Incidence",
                             model == "nn" ~ "Predicted Incidence"),
           train_test = case_when(train_test == "train" ~ "Train",
                                  train_test == "test" ~ "Test"),
           k = paste0(k, "-Step-Ahead Forecast")) %>%
    ggplot(aes(x = time)) + 
    geom_line(aes(y = cases, color = model, linetype = train_test), 
              alpha = 1) +
    facet_wrap(. ~ k,
               ncol = 2,
               scales = "free") + 
    scale_color_manual(values = c("Observed Incidence" = "black", 
                                  "Predicted Incidence" = "blue")) +
    theme(legend.position = "bottom") +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    labs(x = "Time",
         y = "Incidence",
         color = "",
         linetype = "")

ggsave(paste0(save_dir, "nn_compare_london_k1to4", k_temp, "_nonlog.png"), 
       pfacet, width = 8, height = 4)



