library(tidyverse)


rf100_trans_dat <- list.files("../../../output/models/random_forest_100", full.names = T) %>% 
    {.[substr(., nchar(.) - 16, nchar(.)) == "transform.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 42, nchar(x) - 18))) %>% 
    #only 50
    {Reduce(rbind, .)} %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    select(time, city, k, cases_mean, cases_std)
rf100_dat <- list.files("../../../output/models/random_forest_100", full.names = T) %>% 
    {.[substr(., nchar(.) - 13, nchar(.)) == "output.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 42, nchar(x) - 15))) %>% 
    {Reduce(rbind, .)} %>% 
    rename(rf100 = pred) %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    left_join(rf100_trans_dat,
              by = c("time", "city", "k")) %>%
    mutate(rf100_orig = rf100,
           rf100 = exp(rf100 * cases_std + cases_mean) - 1)




write_csv(rf100_dat, "../../../output/data/random_forest_100/random_forest_preds.csv")


