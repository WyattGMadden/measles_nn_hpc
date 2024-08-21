library(tidyverse)


gb_trans_dat <- list.files("../../../output/models/gradientboost", full.names = T) %>% 
    {.[substr(., nchar(.) - 16, nchar(.)) == "transform.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 38, nchar(x) - 18))) %>% 
    #only 50
    {Reduce(rbind, .)} %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    select(time, city, k, cases_mean, cases_std)
gb_dat <- list.files("../../../output/models/gradientboost", full.names = T) %>% 
    {.[substr(., nchar(.) - 13, nchar(.)) == "output.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 38, nchar(x) - 15))) %>% 
    {Reduce(rbind, .)} %>% 
    rename(gb = pred) %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    left_join(gb_trans_dat,
              by = c("time", "city", "k")) %>%
    mutate(gb_orig = gb,
           gb = exp(gb * cases_std + cases_mean) - 1)




write_csv(gb_dat, "../../../output/data/gradientboost/gradientboost_preds.csv")


