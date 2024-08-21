library(tidyverse)


xg_trans_dat <- list.files("../../../output/models/xgboost_100", full.names = T) %>% 
    {.[substr(., nchar(.) - 16, nchar(.)) == "transform.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 36, nchar(x) - 18))) %>% 
    #only 50
    {Reduce(rbind, .)} %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    select(time, city, k, cases_mean, cases_std)
xg_dat <- list.files("../../../output/models/xgboost_100", full.names = T) %>% 
    {.[substr(., nchar(.) - 13, nchar(.)) == "output.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 36, nchar(x) - 15))) %>% 
    {Reduce(rbind, .)} %>% 
    rename(xg = pred) %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    left_join(xg_trans_dat,
              by = c("time", "city", "k")) %>%
    mutate(xg_orig = xg,
           xg = exp(xg * cases_std + cases_mean) - 1)




write_csv(xg_dat, "../../../output/data/xgboost_100/xg_preds.csv")


