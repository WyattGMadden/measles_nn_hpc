library(tidyverse)


nn_trans_dat <- list.files("../../../output/models/basic_nn_bif", full.names = T) %>% 
    {.[substr(., nchar(.) - 16, nchar(.)) == "transform.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 37, nchar(x) - 18))) %>% 
    {Reduce(rbind, .)} %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    select(time, city, k, cases_mean, cases_std)
nn_dat <- list.files("../../../output/models/basic_nn_bif", full.names = T) %>% 
    {.[substr(., nchar(.) - 13, nchar(.)) == "output.parquet"]} %>% 
    lapply(function(x) arrow::read_parquet(x) %>% mutate(k = substr(x, 37, nchar(x) - 15))) %>% 
    {Reduce(rbind, .)} %>% 
    rename(nn = pred) %>% 
    mutate(time = as.double(paste0("19", time))) %>% 
    left_join(nn_trans_dat,
              by = c("time", "city", "k")) %>%
    mutate(nn_orig = nn,
           nn = exp(nn * cases_std + cases_mean) - 1,
           cases_orig = cases,
           cases = exp(cases * cases_std + cases_mean) - 1)


nn_dat |>
    filter(city == "London")


nn_trans_dat |>
    filter(city == "London")



write_csv(nn_dat, "../../../output/data/basic_nn_bif/basic_nn_bif_preds.csv")


