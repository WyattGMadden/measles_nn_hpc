library(tidyverse)

births <- read_csv("https://raw.githubusercontent.com/msylau/measles_LASSO/main/data//E%26W/births_urban.csv") %>% 
    rename("year" = X) %>% 
    pivot_longer(2:ncol(.), names_to = "city", values_to = "births") %>%
    mutate(births = births / 26)



inf_pop_urb <- read_csv("https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/formatted/prevac/inferred_pop_urban.csv") %>% 
    rename("time" = `...1`) %>% 
    pivot_longer(2:ncol(.), names_to = "city", values_to = "pop")

cases <- read_csv("https://raw.githubusercontent.com/msylau/measles_LASSO/main/data//E%26W/cases_urban.csv") %>%
    rename("time" = `...1`) %>% 
    pivot_longer(2:ncol(.), names_to = "city", values_to = "cases") %>% 
    mutate(year = floor(time))

coords <- t(read_csv("https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/formatted/prevac/coordinates_urban.csv")) 
colnames(coords) <- coords[1, ]
coords <- coords[2:nrow(coords), ] %>% 
    as_tibble() %>% 
    rename_all(tolower) %>% 
    mutate_all(as.double) %>% 
    mutate(city = rownames(coords)[2:nrow(coords)])  


all_cities <- cases %>% 
    left_join(births, by = c("year", "city")) %>% 
    left_join(inf_pop_urb, by = c("time", "city")) %>% 
    left_join(coords, by = c("city")) %>% 
    mutate(time = 1900 + time) %>% 
    select(-year) %>% 
    group_by(city) %>% 
    mutate(min_pop = min(pop)) %>% 
    ungroup()

write.csv(all_cities, 
          "../../../output/data/basic_nn_optimal/all_cases_from_max_gh.csv", 
          row.names = F)

