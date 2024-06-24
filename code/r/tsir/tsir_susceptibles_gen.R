library(tsiR)
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

all_cities <- cases %>% 
    left_join(births, by = c("year", "city")) %>% 
    left_join(inf_pop_urb, by = c("time", "city")) %>% 
    mutate(time = 1900 + time) %>% 
    select(-year) %>% 
    arrange(city, time)

tsir_s_dfs <- list()
cities <- unique(all_cities$city)

for (i in 1:length(cities)) {
    fit_data <- all_cities %>% 
        filter(city == cities[i]) %>% 
        select(time, cases, births, pop)

    if (sum(fit_data$cases == 0) / nrow(fit_data) > 0.3) {
        epidemics <- "break"
    } else {
        epidemics <- "cont"
    }

    tsir_s_dfs[[i]] <- fit_data %>% 
    {runtsir(data = ., 
                 IP = 2,
                 xreg = 'cumcases', 
                 regtype='gaussian',
                 alpha = NULL,
                 sbar = NULL,
                 family = 'gaussian', 
                 link = 'identity',
                 method = 'negbin', 
                 nsim = 100,
                 epidemics = epidemics)
                     }
        print(i)
}

tsir_dfs <- list()

for (i in 1:length(tsir_s_dfs)) {
    tsir_dfs[[i]] <- data.frame(susc = tsir_s_dfs[[i]]$simS$mean,
                                city = cities[i])
}

tsir_susc <- Reduce(rbind, tsir_dfs) %>% 
    cbind(all_cities[, c("time", "cases", "births", "pop")]) %>%
    select(time, city, cases, births, pop, susc)

write.csv(tsir_susc, 
          "../../output/data/tsir_susceptibles/tsir_susceptibles.csv", 
          row.names = FALSE)

