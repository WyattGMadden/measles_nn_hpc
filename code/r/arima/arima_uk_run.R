#! /apps/R/4.2.2/bin/Rscript
library(tidyverse)
library(forecast)

set.seed(10161993)

save_dir <- "../../../output/data/autoarima/"

# Read and prepare the data
births <- read_csv("../../../data/data_from_measles_LASSO/births_urban.csv") %>%
    rename("year" = X) %>% 
    pivot_longer(2:ncol(.), names_to = "city", values_to = "births") %>%
    mutate(births = births / 26)

inf_pop_urb <- read_csv("../../../data/data_from_measles_competing_risks/inferred_pop_urban.csv") %>%
    rename("time" = `...1`) %>% 
    pivot_longer(2:ncol(.), names_to = "city", values_to = "pop")

cases <- read_csv("../../../data/data_from_measles_LASSO/cases_urban.csv") %>%
    rename("time" = `...1`) %>% 
    pivot_longer(2:ncol(.), names_to = "city", values_to = "cases") %>% 
    mutate(year = floor(time))

all_cities <- cases %>% 
  left_join(births, by = c("year", "city")) %>% 
  left_join(inf_pop_urb, by = c("time", "city")) %>% 
  mutate(time = 1900 + time) %>% 
  select(-year)

# Collect forecasts from all cities
all_forecasts <- list()

# Define the forecasting horizons
forecasting_horizons <- c(1, 4, 12, 20, 34, 52)

# Fit auto.arima and make selected k-step ahead forecasts at each time step
for (i in unique(all_cities$city)) {
  print(paste("Fitting model for city", i))
  df_to_fit <- all_cities %>% 
    filter(city == i) %>% 
    select(time, cases) %>% 
    arrange(time) %>% 
    na.omit()

  tlag <- 130  # Time lag (window size for the rolling forecast)

  # Rolling forecast
  for (j in seq(tlag, nrow(df_to_fit) - max(forecasting_horizons))) {
    train_set <- df_to_fit[j - (tlag - 1):j, 2]  # Select cases for training
    model <- auto.arima(train_set)
    # Forecast for each specified horizon
    print(paste("Time", df_to_fit$time[j]))
    for (k in forecasting_horizons) {
      future_forecast <- forecast(model, h = k)
      # Create a data frame of this forecast
      forecast_frame <- tibble(
        time = df_to_fit$time[j],
        k = k,
        auto_arima = as.vector(future_forecast$mean),
        city = i
      )
      all_forecasts[[length(all_forecasts) + 1]] <- forecast_frame
    }
  }
}

# Combine all forecasts into a single dataframe
final_forecasts <- bind_rows(all_forecasts)

# Save the combined forecasts to a single CSV file
write_csv(final_forecasts, paste0(save_dir, "autoarima_uk_processed.csv"))
writeLines("All city forecasts for selected steps have been combined and saved.")

