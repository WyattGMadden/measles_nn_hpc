#! /apps/R/4.2.2/bin/Rscript
library(tidyverse)
library(prophet)

set.seed(10161993)
city_index = 1
save_dir = "../../../output/data/prophet/"

fit_prophet <- function(city_index, save_dir = "../../../output/data/prophet/") {

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
      mutate(year = floor(time), biweek = (time - year) * 26, # Convert fractional year to biweek count
             date = as.Date(paste(year, 1, 1, sep="-")) + round(biweek * 14) - 1) %>% 
      select(city, cases, time, date)

    # Collect forecasts from all cities
    all_forecasts <- list()

    # Define the forecasting horizons
    i <- unique(all_cities$city)[city_index]

    # Fit Prophet and make selected k-step ahead forecasts at each time step
    print(paste("Fitting model for city", i))
    df_to_fit <- all_cities %>% 
      filter(city == i) %>% 
      select(date, cases, time) %>% 
      rename(ds = date, y = cases) %>% 
      arrange(ds) %>% 
      na.omit()

    tlag <- 130  # Time lag (window size for the rolling forecast)
    k <- 52

    # Rolling forecast
    for (j in seq(tlag, nrow(df_to_fit) - k)) {
      train_set <- df_to_fit[(j-tlag + 1):j, c("ds", "y")]  # Select cases for training up to j
      model <- prophet(train_set)
      # Forecast for each specified horizon
      print(paste("Time", df_to_fit$time[j]))
      future <- make_future_dataframe(model, periods = k * 2, freq = 'week')
      future_forecast <- predict(model, future)
      # Create a data frame of this forecast, with full times
      proph_pred = (as.vector(tail(future_forecast$yhat, k * 2))[(1:(k * 2)) %% 2 == 0])
      city_temp = rep(i, k)
      forecast_frame <- tibble(
        time = df_to_fit$time[(j + 1):(j + k)],
        k = 1:k,
        prophet_pred = proph_pred,
        city = city_temp
      )
      all_forecasts[[length(all_forecasts) + 1]] <- forecast_frame
      }

    # Combine all forecasts into a single dataframe
    final_forecasts <- bind_rows(all_forecasts)

    # Save the combined forecasts to a single CSV file
    write_csv(final_forecasts, paste0(save_dir, "city_", i, "_prophet_forecasts_uk_processed.csv"))
    writeLines(paste("City", i, "forecasts saved to", paste0(save_dir, "city_", i, "_prophet_forecasts_uk_processed.csv")), stderr())
}



