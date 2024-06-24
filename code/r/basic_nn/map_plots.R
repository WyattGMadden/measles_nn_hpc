
library(tidyverse)
library(patchwork)
library(sf)
library(rnaturalearth) 
library(rnaturalearthdata)
theme_set(theme_classic())
set.seed(42)
save_dir <- "../../../output/figures/"
read_dir <- "../../../data/"

#all original cases/birth/etc data from max's github
all_cities <- read_csv("../../../output/data/basic_nn/all_cases_from_max_gh.csv")

big_cities <- all_cities %>% 
    filter(time == min(time)) %>% 
    slice_max(n = 4, min_pop) %>% 
    pull(city)

#spatial plot
uk_map <- ne_countries(scale = "medium", 
                       country = "united kingdom", 
                       returnclass = "sf")

eng_wales_map <- uk_map[uk_map$admin %in% c("England", "Wales"), ]

first_biweek_1960 <- round(min(abs(full_dat$time - 1961)) + 1961, 2)

spat_plot_dat <- full_dat |>
    mutate(time = round(time, 2)) %>%
    filter(k == 52, 
           round(time, 2) == first_biweek_1960)

ewplot <- ggplot(data = uk_map) +
  geom_sf(fill = "white", color = "black") +
  geom_point(data = spat_plot_dat, 
             aes(x = long, 
                 y = lat,
                 color = log(cases)), 
             size = 1, alpha = 1) +
  cividis::scale_color_cividis() +
  lims(x = c(-6, 2), y = c(50, 56)) +
  labs(x = "Longitude",
       y = "Latitude",
       color = "Log(Incidence)") + 
  theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 10))


time_plot_dat <- all_cities |>
    mutate(time = round(time, 2)) %>%
    filter(city %in% big_cities)

timeplot <- time_plot_dat %>% 
    mutate(city = factor(city, levels = c("London", 
                                          "Birmingham", 
                                          "Liverpool", 
                                          "Manchester"))) %>%
    ggplot() +
    geom_line(aes(x = time, y = cases, color = city, linetype = city), alpha = 1) +
    scale_color_manual(values = c("London" = "black", 
                                  "Liverpool" = "blue3", 
                                  "Birmingham" = "yellow4",
                                  "Manchester" = "grey")) +
    labs(x = "Time",
         y = "Incidence",
         color = "",
         linetype = "") +
    theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 10))

figa <- ewplot


figb <- timeplot

fig1 <- ewplot / timeplot +
    plot_layout(widths = c(1, 2), 
                heights = c(1),
                ncol = 2) +
    plot_annotation(tag_levels = 'A')

scale_factor <- 3
ggsave(paste0(save_dir, "fig1data_spat_temp.png"),
       fig1, 
       width = 3 * scale_factor,
       height = 1.5 * scale_factor,
       dpi = 600)

