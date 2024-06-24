

url <- "https://raw.githubusercontent.com/msylau/measles_competing_risks/master/data/raw_data/measles.RData"
local_path <- "measles.RData"
download.file(url, local_path, method = "auto")

load(local_path)

write.csv(ewBu4464, "../../../data/births/ewBu4464.csv", row.names = FALSE)


