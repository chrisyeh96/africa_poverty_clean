# libraries used:
# - dplyr
# - ggplot2
# - readr
# - scales
# - tidyr

source("utils.R")

filter_year <- function(df, year1, year2) {
    df %>%
    dplyr::mutate(year = as.integer(year)) %>%
    dplyr::filter((year >= 2000) & (year <= 2016))
}

# Create a table of (country, year, # of people sampled by PovcalNet)
# - NOTE 1: We have updated povcal_time_pop.csv since the original paper was published.
#     To match the figure from the paper, replace "../data/surveys/povcal_time_pop.csv"
#     with "../data/surveys/paper/povcal_time_pop.csv".
# - NOTE 2: In the updated version of povcal_time_pop.csv, "x" is recorded to indicate that
#     a survey was conducted but no sample size is reported.
povcal <- readr::read_csv("../data/surveys/paper/povcal_time_pop.csv") %>%
# povcal <- readr::read_csv("../data/surveys/povcal_time_pop.csv") %>%
    tidyr::gather(key = "year", value = "povcal", -country) %>%
    dplyr::mutate(povcal = ifelse(is.na(povcal) | (povcal == ""), 0, povcal)) %>%
    filter_year(2000, 2016)

# Create a table of (country, year, # of people sampled by DHS)
# - NOTE: We have updated dhs_time.csv since the original paper was published.
#     To match the figure from the paper, replace "../data/surveys/dhs_time.csv"
#     with "../data/surveys/paper/dhs_time.csv".
dhs <- readr::read_csv("../data/surveys/paper/dhs_time.csv") %>%
# dhs <- readr::read_csv("../data/surveys/dhs_time.csv") %>%
    tidyr::gather(key = "year", value = "dhs", -country) %>%
    filter_year(2000, 2016)

# Create a table of (country, year, population)
pop <- readr::read_csv("../data/surveys/population_time.csv") %>%
    tidyr::gather(key = "year", value = "pop", -country, -code) %>%
    dplyr::select(iso3 = code, year, pop) %>%
    filter_year(2000, 2016)

africa_countries <- c(
    "Algeria", "Angola",  "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Democratic Republic of the Congo", "Djibouti", "Egypt",
    "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia",
    "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
    "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
    "Republic of the Congo", "Rwanda", "Sao Tome and Principe", "Senegal",
    "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
    "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia",
    "Zimbabwe")

# merge povcal, dhs, and pop by crosswalk country names
cross <- readr::read_csv("../data/surveys/crosswalk_countries.csv")
full <- merge(dhs, cross, by.x = "country", by.y = "country_simp", all.x = TRUE) %>%
    merge(povcal, by = c("country", "year"), all = TRUE) %>%
    merge(pop, by = c("iso3", "year"), all = TRUE) %>%
    dplyr::select(year, country, dhs, pop, povcal) %>%
    dplyr::filter(country %in% africa_countries)

# TODO: deal with the "x" values in the updated povcal


######################################################
# Figure 1b: survey and satellite revisit rates
######################################################

# calculate Africa annual survey revisit rate (avg. # of days between survey revisit per person)
# - creates a table of (year, dhs, pop, povcal, survey_revisit)
africa <- full %>%
    dplyr::group_by(year) %>%
    dplyr::summarise_at(c("dhs", "pop", "povcal"), sum, na.rm = TRUE)
africa$survey_revisit <- (africa$pop * 365) / (africa$dhs + africa$povcal)

# calculate US annual survey revisit rate
# - creates a table of (year, pop, num_surveyed, survey_revisit)
us <- readr::read_csv("../data/surveys/paper/us_surveys_time.csv") %>%
    dplyr::select(-survey) %>%
    dplyr::summarise_all(sum, na.rm = TRUE) %>%
    tidyr::gather(key = "year", value = "num_surveyed") %>%
    filter_year(2000, 2016) %>%
    merge(pop[pop$iso3 == "USA",], by = "year")
us$survey_revisit <- (us$pop * 365) / (us$num_surveyed)

# concatenate the Africa and US data frames
africa$survey_group <- "africa_surveys"
africa <- africa %>% dplyr::select(year, survey_group, survey_revisit)
us$survey_group <- "us_surveys"
us <- us %>% dplyr::select(year, survey_group, survey_revisit)
surveys <- rbind(us, africa)

# calculate annual revisit rate for each satellite (avg. # of days between satellite revisit per location)
# - divide total # of visits by 500 because 500 location samples per year
# - l5 = Landsat-5, l7 = Landsat-7, l8 = Landsat-8
# - s1 = Sentinel-1, s2 = Sentinel-2
# - modis = MODIS
gee <- readr::read_csv("../data/overpass/dhs_sample_GEE.csv") %>%
    dplyr::group_by(year) %>%
    dplyr::summarize(
        l5 = 365 / (sum(num_l5, na.rm = TRUE)/500), 
        l7 = 365 / (sum(num_l7, na.rm = TRUE)/500), 
        l8 = 365 / (sum(num_l8, na.rm = TRUE)/500),
        s1 = 365 / (sum(num_s1, na.rm = TRUE)/500), 
        s2 = 365 / (sum(num_s2, na.rm = TRUE)/500), 
        all_s = 365 / (sum(num_s1, num_s2, na.rm = TRUE)/500),
        all_l = 365 / (sum(num_l5, num_l8, num_l7, na.rm = TRUE)/500), 
        modis = 365 / (sum(num_modis, na.rm = TRUE)*8/500)  # multiply modis by 8 because it's an 8 day composite
    )
# Planet Labs satellites
planet <- readr::read_csv("../data/overpass/dhs_sample_Planet.csv") %>%
    dplyr::select(year, cluster_id, count_PlanetScope, count_RapidEye) %>%
    dplyr::group_by(year) %>%
    dplyr::summarize(
        planetscope = 365 / (sum(count_PlanetScope, na.rm = TRUE)/500),
        rapideye = 365 / (sum(count_RapidEye, na.rm = TRUE)/500),
        all_planet = 365 / (sum(count_RapidEye, count_PlanetScope, na.rm = TRUE)/500)
    )
# DigitalGlobe satellites
# - The version of the code used in the paper incorrectly parsed the "cloud" column
#   as a character vector instead of as an integer vector, so filtering for cloud
#   cover <= 30% was incorrectly done. To match the figure from the paper, uncomment
#   the line below which converts the "cloud" column to a character vector.
# - TODO: resolution??
dg <- readr::read_csv(
    "../data/overpass/landinfo_dhs_sample_nocatalog.csv",
    col_types = readr::cols(
        date = readr::col_date(format = "%d-%b-%y"),  # e.g. "8-Jan-03"
        cloud = readr::col_number(),
        resolution = readr::col_number()  # TODO: dg resolution is up to 1.5m
    )) %>%
    # dplyr::mutate(cloud = as.character(cloud)) %>%  # uncomment this line to reproduce the figure in the paper
    dplyr::filter(cloud <= 30) %>%
    dplyr::mutate(year = as.double(format(date, "%Y"))) %>%
    dplyr::group_by(year) %>%
    dplyr::summarise(dg = 365 / (n() / 500))

# Create a table of (year, satellite, satellite_revisit)
resolution <- data.frame(
    satellite = c("s2", "all_l", "planetscope", "dg", "rapideye"), 
    res       = c(  10,      30,             3,   .6,          5))  # TODO: dg resolution is up to 1.5m
overpass <- gee %>%
    merge(planet, by = "year") %>%
    merge(dg, by = "year") %>%
    dplyr::select(year, s2, all_l, planetscope, dg, rapideye) %>%
    tidyr::gather(key = "satellite", value = "satellite_revisit", -year) %>%
    merge(resolution, by = "satellite")
overpass[overpass == Inf] <- NA

# transformation for y-axis: reverse log base-10
reverselog_trans <- function(base = 10) {
    scales::trans_new(
        name = paste0("reverselog-", format(base)),
        transform = function(x) -log(x, base),
        inverse = function(x) base^(-x),
        breaks = scales::log_breaks(base = base),
        domain = c(1e-100, Inf))
}

p <- ggplot() +
    geom_hline(yintercept = 1, linetype = "66", size = .35, color = "grey") +
    geom_hline(yintercept = 7, linetype = "66", size = .35, color = "grey") +
    geom_hline(yintercept = 30, linetype = "66", size = .35, color = "grey") +
    geom_hline(yintercept = 365, linetype = "66", size = .35, color = "grey") +
    geom_hline(yintercept = 3650, linetype = "66", size = .35, color = "grey") +
    geom_hline(yintercept = 36500, linetype = "66", size = .35, color = "grey") +
    geom_hline(yintercept = 365000, linetype = "66", size = .35, color = "grey") +

    geom_line(data = overpass, aes(year, satellite_revisit, group = satellite, color = as.factor(res)), size = 0.8) +
    geom_line(data = surveys, aes(year, survey_revisit, group = survey_group), size = 0.8, linetype = "1232") +

    scale_y_continuous(
        name = "Avg. household revisit interval (days)",
        trans = reverselog_trans(10),
        limits = c(4927500, 0.5),
        breaks = c(1,  7, 30, 100, 365, 3650,    10000, 36500, 365000,     1000000, 4927500),
        labels = c(1, "", "", 100,  "",   "", "10,000",    "",     "", "1,000,000",      "")) +  # 13,500 years
    scale_x_continuous(
        name = "Year",
        limits = c(2000, 2018),
        breaks = seq(2000, 2018, 2)) +

    scale_colour_manual(values = colorRampPalette(c("#06276E", "#9BC7FF"), bias = 2)(5)) +

    # hide legend and axes lines
    theme_anne("sans", size=25) +
    theme(
        legend.position = "none",
        axis.line.x = element_line(color = NA),
        axis.line.y = element_line(color = NA)
    )

dir.create(file.path("output"), showWarnings = FALSE)
ggsave("output/fig_1b_surveyrates.pdf", plot=p,  width=11.5, height=7.4)