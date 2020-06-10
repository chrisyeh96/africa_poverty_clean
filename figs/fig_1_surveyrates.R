# This script creates an updated version of Figures 1a and 1b from the paper.
#
# Prerequisites: download shapefiles. See
#   `preprocessing/3_download_gadm_shapefiles.sh`.
#
# Libraries used:
# - data.table
# - dplyr
# - ggplot2
# - readr
# - scales
# - sf
# - tibble
# - tidyr

source("utils.R")


######################################################
# Load data
#
# NOTE: We have updated many of the CSV files since the original paper was
# published. We use the updated CSV files in the code below. The old and
# updated filenames are as follows:
#
# old                           | updated
# ------------------------------|-----------------------------
# data/surveys/paper/povcal.csv | data/surveys/povcal.csv
# data/surveys/povcal_pop.csv   | **replaced with povcal.csv**
# data/surveys/dhs.csv          | data/surveys/dhs.csv
# data/surveys/us_surveys.csv   | data/surveys/us_surveys.csv
######################################################

year_cols = as.character(2000:2016)

# data frame for coverting between country names and codes
cross = readr::read_csv("../data/surveys/crosswalk_countries.csv") %>%
    dplyr::select(iso3, country)

# `povcal`, `dhs`, and `pop` are data frames with columns
#   ["country", "iso3", "2000", ..., "2016"]
#   representing annual PovcalNet survey size, DHS survey size, and
#   population for every country in the world, sorted by country
#
# `us` is a data frame with columns
#   ["survey", "2000", ..., "2016"]
#   representing annual survey sizes in the U.S.

# In the updated version of `povcal_time_pop.csv`, "-1" is recorded to indicate
# that a survey was conducted but no sample size is reported. Here, we replace
# "-1" with the size of the largest survey otherwise conducted in that country
# between 2000 and 2016, or "1" if no other survey was conducted.
povcal = readr::read_csv("../data/surveys/povcal_time_pop.csv") %>%
    dplyr::right_join(cross, by = c("country", "iso3")) %>%
    dplyr::arrange(country)
povcal = povcal[, c("country", "iso3", year_cols)]
povcal[is.na(povcal)] = 0
povcal[, year_cols] = t(apply(povcal[, year_cols], 1, function(row) {
    ifelse(row == -1, max(max(row), 1), row)
}))

dhs = readr::read_csv("../data/surveys/dhs_time.csv") %>%
    dplyr::right_join(cross, by = "country") %>%
    dplyr::arrange(country)
dhs = dhs[, c("country", "iso3", year_cols)]
dhs[is.na(dhs)] = 0

pop = readr::read_csv("../data/surveys/population_time.csv") %>%
    dplyr::rename(iso3 = code, country_wb = country) %>%
    dplyr::right_join(cross, by = "iso3") %>%
    dplyr::arrange(country)
pop = pop[, c("country", "iso3", year_cols)]
pop[is.na(pop)] = 0

us = readr::read_csv("../data/surveys/us_surveys_time.csv")
us = us[, c("survey", year_cols)]
us[is.na(us)] = 0


######################################################
# Figure 1a: survey counts per African country
######################################################

surveys_by_country = povcal[, c("country", "iso3")]
surveys_by_country$povcal = rowSums(povcal[, year_cols] > 0, na.rm = TRUE)
surveys_by_country$dhs = rowSums(dhs[, year_cols] > 0, na.rm = TRUE)
surveys_by_country$total = surveys_by_country$povcal + surveys_by_country$dhs
surveys_by_country$`years btwn surveys` = 17 / surveys_by_country$total

breaks = c(1, 2, 3, 4, 5, 10, 20)
color =  c("#F7F7B8", "#F7EB7D", "#F7E04C", "#F7CA23", "#FFA94D", "#FF4D4D")

survey_counts = load_africa_shp() %>%
    dplyr::inner_join(surveys_by_country, by = "iso3")
survey_counts$`years btwn surveys` = cut(survey_counts$`years btwn surveys`, breaks)

p = ggplot(survey_counts) +
    geom_sf(aes(fill = `years btwn surveys`), color = "white", size = 0.1) +
    scale_fill_manual(values = color, drop = FALSE, na.value = "black") +
    theme_anne(font = "sans", size = 10) +
    theme(axis.line.x = element_blank(),
          axis.line.y = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank(),
          line = element_blank())
ggsave("output/fig_1a_mapsurveyscount.pdf", plot = p,  width = 7, height = 7)


######################################################
# Figure 1b: survey and satellite revisit rates
######################################################

# NOTE: The code used to create the original Figure 1b in the paper incorrectly
#   excluded "Cabo Verde" and "Cote d'Ivoire" from this list.
africa_countries = c(
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
    "Cote d'Ivoire", "Democratic Republic of the Congo", "Djibouti", "Egypt",
    "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
    "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
    "Republic of the Congo", "Rwanda", "Sao Tome and Principe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
    "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia",
    "Uganda", "Zambia", "Zimbabwe")

# Reshape `povcal`, `dhs`, and `pop` into tidy tables with columns
# ["country", "year", value]

povcal = povcal %>%
    tidyr::gather(key = "year", value = "povcal", -country, -iso3) %>%
    dplyr::mutate(year = as.integer(year))

dhs = dhs %>%
    tidyr::gather(key = "year", value = "dhs", -country, -iso3) %>%
    dplyr::mutate(year = as.integer(year))

pop = pop %>%
    tidyr::gather(key = "year", value = "pop", -country, -iso3) %>%
    dplyr::mutate(year = as.integer(year))

# merge `povcal`, `dhs`, and `pop` by country code
# - results in a table of (country, iso3, year, dhs, povcal, population)
full = dplyr::inner_join(dhs, povcal, by = c("country", "iso3", "year")) %>%
    dplyr::inner_join(pop, by = c("country", "iso3", "year")) %>%
    dplyr::filter(country %in% africa_countries)

# calculate Africa annual survey revisit rate (avg. # of days between survey revisit per person)
# - creates a table of (year, dhs, pop, povcal, survey_revisit)
africa = full %>%
    dplyr::group_by(year) %>%
    dplyr::summarize_at(c("dhs", "pop", "povcal"), sum, na.rm = TRUE)
africa$survey_revisit = (africa$pop * 365) / (africa$dhs + africa$povcal)

# calculate US annual survey revisit rate
# - creates a table of (year, pop, num_surveyed, survey_revisit)
us = us %>%
    dplyr::select(-survey) %>%
    dplyr::summarize_all(sum, na.rm = TRUE) %>%
    tidyr::gather(key = "year", value = "num_surveyed") %>%
    dplyr::mutate(year = as.integer(year)) %>%
    dplyr::inner_join(pop[pop$iso3 == "USA",], by = "year")
us$survey_revisit = (us$pop * 365) / us$num_surveyed

# concatenate the Africa and US data frames
africa$survey_group = "africa_surveys"
africa = africa %>% dplyr::select(year, survey_group, survey_revisit)
us$survey_group = "us_surveys"
us = us %>% dplyr::select(year, survey_group, survey_revisit)
surveys = rbind(us, africa)

# calculate annual revisit rate for each satellite (avg. # of days between
# satellite revisit per location)
# - divide total # of visits by 500 because 500 location samples per year
# - l5 = Landsat-5, l7 = Landsat-7, l8 = Landsat-8
# - s1 = Sentinel-1, s2 = Sentinel-2
# - modis = MODIS
gee = readr::read_csv("../data/overpass/dhs_sample_GEE.csv",
                      col_types = readr::cols(year = readr::col_integer())) %>%
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
planet = readr::read_csv("../data/overpass/dhs_sample_Planet.csv",
                         col_types = readr::cols(year = readr::col_integer())) %>%
    dplyr::select(year, cluster_id, count_PlanetScope, count_RapidEye) %>%
    dplyr::group_by(year) %>%
    dplyr::summarize(
        planetscope = 365 / (sum(count_PlanetScope, na.rm = TRUE)/500),
        rapideye = 365 / (sum(count_RapidEye, na.rm = TRUE)/500),
        all_planet = 365 / (sum(count_RapidEye, count_PlanetScope, na.rm = TRUE)/500)
    )

# DigitalGlobe satellites
dg_satellites = c("IKONOS", "QuickBird-2", "GeoEye-1", "WorldView-1",
                  "WorldView-2", "WorldView-3", "WorldView-4")
dg = readr::read_csv(
    "../data/overpass/landinfo_dhs_sample_nocatalog.csv",
    col_types = readr::cols_only(
        sensor = readr::col_character(),
        date = readr::col_date(format = "%d-%b-%y"),  # e.g. "8-Jan-03"
        cloud = readr::col_number()
    )) %>%
    dplyr::filter((cloud <= 30) & (sensor %in% dg_satellites)) %>%
    dplyr::mutate(year = as.integer(format(date, "%Y"))) %>%
    dplyr::group_by(year) %>%
    dplyr::summarize(dg = 365 / (dplyr::n() / 500))

# Create a table of (year, satellite, satellite_revisit)
resolution = tibble::tibble(
    satellite = c("s2", "all_l", "planetscope", "dg", "rapideye"),
    res       = c(  10,      30,             3,   .6,          5))
overpass = gee %>%
    dplyr::inner_join(planet, by = "year") %>%
    dplyr::inner_join(dg, by = "year") %>%
    dplyr::select(year, s2, all_l, planetscope, dg, rapideye) %>%
    tidyr::gather(key = "satellite", value = "satellite_revisit", -year) %>%
    dplyr::inner_join(resolution, by = "satellite")
overpass[overpass == Inf] = NA

# transformation for y-axis: reverse log base-10
reverselog_trans = function(base = 10) {
    scales::trans_new(
        name = paste0("reverselog-", format(base)),
        transform = function(x) -log(x, base),
        inverse = function(x) base^(-x),
        breaks = scales::log_breaks(base = base),
        domain = c(1e-100, Inf))
}

p = ggplot() +
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

    scale_color_manual(values = colorRampPalette(c("#06276E", "#9BC7FF"), bias = 2)(5)) +

    # hide legend and axes lines
    theme_anne("sans", size=25) +
    theme(
        legend.position = "none",
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
    )

dir.create(file.path("output"), showWarnings = FALSE)
ggsave("output/fig_1b_surveyrates.pdf", plot=p,  width=11.5, height=7.4)
