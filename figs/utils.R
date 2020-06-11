# Libraries explicitly used:
# - data.table
# - dplyr
# - ggplot2
# - readr
# - sf

library(dplyr, quietly = TRUE)  # for the pipe %>% operator
library(ggplot2, quietly = TRUE)

theme_anne = function(font = "sans", size = 10) {
    ggthemes::theme_tufte(base_size = size, base_family = font) %+replace%
        theme(
            axis.line.x = element_line(color = "black", size = .2),
            axis.line.y = element_line(color = "black", size = .2),
            panel.background = element_blank(),
            plot.background = element_rect(fill = "transparent", color = NA),
            plot.title = element_text(hjust = 0.5)
        )
}


# Reads in admin-0 shapefiles for African countries and concatenates them into
# a single sf tibble with columns ["iso3", "NAME_0", "geometry"], where each
# row corresponds to a country.
load_africa_shp = function() {
    # the usual African countries, plus "Western Sahara", "Réunion", and "Mayotte"
    africa_countries = c(
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
        "Cote d'Ivoire", "Democratic Republic of the Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana",
        "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mayotte",
        "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
        "Republic of the Congo", "Réunion", "Rwanda", "Sao Tome and Principe",
        "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
        "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia",
        "Uganda", "Western Sahara", "Zambia", "Zimbabwe")

    # get the ISO3 code for each country
    africa_iso3s = readr::read_csv("../data/surveys/crosswalk_countries.csv") %>%
        dplyr::filter(country %in% africa_countries) %>%
        dplyr::pull(iso3)

    # read all admin-0 shapefiles and concatenate together into a single sf tibble
    # with columns ["iso3", "NAME_0", "geometry"]
    africa_shp = list()
    for (i in seq(1, length(africa_iso3s))) {
        iso3 = africa_iso3s[i]
        path = sprintf("../data/shapefiles/gadm36_%s_shp/gadm36_%s_0.shp", iso3, iso3)

        # "type = 6" converts all geometries to MULTIPOLYGON so we don't get any
        # weird problems when concatenating individual sf tibbles together
        africa_shp[[i]] = sf::st_read(path, stringsAsFactors = FALSE, quiet = TRUE, type = 6)
    }
    africa_shp = sf::st_as_sf(data.table::rbindlist(africa_shp)) %>%
        sf::st_simplify(dTolerance = 0.05) %>%
        dplyr::rename(iso3 = GID_0)
}
