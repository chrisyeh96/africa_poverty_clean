#!/bin/bash

# This Bash script downloads shapefiles from GADM v3.6 into the
# data/shapefiles/ directory.
#
# Run this script from within the preprocessing/ directory.
#
# Prerequisites: None.

mkdir -p ../data/shapefiles
cd ../data/shapefiles

africa_country_codes=(
    "DZA"  # Algeria
    "AGO"  # Angola
    "BEN"  # Benin
    "BWA"  # Botswana
    "BFA"  # Burkina Faso
    "BDI"  # Burundi
    "CPV"  # Cabo Verde
    "CMR"  # Cameroon
    "CAF"  # Central African Republic
    "TCD"  # Chad
    "COM"  # Comoros
    "CIV"  # Cote d'Ivoire
    "COD"  # Democratic Republic of the Congo
    "DJI"  # Djibouti
    "EGY"  # Egypt
    "GNQ"  # Equatorial Guinea
    "ERI"  # Eritrea
    "ETH"  # Ethiopia
    "GAB"  # Gabon
    "GMB"  # Gambia
    "GHA"  # Ghana
    "GIN"  # Guinea
    "GNB"  # Guinea-Bissau
    "KEN"  # Kenya
    "LSO"  # Lesotho
    "LBR"  # Liberia
    "LBY"  # Libya
    "MDG"  # Madagascar
    "MWI"  # Malawi
    "MLI"  # Mali
    "MRT"  # Mauritania
    "MUS"  # Mauritius
    "MAR"  # Morocco
    "MOZ"  # Mozambique
    "MYT"  # **Mayotte - technically a French territory
    "NAM"  # Namibia
    "NER"  # Niger
    "NGA"  # Nigeria
    "REU"  # **Réunion - technically a French territory
    "COG"  # Republic of the Congo
    "RWA"  # Rwanda
    "STP"  # Sao Tome and Principe
    "SEN"  # Senegal
    "SYC"  # Seychelles
    "SLE"  # Sierra Leone
    "SOM"  # Somalia
    "ZAF"  # South Africa
    "SSD"  # South Sudan
    "SDN"  # Sudan
    "SWZ"  # Swaziland
    "TZA"  # Tanzania
    "TGO"  # Togo
    "TUN"  # Tunisia
    "UGA"  # Uganda
    "ESH"  # **Western Sahara - disputed territory
    "ZMB"  # Zambia
    "ZWE"  # Zimbabwe
)
# ** indicates a territory that is only included for map-plotting purposes
#    but was otherwise not considered in this paper

dhs_country_codes=(
    "AGO"  # Angola
    "BEN"  # Benin
    "BFA"  # Burkina Faso
    "CMR"  # Cameroon
    "CIV"  # Cote d'Ivoire
    "COD"  # Democratic Republic of the Congo
    "ETH"  # Ethiopia
    "GHA"  # Ghana
    "GIN"  # Guinea
    "KEN"  # Kenya
    "LSO"  # Lesotho
    "MWI"  # Malawi
    "MLI"  # Mali
    "MOZ"  # Mozambique
    "NGA"  # Nigeria
    "RWA"  # Rwanda
    "SEN"  # Senegal
    "SLE"  # Sierra Leone
    "TZA"  # Tanzania
    "TGO"  # Togo
    "UGA"  # Uganda
    "ZMB"  # Zambia
    "ZWE"  # Zimbabwe
)

for code in ${africa_country_codes[@]}
do
    # download ZIP'ed shapefiles from GADM v3.6
    wget "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_${code}_shp.zip"

    # for all African countries, unzip shapefiles for level-0 administrative
    #   regions (country-level), overwriting existing files
    unzip -o "gadm36_${code}_shp.zip" *_0.* -d "gadm36_${code}_shp"

    # for DHS countries, unzip shapefiles for level-2 administrative regions
    #   (district-level), overwriting existing files
    if [[ " ${dhs_country_codes[@]} " =~ " ${code} " ]]
    then
        unzip -o "gadm36_${code}_shp.zip" *_2.* -d "gadm36_${code}_shp"
    fi

    # delete the zip file
    rm "gadm36_${code}_shp.zip"
done