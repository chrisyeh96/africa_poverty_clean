#!/bin/bash

# This Bash script downloads shapefiles from GADM v3.6 into the
# data/shapefiles/ directory.
#
# Run this script from within the preprocessing/ directory.

mkdir ../data/shapefiles
cd ../data/shapefiles

country_codes=(
    "AGO"  # Angola
    "BEN"  # Benin
    "BFA"  # Burkina Faso
    "CMR"  # Cameroon
    "CIV"  # Côte d'Ivoire
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

for code in "${country_codes[@]}"
do
    # download ZIP'ed shapefiles from GADM v3.6
    wget "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_${code}_shp.zip"

    # unzip only level-2 administrative region shapefiles
    unzip "gadm36_${code}_shp.zip" *_2.* -d "gadm36_${code}_shp"

    # delete the zip file
    rm "gadm36_${code}_shp.zip"
done