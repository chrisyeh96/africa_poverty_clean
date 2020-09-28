## Table of Contents

* [Folder structure](#folder-structure)
* [CSV Files](#csv-files)
* [Python Pickle (`.pkl`) Files](#python-pickle-pkl-files)
* [Files in `data/overpass/`](#files-in-dataoverpass)
* [Files in `data/survey/`](#files-in-datasurvey)
* [GPS coordinate displacement](#gps-coordinate-displacement)
* [Other notes about the data files](#other-notes-about-the-data-files)


## Folder structure

```
dhs_tfrecords/          # created by preprocessing/1_process_tfrecords.ipynb
    angola_2011/
    ...
    zimbabwe_2015/
dhs_tfrecords_raw/      # created by preprocessing/0_export_tfrecords.ipynb
dhsnl_tfrecords/        # created by preprocessing/1_process_tfrecords.ipynb
    angola_2010/
    ...
    zimbabwe_2016/
dhsnl_tfrecords_raw/    # created by preprocessing/0_export_tfrecords.ipynb
lsms_tfrecords/         # created by preprocessing/1_process_tfrecords.ipynb
    ethiopia_2011/
    ...
    uganda_2013/
lsms_tfrecords_raw/     # created by preprocessing/0_export_tfrecords.ipynb
overpass/
shapefiles/             # created by preprocessing/3_download_gadm_shapefiles.sh
surveys/
```

## CSV Files

[**`dhs_clusters.csv`**](./dhs_clusters.csv): This CSV file contains data derived from DHS surveys, aggregated to the cluster level. Each of the 19,669 rows (excluding the CSV header) represents a cluster from a single survey. The columns are as follows:

column          | description
----------------|------------
`country`       | country
`year`          | year that the survey started (some surveys lasted more than 1 year)
`lat`           | latitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
`lon`           | longitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
`GID_1`         | level 1 administrative region ID
`GID_2`         | level 2 administrative region ID
`wealthpooled`  | mean asset wealth index (AWI) of households within the cluster
`households`    | number of households surveyed in the cluster
`urban_rural`   | 0 = rural, 1 = urban

The administrative region IDs are taken from the [GADM](https://gadm.org/) v3.6 database. Due to the random displacement of geocoordinates, certain (lat, lon) coordinates do not strictly fall within an administrative region of the country. In these cases, the coordinates were assigned to the closest region. For points in Lesotho, where level 1 administrative regions are not further divided into level 2 administrative regions, the level 2 region ID is set to be the same as the level 1 region ID. For details, see [`other/match_gids.py`](../other/match_gids.py).

The AWI used in the `wealthpooled` index was computed as follows:
1. Take the PCA of household-level assets from 86 DHS surveys spanning 1994 to 2016. The value of the first principle component is the household-level asset wealth index (AWI).
    - TODO: explain which 86 DHS surveys. See `/atlas/group/poverty_data/surveys/wealthpooled/original_survey_file/AllCountryWealthIndex.csv`
    - TODO: explain how the AWI was created. See Neal Jean's poverty prediction repo.
2. Within each cluster, compute the mean household-level AWI to get the cluster-level AWI. We computed the AWI for 35,235 clusters.
3. Remove clusters from surveys started in 2008 or earlier (only keep surveys between 2009-2016, inclusive), and remove clusters whose GPS coordinates or urban/rural status were unknown. This leaves us with 19,669 clusters spanning 43 DHS surveys.


[**`dhsnl_locs.csv`**](./dhsnl_locs.csv): This CSV file contains locations used for training the transfer learning models. The 260,415 locations were sampled randomly from an 18x18 grid centered on each DHS survey location, where each grid cell has dimensions 0.00833° latitude and longitude. The original DHS cluster locations (from `dhs_clusters.csv`) are included as well. Locations for which satellite images could not be obtained were filtered out. The columns are as follows:

column    | description
----------|------------
`country` | country
`year`    | year
`lat`     | latitude coordinate
`lon`     | longitude coordinate


[**`lsms_clusters.csv`**](./lsms_clusters.csv): This CSV file indicates the locations and years of the LSMS clusters. Each of the 2,913 rows (excluding the CSV header) represents a cluster from a single survey. The columns are as follows:

column          | description
----------------|------------
`country`       | country
`year`          | year that the survey started (some surveys lasted more than 1 year)
`lat`           | latitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
`lon`           | longitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
`geolev1`       | administrative level-1 region ID
`geolev2`       | administrative level-2 region ID


[**`lsms_diffs.csv`**](./lsms_diffs.csv): This CSV file contains the changes in asset wealth index (AWI) over time for LSMS clusters. Each of the 1,539 rows (excluding the CSV header) represents a cluster across two surveys. The columns are as follows:

column          | description
----------------|------------
`country`       | country
`lat`           | latitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
`lon`           | longitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
`year.x`        | year that 1st survey started (some surveys lasted more than 1 year)
`year.y`        | year that 2nd survey started, `year.y` > `year.x`
`diff_of_index` | change in cluster-level AWI, see below
`index_of_diff` | cluster-level index of asset differences, see below
`households`    | number of households that were included in the calculation of the change in cluster-level AWI
`geolev1`       | administrative level-1 region ID
`geolev2`       | administrative level-2 region ID

For each pair of years, the `diff_of_index` value for a given cluster *C* is (cluster *C*'s AWI in year `year.y`) - (cluster *C*'s AWI in year `year.x`). Only households that were surveyed in both `year.x` and `year.y` were included in the mean. Here, the cluster-level AWI is computed in a similar fasion as for the DHS surveys, taking the PCA of the household assets and then taking the mean across households.

In contrast, the `index_of_diff` value for a given cluster is computed as follows:
1. For each household that exists across 2 surveys, compute the difference in that household's assets between the two surveys.
2. Compute the PCA of these asset-differences across the 5 countries for which we have LSMS surveys. The value of the first principle component is the household-level index of asset differences.
3. Within each cluster, compute the mean household-level index of asset differences to get the cluster-level index of asset differences.

Note that for the same cluster, the number of households used in one pair of years (`year.x`, `year.y`) might be different from the number of households in a different pairt of years. This is because even though LSMS attempts to be a panel survey, some households are not recorded at each survey. For example, a household surveyed in 2005 and 2009 rounds may have moved between 2009 and 2013. Thus, this household would be included in the cluster household count in (`year.x` = 2005, `year.y` = 2009) but not in (`year.x` = 2009, `year.y` = 2013).

The following DHS surveys were included:
TODO for wealthpooled
TODO for predictions

The following LSMS surveys were included (both for creating the "wealthpooled" index and for training the machine learning models):
- "Ethiopia 2011": Ethiopia Rural Socioeconomic Survey 2011-2012, Survey ID `ETH_2011_ERSS_v02_M`([link](https://microdata.worldbank.org/index.php/catalog/2053))
- "Ethiopia 2015": Ethiopia Socioeconomic Survey 2015-2016, Wave 3, Survey ID `ETH_2015_ESS_v03_M ` ([link](https://microdata.worldbank.org/index.php/catalog/2783))
- "Malawi 2010": Malawi Third Integrated Household Survey 2010-2011, Survey ID `MWI_2010_IHS-III_v01_M` ([link](https://microdata.worldbank.org/index.php/catalog/1003))
- "Malawi 2016": Malawi Integrated Household Panel Survey 2010-2013-2016, Survey ID `MWI_2010-2016_IHPS_v03_M` ([link](https://microdata.worldbank.org/index.php/catalog/2939))
- "Nigeria 2010": Nigeria General Household Survey, Panel 2010-2011, Wave 1, Survey ID `NGA_2010_GHSP-W1_V03_M` ([link](https://microdata.worldbank.org/index.php/catalog/1002))
- "Nigeria 2015": Nigeria General Household Survey, Panel 2015-2016, Wave 3, Survey ID ` NGA_2015_GHSP-W3_v02_M` ([link](https://microdata.worldbank.org/index.php/catalog/2734/))
- "Tanzania 2008": Tanzania National Panel Survey 2008-2009, Wave 1, Survey ID `TZA_2008_NPS-R1_v03_M` ([link](https://microdata.worldbank.org/index.php/catalog/76))
- "Tanzania 2012": Tanzania National Panel Survey 2012-2013, Wave 3, Survey ID `TZA_2012_NPS-R3_v01_M` ([link](https://microdata.worldbank.org/index.php/catalog/2252))
- "Uganda 2005" and "Uganda 2009": Uganda National Panel Survey 2005-2009, Survey ID `UGA_2005-2009_UNPS_v01_M` ([link](https://microdata.worldbank.org/index.php/catalog/1001))
- "Uganda 2013": Uganda National Panel Survey 2013-2014, Survey ID `UGA_2013_UNPS_v01_M` ([link](https://microdata.worldbank.org/index.php/catalog/2663))


## Python Pickle (`.pkl`) Files

[**`dhs_incountry_folds.pkl`**](./dhs_incountry_folds.pkl): This Python Pickle file contains a Python dictionary representing the "in-country" cross-validation folds for DHS clusters. See [`preprocessing/2_create_incountry_folds.ipynb`](../preprocessing/2_create_incountry_folds.ipynb) for more details.


[**`lsms_incountry_folds.pkl`**](./lsms_incountry_folds.pkl): Similar to `dhs_incountry_folds.pkl`, except for LSMS clusters.


## Files in `data/overpass/`

[**`dhs_sample_GEE.csv`**](./overpass/dhs_sample_GEE.csv): Number of times various satellites imaged each of 500 clusters locations randomly sampled from DHS surveys per year from 2000 to 2018, inclusive. The images from satellites in this file (Landsat-5/7/8, MODIS, and Sentinel-1/2) are publicly accessible.
- TODO(ztang): include script used to generate this file

column         | description
---------------|------------
`system.index` | index variable used by Google Earth Engine
`LONGNUM`      | longitude
`cluster_id`   | DHS survey and cluster ID
`num_l5`       | number of times Landsat-5 satellite imaged the location in the given year
`num_l7`       | number of times Landsat-7 satellite imaged the location in the given year
`num_l8`       | number of times Landsat-8 satellite imaged the location in the given year
`num_modis`    | number of times MODIS satellite imaged the location in the given year
`num_s1`       | number of times Sentinel-1 satellite imaged the location in the given year
`num_s2`       | number of times Sentinel-2 satellite imaged the location in the given year
`.geo`         | GeoJSON representation of the sampled location
`year`         | year


[**`dhs_sample_Planet.csv`**](./overpass/dhs_sample_Planet.csv): Number of times the PlanetScope and RapidEye satellites from Planet Labs imaged each of 500 clusters locations randomly sampled from DHS surveys per year from 2000 to 2018, inclusive. The locations are the same as in the `dhs_sample_GEE.csv` file.
- TODO(ztang): include script used to generate this file

column              | description
--------------------|------------
`year`              | year
`count_PlanetScope` | number of times PlanetScope satellite imaged the location in the given year
`count_RapidEye`    | number of times RapidEye satellite imaged the location in the given year
`cluster_id`        | DHS survey and cluster ID


[**`landinfo_dhs_sample_nocatalog.csv`**](./overpass/landinfo_dhs_sample_nocatalog.csv): Record of various other satellites imaging each of 500 clusters locations randomly sampled from DHS surveys per year from 2000 to 2018, inclusive. The locations are the same as in the `dhs_sample_GEE.csv` file. Includes the following satellites:
- DigitalGlobe: IKONOS (≤1.06m), QuickBird-2 (≤0.85m), GeoEye-1 (≤0.61m), WorldView-1/2/3/4 (≤0.70m/0.63m/0.42m/0.40m)
- Airbus: Pléiades-1A/1B (0.50m), SPOT-6/7 (1.50m)
- Planet Labs: SkySat (≤1.20m)
- KOMPSAT-2/3/3A (1.00m/0.70m/0.30m)
- TripleSat (1.00m)

TODO(ztang): include script used to generate this file

column        | description
--------------|------------
`sensor`      | name of the satellite
`date`        | date of satellite image, in format `dd-mmm-yy`
`resolution`  | resolution per pixel
`cloud`       | estimated cloud cover
`off-nadir`   | n/a
`sun`         | n/a
`stereo-pair` | n/a
`name`        | n/a
`cluster_id`  | DHS survey and cluster ID
`lon`         | longitude
`lat`         | latitude


## Files in `data/survey/`

[**`crosswalk_countries.csv`**](./surveys/crosswalk_countries.csv): "crosswalk" between ISO3, country names, and country names used in prediction data. Used in code for various figures.

column          | description
----------------|------------
`iso3`          | ISO 3166-1 alpha-3 ("ISO 3") country code (as used by `population_time.csv`)
`country`       | country names (as used by `dhs_time.csv` and `povcal_time_pop.csv`)
`country_pred`  | TODO: unclear


[**`dhs_time.csv`**](./surveys/dhs_time.csv): Number of individuals in each African country surveyed each year by DHS in nationally-representative asset wealth surveys, for completed surveys started between 2000 and 2016, inclusive. A DHS survey is considered to be a nationally-representative asset wealth survey if it was listed under the [DHS wealth index page](https://dhsprogram.com/topics/wealth-index/Wealth-Index-Construction.cfm) (accessed on [May 21, 2020](https://web.archive.org/web/20200521053030/https://dhsprogram.com/topics/wealth-index/Wealth-Index-Construction.cfm)) or if its survey report indicated as such. We gathered the list of all non-SPA surveys from the [DHS Survey Search website](https://dhsprogram.com/What-We-Do/Survey-Search.cfm) (accessed on [May 20, 2020](https://web.archive.org/web/20200521011046/https://dhsprogram.com/what-we-do/survey-search.cfm?sendsearch=1&sur_status=Completed&YrFrom=2000&YrTo=2020&str1=10,27,43,76,52,3,50,51,4,100,5,59,53,60,243,7,118,12,220,65,66,129,14,67,20,160,22,23,24,25,68,61,28,29,30,62,35,205,36,208,55,38,39,41,44,47,48,,&str2=1,2,3,17,4,7,8,9,13,18,16,,&crt=1&listview=2&listgrp=0)) and removed the three surveys that did not include a nationally-representative survey of asset wealth: Central African Republic MICS 2010, Mauritania Special 2003-04, and Sao Tome and Principe MICS 2014. Country names were changed to match those in `crosswalk_countries.csv`. Used in [`figs/fig_1_surveyrates.R`](../figs/fig_1_surveyrates.R).

> The original version of this file used in the paper ([`data/surveys/paper/dhs_time.csv`](./surveys/paper/dhs_time.csv)) was incomplete, missed some surveys, and had some inaccurate numbers. See the [errata](../errata.md). Used in [`figs/fig_1_surveyrates_paper.R`](../figs/fig_1_surveyrates_paper.R).


[**`population_time.csv`**](./surveys/population_time.csv): Annual population estimates for each country between 1960 and 2017, inclusive. Downloaded from the World Bank, variable "Population, total (SP.POP.TOTL)", version "2018 Oct". See the World Bank World Development Indicators Database Archives: [https://datacatalog.worldbank.org/dataset/wdi-database-archives](https://datacatalog.worldbank.org/dataset/wdi-database-archives). Used in [`figs/fig_1_surveyrates.R`](../figs/fig_1_surveyrates.R).


[**`povcal_time_pop.csv`**](./surveys/povcal_time_pop.csv): Number of individuals in each country surveyed each year for the World Bank's PovcalNet. Compiled from [http://iresearch.worldbank.org/PovcalNet/povOnDemand.aspx](http://iresearch.worldbank.org/PovcalNet/povOnDemand.aspx) by finding the number of observations in each "detailed output" for each survey. For type "C" surveys (where consumption is measured at a group level and individual sample sizes aren't reported), "-1" is recorded to indicate there was a survey but no sample size is given. Used in [`figs/fig_1_surveyrates.R`](../figs/fig_1_surveyrates.R).

> TODO: describe the `./paper/` version. See the [errata](../errata.md). Used in [`figs/fig_1_surveyrates_paper.R`](../figs/fig_1_surveyrates_paper.R).


[**`us_surveys_time.csv`**](./surveys/us_surveys_time.csv): Number of people sampled in surveys as pulled from the following surveys. All sample numbers that are household counts are multiplied by the mean household size in that year, as found at [https://www.census.gov/data/tables/time-series/demo/families/households.html](https://www.census.gov/data/tables/time-series/demo/families/households.html). Used in [`figs/fig_1_surveyrates.R`](../figs/fig_1_surveyrates.R).
- ACS: https://www.census.gov/acs/www/methodology/sample-size-and-data-quality/sample-size/index.php
    calculated by summing the Final Interviews and Final Actual Interviews columns
- AHS until 2015: https://www.census.gov/content/dam/Census/programs-surveys/ahs/publications/AHS%20Sample%20Determination%20and%20Decisions.pdf
- AHS in 2017: https://www.census.gov/programs-surveys/ahs/about/methodology.html
- AHS in 2019: https://www.reginfo.gov/public/do/PRAViewDocument?ref_nbr=201810-2528-002
- CPS: http://www.census.gov/prod/2006pubs/tp-66.pdf
- NSCG: https://www.nsf.gov/statistics/srvygrads/overview.htm
- PSID 1: https://psidonline.isr.umich.edu/publications/Papers/tsp/2000-04_Imm_Sample_Addition.pdf
- PSID 2: https://nsf.gov/news/special_reports/survey/index.jsp?id=income (interpolated between)
- SIPP 1993: https://www2.census.gov/prod2/sipp/wp/SIPP_WP_203.pdf
- SIPP: http://www.nber.org/sipp/2008/ch2_nov20.pdf. For 2014 sample we assume same size as 2008.

> TODO: describe the `./paper/` version. See the [errata](../errata.md). Used in [`figs/fig_1_surveyrates_paper.R`](../figs/fig_1_surveyrates_paper.R).


## GPS coordinate displacement

In DHS surveys, all households within a single cluster are assigned the same GPS coordinates. (A *cluster* roughly corresponds to a village in rural regions and a neighborhood in urban regions.) Furthermore, DHS surveys displace the true GPS coordinates of each cluster in order to protect respondent confidentiality. According to the [DHS website](https://dhsprogram.com/What-We-Do/GPS-Data-Collection.cfm):

> Urban clusters contain a minimum of 0 and a maximum of 2 kilometers of error.
> Rural clusters contain a minimum of 0 and a maximum of 5 kilometers of positional error with a further 1% of the rural clusters displaced a minimum of 0 and a maximum of 10 kilometers.

The LSMS surveys have adopted the same location displacement procedure, as explained in the Basic Information Document for each of the LSMS surveys.
- Note: The Tanzania 2008 survey and Uganda 2005/2009 survey Basic Information Documents do not specify any location displacement procedure, but the location variables are described as "modified," which we assume to mean that they followed the same location displacement procedure as the other LSMS surveys.
- Note: The Uganda 2013 survey microdata does not provide geocoordinates, so the geocoordinates are matched by household ID to the Uganda 2005/2009 survey.
    - TODO: confirm this with Anne

For more information on the geographic displacement procedure, please consult the following reference:

> Burgert, Clara R., Josh Colston, Thea Roy, and Blake Zachary. 2013. *Geographic Displacement Procedure and Georeferenced Data Release Policy for the Demographic and Health Surveys*. DHS Spatial Analysis Reports No. 7. Calverton, Maryland, USA: ICF International. [https://dhsprogram.com/publications/publication-SAR7-Spatial-Analysis-Reports.cfm](https://dhsprogram.com/publications/publication-SAR7-Spatial-Analysis-Reports.cfm).


## Other notes about the data files

The (lat, lon) coordinates in the CSV files should be read according to the IEEE 32-bit floating point standard. For example, the CSV files can be read using the Python pandas library as follows:

```python
import pandas as pd

df = pd.read_csv('dhs_clusters.csv', float_precision='high', index_col=False)
for col in ['lat', 'lon']:
    df[col] = df[col].astype('float32')
```


## TODO

- For DHS, add district ID to each village
- Check that the links in this README work.
- Why do none of the values in the new us_surveys_time.csv match the paper version?
