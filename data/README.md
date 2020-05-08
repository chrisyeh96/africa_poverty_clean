## CSV Files

**dhs_clusters.csv**: This CSV file contains DHS survey data, aggregated to the cluster level. A cluster roughly corresponds to a village. The columns are as follows:

column        | description
--------------|------------
country       | country
year          | year that the survey started (some surveys lasted more than 1 year)
lat           | latitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
lon           | longitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
wealthpooled  | mean asset wealth index (AWI) of households within the cluster
households    | number of households surveyed in the cluster
urban_rural   | 0 = rural, 1 = urban

The AWI used in the `wealthpooled` index was computed as follows:
1. Take the PCA of household-level assets from 86 DHS surveys spanning 1994 to 2016. The value of the first principle component is the household-level asset wealth index (AWI).
2. Within each cluster, compute the mean household-level AWI to get the cluster-level AWI. We computed the AWI for 35,235 clusters.
3. Remove clusters from surveys started in 2008 or earlier (only keep surveys between 2009-2016, inclusive), and remove clusters whose GPS coordinates or urban/rural status were unknown. This leaves us with 19,669 clusters spanning 43 DHS surveys.


**lsms_clusters.csv**: This CSV file indicates the locations and years of the LSMS clusters. A cluster roughly corresponds to a village. The columns are as follows:

column        | description
--------------|------------
country       | country
year          | year that the survey started (some surveys lasted more than 1 year)
lat           | latitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
lon           | longitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
geolev1       | administrative level-1 region name
geolev2       | administrative level-2 region name


**lsms_diffs.csv**: This CSV file contains the changes in asset wealth index (AWI) over time for LSMS clusters. A cluster roughly corresponds to a village. The columns are as follows:

column        | description
--------------|------------
country       | country
lat           | latitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
lon           | longitude coordinate of the cluster, possibly displaced (see [below](#gps-coordinate-displacement))
year.x        | year that 1st survey started (some surveys lasted more than 1 year)
year.y        | year that 2nd survey started, `year.y` > `year.x`
diff_of_index | change in cluster-level AWI, see below
index_of_diff | cluster-level index of asset differences, see below
households    | number of households that were included in the calculation of the change in cluster-level AWI
geolev1       | administrative level-1 region name
geolev2       | administrative level-2 region name

For each pair of years, the `diff_of_index` value for a given cluster *C* is (cluster *C*'s AWI in year `year.y`) - (cluster *C*'s AWI in year `year.x`). Only households that were surveyed in both `year.x` and `year.y` were included in the mean. Here, the cluster-level AWI is computed in a similar fasion as for the DHS surveys, taking the PCA of the household assets and then taking the mean across households.

In contrast, the `index_of_diff` value for a given cluster is computed as follows:
1. For each household that exists across 2 surveys, compute the difference in that household's assets between the two surveys.
2. Compute the PCA of these asset-differences across the 5 countries for which we have LSMS surveys. The value of the first principle component is the household-level index of asset differences.
3. Within each cluster, compute the mean household-level index of asset differences to get the cluster-level index of asset differences.

Note that for the same cluster, the number of households used in one pair of years (`year.x`, `year.y`) might be different from the number of households in a different pairt of years. This is because even though LSMS attempts to be a panel survey, some households are not recorded at each survey. For example, a household surveyed in 2005 and 2009 rounds may have moved between 2009 and 2013. Thus, this household would be included in the cluster household count in (`year.x` = 2005, `year.y` = 2009) but not in (`year.x` = 2009, `year.y` = 2013).


## Python Pickle (`.pkl`) Files

**dhs_incountry_folds.pkl**: This Python Pickle file is created by [`preprocessing/2_create_incountry_folds.pkl`](../preprocessing/2_create_incountry_folds.pkl) and contains a Python dictionary representing the "in-country" cross-validation folds for DHS clusters. See [`preprocessing/2_create_incountry_folds.pkl`](../preprocessing/2_create_incountry_folds.pkl) for more details.

**lsms_incountry_folds.pkl**: Similar to `dhs_incountry_folds.pkl`, except for LSMS clusters.


## GPS coordinate displacement

DHS surveys displace the true GPS coordinates of respondents in order to protect their confidentiality. According to the [DHS website](https://dhsprogram.com/What-We-Do/GPS-Data-Collection.cfm):

> Urban clusters contain a minimum of 0 and a maximum of 2 kilometers of error.
> Rural clusters contain a minimum of 0 and a maximum of 5 kilometers of positional error with a further 1% of the rural clusters displaced a minimum of 0 and a maximum of 10 kilometers.

Recently, LSMS surveys have also adopted the same location displacement procedure. For more information, please consult the [*Geographic Displacement Procedure and Georeferenced Data Release Policy for the Demographic and Health Surveys*](https://dhsprogram.com/publications/publication-SAR7-Spatial-Analysis-Reports.cfm).

Burgert, Clara R., Josh Colston, Thea Roy, and Blake Zachary. 2013. *Geographic Displacement Procedure and Georeferenced Data Release Policy for the Demographic and Health Surveys*. DHS Spatial Analysis Reports No. 7. Calverton, Maryland, USA: ICF International.


## TODO

- For DHS, add district ID to each village
- Check that the links in this README work.
- For LSMS, check terminology on geolev1 / geolev2 description
- For LSMS, double-check GPS displacement policy
