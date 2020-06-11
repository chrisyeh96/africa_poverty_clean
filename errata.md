This file contains a list of known errata in the code for the original published paper. These known errata have been fixed in the GitHub repo.


**Incorrect data on nationally-representative asset-wealth surveys in Africa and in the US**
- data/surveys/dhs_time.csv:
    - lists "Congo", "Democratic Republic of the Congo", and "Republic of the Congo", but there are only 2 "Congos" in the world
    - If one of the "Congos" was an accidental repeat, then we would expect that two of the rows would be identical (except in the country name). Oddly, none of the three "Congos" listed have identical rows.


### Figure 1

**Figure 1a**
- Used the incorrect data on nationally-representative asset-wealth surveys.
- Incorrectly counted 16 years (instead of 17 years) in the numerator for the frequency of surveys between 2000 and 2016, inclusive.

**Figure 1b**
- Used the incorrect data on nationally-representative asset-wealth surveys.
- For DigitalGlobe satellites, it incorrectly filtered the cloud cover percentage.
- For DigitalGlobe satellites, incorrectly reported the satellite resolution as "<1m" when it should be "≤1.06m"
- The line indicating DigitalGlobe satellites mistakenly counted non-DigitalGlobe satellites as well.


### Supplementary Note 1

Supplementary Note 1 states: "Each village has *n_c* households ranging from 1 to 50." This statement is misleading because it does not accurately describe how we assigned households to villages. Due to how we randomly assigned the 1000 households to villages (also called "clusters"), some villages may in fact be assigned more than 50 households; although extremely unlikely, all 1000 households could be assigned to the same village. Instead, this sentence should read: "Each household is assigned to a village *c* where *c* is drawn from a Gaussian distribution with standard deviation 10, rounded to the nearest integer."

For each of the 1000 households, we sample a cluster ID *c* from Gaussian distribution with standard deviation 10, rounded to the nearest integer. This means that the number *n_c* of households assigned to each cluster ID *c* follows a `Binomial(n=1000, p=p_c)` distribution, where `p_c = 1000 * normalcdf(mean=25, std=10, lower=c-0.5, upper=c+0.5)`. Therefore, *n_c*'s support is over all integers between 0 and 1000, inclusive.

See `figs/fig_s9_simulation.R` for details.
