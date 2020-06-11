# This script creates Figure S9 from the paper.
#
# Prerequisites: None.
#
# Errata: The method by which we assigned households to clusters was poorly
#   explained in Supplementary Note 1. There is no error with the code
#   itself. See errata.md for details.
#
# Libraries explicitly used:
# - dplyr

library(dplyr)

n = 1000  # number of households
xsd = 0.9  # sd of household features
nsd = 0.6  # sd of noise in cross section
mc = 0.08  # mean change in features
sdc = 0.25  # sd of change in features

rc1 = rc2 = rd1 = rc1w = rc2w = rd1w = c()  # vectors to hold results
for (i in 1:100) {

    x1 = rnorm(n, 0, xsd)  # features in year 1
    y1 = x1 + rnorm(n, 0, nsd)  # features + noise in year 1

    # generate cluster-specific change in features
    clustnum = data.frame(clust = round(rnorm(n, 25, 10)))  # generate cluster id for each obs, with different numbers of hholds in each cluster, row = household
    clust = clustnum %>% dplyr::group_by(clust) %>% dplyr::summarise(n = dplyr::n())  # cols: ["clust", "n"], row = cluster
    chg = data.frame(clust, chg = rnorm(dim(clust)[1], mc, sdc))  # generate a cluster specific change, cols ["clust", "n", "chg"], row = cluster
    clust = dplyr::left_join(clustnum, chg)  # cluster specific change in features related to wealth, cols ["clust", "n", "chg"], row = household
    x2 = x1 + clust$chg  # houseshold change in features
    y2 = x2 + rnorm(n, 0, nsd)  # features + noise in year 2
    yd = y2 - y1  # change in y

    xd = x2 - x1
    data = data.frame(hhold = 1:n, clust, x1, x2, y1, y2, yd, xd)

    cdata = data %>% dplyr::group_by(clust) %>% dplyr::summarise_all(mean)  # collapse to cluster level
    rc1 = c(rc1, summary(lm(y1 ~ x1, data = cdata))$r.squared)  # cross sectional regression in first year on observed
    rc2 = c(rc2, summary(lm(y2 ~ x2, data = cdata))$r.squared)  # cross sectional regression in second year on observed
    rd1 = c(rd1, summary(lm(yd ~ xd, data = cdata))$r.squared)  # over time regression on observed
    print(i)
}

dir.create(file.path("output"), showWarnings = FALSE)
pdf('output/fig_s9_simulation.pdf', width = 12, height = 4)
par(mfrow = c(1, 3))
hist(rc1, main = "cross section year 1", xlim = c(0, 1), las = 1, xlab = "r2")
abline(v = mean(rc1), col = "red", lty = 2, lwd = 2)
hist(rc2, main = "cross section year 2", xlim = c(0, 1), las = 1, xlab = "r2")
abline(v = mean(rc2), col = "red", lty = 2, lwd = 2)
hist(rd1, main = "deltas", xlim = c(0, 1), las = 1, xlab = "r2")
abline(v = mean(rd1), col = "red", lty = 2, lwd = 2)
dev.off()
