'''
This file adds GADM administrative region IDs as columns to CSV files containing
DHS and LSMS cluster locations.

Prerequisites: download shapefiles. See
  `preprocessing/3_download_gadm_shapefiles.sh`.
'''
import geopandas as gpd
import numpy as np
import pandas as pd

import os


def merge_df_shp(df: pd.DataFrame, shp_path: str, level: int = 2) -> gpd.GeoDataFrame:
    '''
    Args
    - df: pd.DataFrame, representing a single country
    - shp_path: str, path to shapefile
    - level: int, administrative region level, either 1 or 2

    Returns
    - merged: gpd.GeoDataFrame
    '''
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='epsg:4326')

    gid_cols = ['GID_1', 'GID_2']
    gid_cols = gid_cols[:level]

    shp = gpd.read_file(shp_path)[gid_cols + ['geometry']]

    merged = gpd.sjoin(gdf, shp, how='left', op='within').drop('index_right', axis=1)

    # for points that aren't contained within a region, match the points to the
    # nearest region
    missing = merged[merged.isna().any(axis=1)]
    if len(missing) > 0:
        print('Missing:', len(missing))

        for i in missing.index:
            shp_idx = shp.distance(missing.loc[i, 'geometry']).argmin()
            merged.loc[i, gid_cols] = shp.loc[shp_idx, gid_cols]

    for col in gid_cols:
        merged[col] = merged[col].map(lambda s: s.split('_')[0])

    return merged


def match_dhs_gids():
    sort_cols = ['country', 'year', 'lat', 'lon']

    cross_path = '../data/surveys/crosswalk_countries.csv'
    cross = pd.read_csv(cross_path)
    cross = cross.loc[cross.country_pred.notna(), ['iso3', 'country_pred']]
    country_to_iso3 = cross.set_index('country_pred')['iso3'].to_dict()

    dhs_path = '../data/dhs_clusters.csv'
    dhs = pd.read_csv(dhs_path, float_precision='high')
    dhs.sort_values(sort_cols, inplace=True)
    dhs = dhs[['country', 'year', 'lat', 'lon', 'wealthpooled', 'households', 'urban_rural']]

    shp2_path_template = '../data/shapefiles/gadm36_{iso3}_shp/gadm36_{iso3}_2.shp'
    shp1_path_template = '../data/shapefiles/gadm36_{iso3}_shp/gadm36_{iso3}_1.shp'

    country_dfs = []
    for country in np.sort(dhs.country.unique()):
        print(country)

        country_df = dhs.loc[dhs.country == country]
        iso3 = country_to_iso3[country]
        shp_path = shp2_path_template.format(iso3=iso3)
        level = 2

        if not os.path.exists(shp_path):
            shp_path = shp1_path_template.format(iso3=iso3)
            level = 1

        country_df = merge_df_shp(df=country_df, shp_path=shp_path, level=level)
        country_dfs.append(country_df)

    dhs = pd.concat(country_dfs).sort_values(sort_cols)
    dhs = dhs[['country', 'year', 'lat', 'lon', 'GID_1', 'GID_2', 'wealthpooled', 'households', 'urban_rural']]

    # for Lesotho, which doesn't have level-2 regions, set the level-2 ID to be the level-1 ID
    dhs.loc[dhs.GID_2.isna(), 'GID_2'] = dhs.loc[dhs.GID_2.isna(), 'GID_1']

    dhs.to_csv(dhs_path, index=False)


if __name__ == '__main__':
    match_dhs_gids()
