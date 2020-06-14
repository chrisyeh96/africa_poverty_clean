DHS_COUNTRIES = [
    'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
    'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
    'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
    'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']

LSMS_COUNTRIES = ['ethiopia', 'malawi', 'nigeria', 'tanzania', 'uganda']

_SURVEY_NAMES_5country = {
    'train': ['uganda_2011', 'tanzania_2010', 'rwanda_2010', 'nigeria_2013'],
    'val': ['malawi_2010']
}

_SURVEY_NAMES_DHS_OOC_A = {
    'train': ['cameroon', 'democratic_republic_of_congo', 'ghana', 'kenya',
              'lesotho', 'malawi', 'mozambique', 'nigeria', 'senegal',
              'togo', 'uganda', 'zambia', 'zimbabwe'],
    'val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    'test': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
}
_SURVEY_NAMES_DHS_OOC_B = {
    'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
              'ethiopia', 'kenya', 'lesotho', 'mali', 'mozambique',
              'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
    'val': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
    'test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
}
_SURVEY_NAMES_DHS_OOC_C = {
    'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
              'guinea', 'kenya', 'lesotho', 'mali', 'rwanda', 'senegal',
              'sierra_leone', 'tanzania', 'zambia'],
    'val': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
    'test': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
}
_SURVEY_NAMES_DHS_OOC_D = {
    'train': ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
              'ethiopia', 'ghana', 'guinea', 'malawi', 'mali', 'rwanda',
              'sierra_leone', 'tanzania', 'zimbabwe'],
    'val': ['kenya', 'lesotho', 'senegal', 'zambia'],
    'test': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
}
_SURVEY_NAMES_DHS_OOC_E = {
    'train': ['benin', 'burkina_faso', 'cameroon', 'democratic_republic_of_congo',
              'ghana', 'guinea', 'malawi', 'mozambique', 'nigeria', 'sierra_leone',
              'tanzania', 'togo', 'uganda', 'zimbabwe'],
    'val': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
    'test': ['kenya', 'lesotho', 'senegal', 'zambia'],
}

SURVEY_NAMES = {  # TODO: rename to SURVEY_NAMES_DHS?
    '5country': _SURVEY_NAMES_5country,  # TODO: is this needed
    'DHS_OOC_A': _SURVEY_NAMES_DHS_OOC_A,
    'DHS_OOC_B': _SURVEY_NAMES_DHS_OOC_B,
    'DHS_OOC_C': _SURVEY_NAMES_DHS_OOC_C,
    'DHS_OOC_D': _SURVEY_NAMES_DHS_OOC_D,
    'DHS_OOC_E': _SURVEY_NAMES_DHS_OOC_E
}

SURVEY_NAMES_LSMS = ['ethiopia_2011', 'ethiopia_2015', 'malawi_2010', 'malawi_2016',
                      'nigeria_2010', 'nigeria_2015', 'tanzania_2008', 'tanzania_2012',
                      'uganda_2005', 'uganda_2009', 'uganda_2013']

SIZES = {
    'DHS': {'train': 12319, 'val': 3257, 'test': 4093, 'all': 19669},  # TODO: is this needed?
    'DHSNL': {'all': 260415},
    'DHS_OOC_A': {'train': 11797, 'val': 3909, 'test': 3963, 'all': 19669},
    'DHS_OOC_B': {'train': 11820, 'val': 3940, 'test': 3909, 'all': 19669},
    'DHS_OOC_C': {'train': 11800, 'val': 3929, 'test': 3940, 'all': 19669},
    'DHS_OOC_D': {'train': 11812, 'val': 3928, 'test': 3929, 'all': 19669},
    'DHS_OOC_E': {'train': 11778, 'val': 3963, 'test': 3928, 'all': 19669},
    'DHS_incountry_A': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_B': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_C': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_D': {'train': 11802, 'val': 3933, 'test': 3934, 'all': 19669},
    'DHS_incountry_E': {'train': 11802, 'val': 3934, 'test': 3933, 'all': 19669},
    'LSMSincountry': {'train': 1812, 'val': 604, 'test': 604, 'all': 3020},  # TODO: is this needed?
    'LSMS': {'ethiopia_2011': 327, 'ethiopia_2015': 327, 'malawi_2010': 102,
             'malawi_2016': 102, 'nigeria_2010': 480, 'nigeria_2015': 480,
             'tanzania_2008': 300, 'tanzania_2012': 300, 'uganda_2005': 165,
             'uganda_2009': 165, 'uganda_2013': 165},
}

URBAN_SIZES = {
    'DHS': {'train': 3954, 'val': 1212, 'test': 1635, 'all': 6801},
    'DHS_OOC_A': {'train': 4264, 'val': 1221, 'test': 1316, 'all': 6801},
    'DHS_OOC_B': {'train': 4225, 'val': 1355, 'test': 1221, 'all': 6801},
    'DHS_OOC_C': {'train': 4010, 'val': 1436, 'test': 1355, 'all': 6801},
    'DHS_OOC_D': {'train': 3892, 'val': 1473, 'test': 1436, 'all': 6801},
    'DHS_OOC_E': {'train': 4012, 'val': 1316, 'test': 1473, 'all': 6801},
}

RURAL_SIZES = {
    'DHS': {'train': 8365, 'val': 2045, 'test': 2458, 'all': 12868},
    'DHS_OOC_A': {'train': 7533, 'val': 2688, 'test': 2647, 'all': 12868},
    'DHS_OOC_B': {'train': 7595, 'val': 2585, 'test': 2688, 'all': 12868},
    'DHS_OOC_C': {'train': 7790, 'val': 2493, 'test': 2585, 'all': 12868},
    'DHS_OOC_D': {'train': 7920, 'val': 2455, 'test': 2493, 'all': 12868},
    'DHS_OOC_E': {'train': 7766, 'val': 2647, 'test': 2455, 'all': 12868},
}

# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands

_MEANS_DHS = {
    'BLUE':  0.059183,
    'GREEN': 0.088619,
    'RED':   0.104145,
    'SWIR1': 0.246874,
    'SWIR2': 0.168728,
    'TEMP1': 299.078023,
    'NIR':   0.253074,
    'DMSP':  4.005496,
    'VIIRS': 1.096089,
    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
}
_MEANS_DHSNL = {
    'BLUE':  0.063927,
    'GREEN': 0.091981,
    'RED':   0.105234,
    'SWIR1': 0.235316,
    'SWIR2': 0.162268,
    'TEMP1': 298.736746,
    'NIR':   0.245430,
    'DMSP':  7.152961,
    'VIIRS': 2.322687,
}
_MEANS_LSMS = {
    'BLUE':  0.062551,
    'GREEN': 0.090696,
    'RED':   0.105640,
    'SWIR1': 0.242577,
    'SWIR2': 0.165792,
    'TEMP1': 299.495280,
    'NIR':   0.256701,
    'DMSP':  5.105815,
    'VIIRS': 0.557793,
}

_STD_DEVS_DHS = {
    'BLUE':  0.022926,
    'GREEN': 0.031880,
    'RED':   0.051458,
    'SWIR1': 0.088857,
    'SWIR2': 0.083240,
    'TEMP1': 4.300303,
    'NIR':   0.058973,
    'DMSP':  23.038301,
    'VIIRS': 4.786354,
    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
}
_STD_DEVS_DHSNL = {
    'BLUE':  0.023697,
    'GREEN': 0.032474,
    'RED':   0.051421,
    'SWIR1': 0.095830,
    'SWIR2': 0.087522,
    'TEMP1': 6.208949,
    'NIR':   0.071084,
    'DMSP':  29.749457,
    'VIIRS': 14.611589,
}
_STD_DEVS_LSMS = {
    'BLUE':  0.023979,
    'GREEN': 0.032121,
    'RED':   0.051943,
    'SWIR1': 0.088163,
    'SWIR2': 0.083826,
    'TEMP1': 4.678959,
    'NIR':   0.059025,
    'DMSP':  31.688320,
    'VIIRS': 6.421816,
}

MEANS_DICT = {
    'DHS': _MEANS_DHS,
    'DHSNL': _MEANS_DHSNL,
    'LSMS': _MEANS_LSMS,
}

STD_DEVS_DICT = {
    'DHS': _STD_DEVS_DHS,
    'DHSNL': _STD_DEVS_DHSNL,
    'LSMS': _STD_DEVS_LSMS,
}
