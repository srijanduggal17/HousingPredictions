from os import listdir
from os.path import (
    join, isfile,
    splitext, basename)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

COUNTY_PCT_COLS = {
    'Population Estimate, Total (5-year estimate)': [
        'Population Estimate, Total, Not Hispanic or Latino, American Indian and Alaska Native Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Asian Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Black or African American Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Native Hawaiian and Other Pacific Islander Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Some Other Race Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Two or More Races (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Two or More Races, Two Races Excluding Some Other Race, and Three or More Races (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, Two or More Races, Two Races Including Some Other Race (5-year estimate)',  # noqa
        'Population Estimate, Total, Not Hispanic or Latino, White Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, American Indian and Alaska Native Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Asian Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Black or African American Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Native Hawaiian and Other Pacific Islander Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Some Other Race Alone (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Two or More Races (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Two or More Races, Two Races Excluding Some Other Race, and Three or More Races (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, Two or More Races, Two Races Including Some Other Race (5-year estimate)',  # noqa
        'Population Estimate, Total, Hispanic or Latino, White Alone (5-year estimate)',  # noqa
    ],
    'Poverty Universe, All Ages': [
        'Poverty Universe, Age 0-17',
        'Poverty Universe, Age 5-17 related'
    ],
    'Real Gross Domestic Product: All Industries': [
        'Real Gross Domestic Product: Government and Government Enterprises'
    ]
}
COUNTY_DROP_COLS = [
    '90% Confidence Interval Lower Bound of Estimate of People Age 0-17 in Poverty',  # noqa
    '90% Confidence Interval Lower Bound of Estimate of People of All Ages in Poverty',  # noqa
    '90% Confidence Interval Lower Bound of Estimate of Related Children Age 5-17 in Families in Poverty',  # noqa
    '90% Confidence Interval Upper Bound of Estimate of People Age 0-17 in Poverty',  # noqa
    '90% Confidence Interval Upper Bound of Estimate of People of All Ages in Poverty',  # noqa
    'Estimate of Median Household Income',
    'Estimate of People Age 0-17 in Poverty',
    'Estimated Percent of Related Children Age 5-17 in Families in Poverty',
    'Gross Domestic Product: All Industries',
    'Gross Domestic Product: Government and Government Enterprises',
    'Personal Income',
    'Population Estimate, Total, Hispanic or Latino (5-year estimate)',
    'Population Estimate, Total, Not Hispanic or Latino (5-year estimate)'
]
STATE_PCT_COLS = {
    'Net Earnings by Place of Residence in': [
        'Accommodation and Food Services Earnings in',
        'Administrative and Waste Services Earnings in',
        'Arts, Entertainment and Recreation Earnings in',
        'Construction Earnings in',
        'Durable Manufacturing Earnings in',
        'Educational Services Earnings in',
        'Information Earnings in',
        'Management of Companies and Enterprises Earnings in',
        'Nondurable Manufacturing Earnings in',
        'Other Services (except Public Administration) Earnings in',
        'Professional and Technical Services Earnings in',
        'Real Estate, Rental and Leasing Earnings in',
        'Retail Trade Earnings in',
        'Transportation and Warehousing Earnings in',
        'Utilities Earnings in',
        'Wholesale Trade Earnings in'
    ],
    'Total Wages and Salaries in': [
        'Accommodation and Food Services Wages and Salaries in',
        'Administrative and Waste Services Wages and Salaries in',
        'Arts, Entertainment and Recreation Wages and Salaries in',
        'Construction Wages and Salaries in',
        'Durable Manufacturing Wages and Salaries in',
        'Educational Services Wages and Salaries in',
        'Information Wages and Salaries in',
        'Management of Companies and Enterprises Wages and Salaries in',
        'Nondurable Manufacturing Wages and Salaries in',
        'Other Services (Except Public Administration) Wages and Salaries in',
        'Professional and Technical Services Wages and Salaries in',
        'Real Estate, Rental and Leasing Wages and Salaries in',
        'Retail Trade Wages and Salaries in',
        'Transportation and Warehousing Wages and Salaries in',
        'Utilities Wages and Salaries in',
        'Wholesale Trade Wages and Salaries in'
    ],
    'Resident Population in': [
        'Unemployment Level for',
        'Business Applications for',
        'Business Applications from Corporations for',
        'Business Applications with Planned Wages for',
        'Child Tax Exemptions for',
        'Commercial Carbon Dioxide Emissions, All Fuels for',
        'Continued Claims (Insured Unemployment) in',
        'Covered Employment in',
        'Dividends, Interest and Rent in',
        'Electric Power Carbon Dioxide Emissions, All Fuels for',
        'Employed Involuntary Part-Time for',
        'Employed Persons in',
        'Commercial Banks in'
    ],
    'Imports of Goods for': [
        'Imports of Goods: Non-Manufactured Commodities for',
        'Imports of Goods: Manufactured Commodities for'
    ],
    'Exports of Goods for': [
        'Exports of Goods: Non-Manufactured Commodities for',
        'Exports of Goods: Manufactured Commodities for'
    ],
    'Employed Persons in': [
        'All Employees: Education and Health Services in',
        'All Employees: Wholesale Trade in',
        'All Employees: Trade, Transportation, and Utilities in',
        'All Employees: Other Services in'
    ],
    'Average Weekly Hours of All Employees: Total Private in': [
        'Average Weekly Hours of Production Employees: Manufacturing in',
        'Average Weekly Hours of All Employees: Trade, Transportation, and Utilities in',  # noqa
        'Average Weekly Hours of All Employees: Professional and Business Services in',  # noqa
        'Average Weekly Hours of All Employees: Private Service Providing in',
        'Average Weekly Hours of All Employees: Leisure and Hospitality in',
        'Average Weekly Hours of All Employees: Goods Producing in'
    ],
    'Average Weekly Earnings of All Employees: Total Private in': [
        'Average Weekly Earnings of All Employees: Goods Producing in',
        'Average Weekly Earnings of All Employees: Leisure and Hospitality in',
        'Average Weekly Earnings of All Employees: Private Service Providing in',  # noqa
        'Average Weekly Earnings of All Employees: Professional and Business Services in',  # noqa
        'Average Weekly Earnings of All Employees: Trade, Transportation, and Utilities in',  # noqa
        'Average Weekly Earnings of Production Employees: Manufacturing in'
    ],
    'Allowance for Loan and Lease Losses for Commercial Banks in': [
        'Charge-offs on Allowance for Loan and Lease Losses for Commercial Banks in'  # noqa
    ],
    'New Private Housing Units Authorized by Building Permits for': [
        'New Private Housing Units Authorized by Building Permits: 1-Unit Structures for'  # noqa
    ],
    'Poverty Tax Exemptions for': [
        'Poverty Tax Exemptions Under Age 65 for'
    ],
    'Poverty Universe, All Ages for': [
        'Poverty Universe, Age 0-17 for',
        'Poverty Universe, Age 0-4 for',
        'Poverty Universe, Age 5-17 related for'
    ],
    'Real Per Capita Personal Income for': [
        'Real Per Capita Personal Income: Metropolitan Portion for',
        'Real Per Capita Personal Income: Nonmetropolitan Portion for'
    ],
    'Total Tax Exemptions for': [
        'Total Tax Exemptions Under Age 65 for'
    ]
}
STATE_DROP_COLS = [
    'All Employees: Transportation and Utilities: Transportation, Warehousing, and Utilities in',  # noqa
    'Civilian Labor Force for',
    'Loan Loss Reserves for Commercial Banks in',
    'Quarterly Average of Total Assets for Commercial Banks in',
    'Total Personal Income in',
    'Personal Consumption Expenditures: Services for',
    'Personal Consumption Expenditures: Goods for',
    'Real Personal Income for',
    'Real Personal Income: Metropolitan Portion for',
    'Real Personal Income: Nonmetropolitan Portion for',
    'Coefficient for Residential Carbon Dioxide Emissions, Residential for',
    'Residential Carbon Dioxide Emissions, Residential for'
]
FEDERAL_DROP_COLS = [
    'Federal Minimum Hourly Wage for Nonfarm Workers for the United States'
]


class CountyDataset(Dataset):
    # DATA_DIR = 'utils/data'
    # COUNTY_DIR = 'utils/data/county'
    # STATE_DIR = 'utils/data/state'
    # COUNTRY_DIR = 'utils/data/country'
    DATA_DIR = '/home/kbhakta/Dropbox (GaTech)/Georgia Tech Classes/Spring 2020/Deep Learning/Final_Project/data'
    COUNTY_DIR = '/home/kbhakta/Dropbox (GaTech)/Georgia Tech Classes/Spring 2020/Deep Learning/Final_Project/data/county'
    STATE_DIR = '/home/kbhakta/Dropbox (GaTech)/Georgia Tech Classes/Spring 2020/Deep Learning/Final_Project/data/state'
    COUNTRY_DIR = '/home/kbhakta/Dropbox (GaTech)/Georgia Tech Classes/Spring 2020/Deep Learning/Final_Project/data/country'

    def __init__(self, master_path, country='usa',
                 target_col='Zillow Price Index', Tback=20, Tfwd=10,
                 standardize=True, stats=None):
        # validate paths for missing data
        valid_locals = self.get_valid_paths(path_type='county')
        county_ids = [int(basename(p).split('_')[0]) for p in valid_locals]

        valid_states = self.get_valid_paths(path_type='state')
        state_ids = [int(splitext(basename(p))[0].split('_')[0]) for p in valid_states]

        # load master reference file and filter valid counties
        self.df_master = pd.read_csv(master_path,
                                     usecols=['county_id', 'state_id'],
                                     dtype=np.integer)
        self.df_master = self.df_master[
            self.df_master.county_id.isin(county_ids)]
        self.df_master = self.df_master[
            self.df_master.state_id.isin(state_ids)]
        self.df_master = self.df_master.reset_index(drop=True)

        self.target_col = target_col
        self.Tback = Tback
        self.Tfwd = Tfwd
        self.T = Tback+Tfwd
        self.df_country = pd.read_csv(join(self.COUNTRY_DIR, f'{country}.csv'),
                                      dtype=np.float32, index_col='date',
                                      parse_dates=['date'])
        self.df_country = self.df_country.drop(labels=FEDERAL_DROP_COLS,
                                               axis=1)

        self.xcols = None
        self.date_index = None

        self.standardize = standardize
        self.stats = None
        if standardize and stats is None:
            self.stats = self.get_stats()
        elif standardize:
            self.stats = stats

    def __len__(self):
        return len(self.df_master)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        df_out = self._get_df(idx)

        # store period
        if self.date_index is None:
            self.date_index = df_out.index

        # split features and target into np arrays
        if self.xcols is None:
            self.xcols = [c for c in df_out.columns.values
                          if c != self.target_col]
        X = df_out.loc[:, df_out.columns.isin(self.xcols)].values
        Y = df_out[self.target_col].values

        # conditionally transform (standardize etc)
        if self.standardize:
            (xmu, xstd), (ymu, ystd) = self.stats
            X, Y = (X-xmu.numpy())/xstd.numpy(), (Y-ymu.numpy())/ystd.numpy()

        # convert to tensors
        X = torch.tensor(X[:-self.Tfwd, :], dtype=torch.float)
        Y = torch.tensor(Y[-self.Tfwd:], dtype=torch.float)

        return X, Y

    def get_stats(self):
        if self.stats:
            return self.stats

        print(f'calculating stats...')
        tmp = self.standardize
        self.standardize = False
        X, Y = [], []
        for i in range(self.__len__()-1):
            x, y = self.__getitem__(i)
            X.append(x)
            Y.append(y)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        xmu, xstd = X.mean(dim=0), X.std(dim=0)
        ymu, ystd = Y.mean(dim=0), Y.std(dim=0)
        self.standardize = tmp
        return (xmu, xstd), (ymu, ystd)

    def _get_df(self, idx):
        county_id, state_id = self.df_master.loc[idx].values
        
        # merge county with country data
        df_local = self._get_local(county_id)
        df_out = pd.merge(self.df_country, df_local, on='date', how='outer',
                          suffixes=['_country', '_local']).ffill().bfill()

        # merge state with output data
        df_state = self._get_state(state_id)
        return pd.merge(df_out, df_state, on='date',
                        how='outer').ffill().bfill()

    def _get_local(self, county_id):
        # read county data and merge with country
        df_local = pd.read_csv(join(self.COUNTY_DIR,
                                    f'{county_id}_zillow.csv'),
                               dtype=np.float32, parse_dates=[0],
                               na_values='.').ffill().bfill()
        df_local = df_local.rename(
            columns={'Unnamed: 0': 'date'}).set_index('date')

        # drop county cols and update pct cols
        df_local = df_local.drop(labels=COUNTY_DROP_COLS, axis=1)
        for k, v in COUNTY_PCT_COLS.items():
            df_local[v] = df_local[v].div(df_local[k], axis=0)
        return df_local

    def _get_state(self, state_id):
        # read state data
        df_state = pd.read_csv(join(self.STATE_DIR, f'{state_id}_zillow.csv'),
                               dtype=np.float32, parse_dates=[0],
                               na_values='.').ffill().bfill()

        df_state = df_state.rename(
            columns={'Unnamed: 0': 'date'}).set_index('date')

        # drop state cols and update pct cols
        df_state = df_state.drop(labels=STATE_DROP_COLS, axis=1)
        for k, v in STATE_PCT_COLS.items():
            df_state[v] = df_state[v].div(df_state[k], axis=0)
        return df_state

    def get_valid_paths(cls, path_type='state', verbose=True):
        # valid path_types are 'state', 'county', 'country'
        if path_type == 'county':
            base_dir = cls.COUNTY_DIR
        elif path_type == 'state':
            base_dir = cls.STATE_DIR
        elif path_type == 'country':
            base_dir = cls.COUNTRY_DIR
        else:
            raise f'unk path_type {path_type}; must be [county|state|country]'

        if verbose:
            print('-'*89)
            print(f'{path_type.upper()} DATA SUMMARY')
            print('-'*89)
        
        all_paths = [join(base_dir, f)
                     for f in listdir(base_dir)
                     if isfile(join(base_dir, f))]
        has_nas = []
        periods, feats = {}, {}
        max_periods, max_feats = 0, 0
        for p in all_paths:
            df = pd.read_csv(p, dtype=np.float32, parse_dates=[0],
                             na_values='.').ffill().bfill()
            if df.isna().any(axis=1).sum() > 0:
                has_nas.append(p)
                if verbose:
                    print(f'missing feature --> dropping {path_type} '
                          f'| path {p} '
                          f'| feature {df.columns[df.isna().any()]}')

            period, feat = df.shape
            if feat > max_feats:
                max_feats = feat

            if period > max_periods:
                max_periods = period

            if period not in periods:
                periods[period] = 1
            else:
                periods[period] += 1

            if feat not in feats:
                feats[feat] = 1
            else:
                feats[feat] += 1

        valid_paths = []
        for p in all_paths:
            df = pd.read_csv(p, dtype=np.float32, parse_dates=[0],
                             na_values='.').ffill().bfill()
            period, feat = df.shape
            if period == max_periods and feat == max_feats \
                    and p not in has_nas:
                valid_paths.append(p)

        if verbose:
            print(f'valid count {len(valid_paths)} '
                  f'| periods {max_periods} '
                  f'| features {max_feats}')
        return valid_paths