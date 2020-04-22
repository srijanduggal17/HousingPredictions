from os import listdir
from os.path import (
    join, isfile,
    splitext, basename)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CountyDataset(Dataset):
    DATA_DIR = 'data'
    COUNTY_DIR = 'data/county'
    STATE_DIR = 'data/state'
    COUNTRY_DIR = 'data/country'

    def __init__(self, master_path, country='usa',
                 target_col='All-Transactions House Price Index',
                 Tback=20, Tfwd=10, transform=None):
        # validate paths for missing data
        valid_locals = self.get_valid_paths(path_type='county')
        county_ids = [int(splitext(basename(p))[0]) for p in valid_locals]

        valid_states = self.get_valid_paths(path_type='state')
        state_ids = [int(splitext(basename(p))[0]) for p in valid_states]

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
        self.transform = transform
        self.xcols = None
        self.date_index = None

    def __len__(self):
        return len(self.df_master)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read county data and merge with country
        county_id, state_id = self.df_master.loc[idx].values
        df_local = pd.read_csv(join(self.COUNTY_DIR, f'{county_id}.csv'),
                               dtype=np.float32, parse_dates=[0],
                               na_values='.').ffill().bfill()
        df_local = df_local.rename(
            columns={'Unnamed: 0': 'date'}).set_index('date')
        df_out = pd.merge(self.df_country, df_local, on='date', how='outer',
                          suffixes=['_country', '_local']).ffill().bfill()

        # read state data and merge with output
        df_state = pd.read_csv(join(self.STATE_DIR, f'{state_id}.csv'),
                               dtype=np.float32, parse_dates=[0],
                               na_values='.').ffill().bfill()
        df_state = df_state.rename(
            columns={'Unnamed: 0': 'date'}).set_index('date')

        # merge with output data
        df_out = pd.merge(df_out, df_state, on='date',
                          how='outer').ffill().bfill()

        # split features and target into np arrays
        if self.xcols is None:
            self.xcols = [c for c in df_out.columns.values
                          if c != self.target_col]
        X = df_out.loc[:, df_out.columns.isin(self.xcols)].values
        Y = df_out[self.target_col].values

        # store period
        if self.date_index is None:
            self.date_index = df_out.index

        # conditionally transform (standardize etc)
        if self.transform:
            X, Y = self.transform(X, Y)

        # convert to tensors
        X = torch.tensor(X[:-self.Tfwd, :], dtype=torch.float)
        Y = torch.tensor(Y[-self.Tfwd:], dtype=torch.float)

        return X, Y

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

    def get_dfs(self):
        dfs = []
        for i in range(self.__len__()):
            # read county data and merge with country
            county_id, state_id = self.df_master.loc[i].values
            df_local = pd.read_csv(join(self.COUNTY_DIR, f'{county_id}.csv'),
                                   dtype=np.float32, parse_dates=[0],
                                   na_values='.').ffill().bfill()
            df_local = df_local.rename(
                columns={'Unnamed: 0': 'date'}).set_index('date')
            df_out = pd.merge(self.df_country, df_local,
                              on='date', how='outer',
                              suffixes=['_country', '_local']).ffill().bfill()

            # read state data and merge with output
            df_state = pd.read_csv(join(self.STATE_DIR, f'{state_id}.csv'),
                                   dtype=np.float32, parse_dates=[0],
                                   na_values='.').ffill().bfill()
            df_state = df_state.rename(
                columns={'Unnamed: 0': 'date'}).set_index('date')

            # merge with output data
            df_out = pd.merge(df_out, df_state, on='date',
                              how='outer').ffill().bfill()

            # split features and target into np arrays
            if self.xcols is None:
                self.xcols = [c for c in df_out.columns.values
                              if c != self.target_col]
            df_out = df_out[self.xcols+[self.target_col]]
            dfs.append(df_out)

        return dfs
