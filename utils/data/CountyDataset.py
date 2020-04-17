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
        # validate paths (some county data missing)
        valid_locals = [join(self.COUNTY_DIR, f)
                        for f in listdir(self.COUNTY_DIR)
                        if isfile(join(self.COUNTY_DIR, f))]
        county_ids = [int(splitext(basename(p))[0]) for p in valid_locals]
        valid_states = [join(self.STATE_DIR, f)
                        for f in listdir(self.STATE_DIR)
                        if isfile(join(self.STATE_DIR, f))]
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

    def __len__(self):
        return len(self.df_master)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read county data and merge with country
        county_id, state_id = self.df_master.loc[idx].values
        df_local = pd.read_csv(join(self.COUNTY_DIR, f'{county_id}.csv'),
                               dtype=np.float32, parse_dates=[0],
                               na_values='.').dropna()
        df_local = df_local.rename(
            columns={'Unnamed: 0': 'date'}).set_index('date')
        df_out = pd.merge(self.df_country, df_local, on='date', how='outer',
                          suffixes=['_country', '_local']).ffill().bfill()

        # read state data and merge with output
        df_state = pd.read_csv(join(self.STATE_DIR, f'{state_id}.csv'),
                               dtype=np.float32, parse_dates=[0],
                               na_values='.').dropna()
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

        # conditionally transform (standardize etc)
        if self.transform:
            X, Y = self.transform(X, Y)

        # conver tto tensors
        X = torch.tensor(X[:, :-self.Tfwd], dtype=torch.float)
        Y = torch.tensor(Y[-self.Tfwd:], dtype=torch.float)

        return X, Y
