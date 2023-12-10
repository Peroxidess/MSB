from copy import deepcopy
import numpy as np
from numpy.random.mtrand import random, seed
import pandas as pd


class MVI():
    def __init__(self, co_col, ca_col, seed):
        self.co_col = co_col
        self.ca_col = ca_col
        self.seed = seed
        np.random.seed(seed)
        self.model = Statistics(self.ca_col, self.co_col)

    @staticmethod
    def drop_nan_sample(data, label, ratio=0.9):
        nan_sample = data.isna().sum(axis=1)
        print(f'sample raw: {data.shape[0]}')
        nan_sample_ratio = nan_sample / data.shape[1]
        drop_nan_sample = nan_sample_ratio[nan_sample_ratio < ratio]
        data_dropnansample = data.loc[drop_nan_sample.index]
        label_dropnansample = label.loc[drop_nan_sample.index]
        print(f'sample curr: {data_dropnansample.shape[0]}')
        print(f'sample drop ratio: {1 - data_dropnansample.shape[0] / data.shape[0]}')
        return data_dropnansample, label_dropnansample

    @staticmethod
    def show_nan_ratio(data):
        sum_nan_all = data.isnull().sum().sum()
        sum_nan_ratio_micro = sum_nan_all / (data.shape[0] * data.shape[1])
        print(f'nan_ratio_micro: {sum_nan_ratio_micro}')
        return sum_nan_ratio_micro

    def fit_transform(self, data):
        data_randfillna = deepcopy(data)
        data_filled = self.model.fit_transform(data_randfillna)
        if type(data_filled) == np.ndarray:
            data_filled_ = pd.DataFrame(data_filled,
                                       index=data_randfillna.index,
                                       columns=data_randfillna.columns
                                       )
        else:
            data_filled_ = data_filled
        return data_filled_

    def transform(self, data):
        data_randfillna = data
        data_filled = self.model.transform(data_randfillna)
        data_filled = pd.DataFrame(data_filled, index=data.index, columns=data.columns)
        return data_filled


class Statistics():
    def __init__(self, ca_col, co_col):
        self.ca_col = ca_col
        self.co_col = co_col

    def fit_transform(self, data):
        data_filled = deepcopy(data)
        data_filled[self.ca_col] = data_filled[self.ca_col].fillna(data_filled[self.ca_col].mode(axis=0))
        data_filled[self.ca_col] = data_filled[self.ca_col].fillna(0)
        data_filled[self.co_col] = data[self.co_col].fillna(data[self.co_col].mean(axis=0))
        return data_filled

    def transform(self, data):
        data_filled = deepcopy(data)
        data_filled[self.ca_col] = data_filled[self.ca_col].fillna(data[self.ca_col].mode(axis=0))
        data_filled[self.ca_col] = data_filled[self.ca_col].fillna(0)
        data_filled[self.co_col] = data[self.co_col].fillna(data[self.co_col].mean(axis=0))
        return data_filled
