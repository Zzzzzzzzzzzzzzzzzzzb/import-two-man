import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
import joblib

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, size=None,
                 features='MS', data_path='ETTh1.csv', N=-1,
                 target='OT', scale=True, timeenc=1, freq='1D', index=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.source_index = index
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        cols.remove('id')

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            joblib.dump(self.scaler, rf'scalar\scalar_{self.freq}')
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # self.data_x = data[border1:border2]
        self.data_x = data
        self.data_y = data
        # self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = self.source_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.source_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Tide(Dataset):
    def __init__(self, root_path, size=None,
                 features='MS', data_path='ETTh1.csv', N=-1,
                 target='OT', scale=True, timeenc=1, freq='1D', index=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.N = N

        self.root_path = root_path
        self.data_path = data_path
        self.source_index = index
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.fillna(0, inplace=True)
        cols = list(df_raw.columns)

        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        cols.remove('id')

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date', 'id'] + cols]
            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]
        else:
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_data.values)
            joblib.dump(self.scaler, rf'scalar\scalar_{self.data_path[:-4]}')
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        self.time_stamp = df_stamp
        self.cols = ['date', 'id'] + cols[:self.N]

        self.attributes = df_raw.iloc[:, 1].values
        self.seq_x = data[:, :self.N]
        self.covariates = data[:, self.N:]

    def __getitem__(self, index):
        s_begin = self.source_index[index]
        s_end = s_begin + self.seq_len
        seq_x = self.seq_x[s_begin: s_end]
        seq_y = self.seq_x[s_end: s_end + self.pred_len]
        covariates = self.covariates[s_begin: s_end + self.pred_len]
        attributes = self.attributes[s_begin]

        return seq_x, seq_y, covariates, attributes

    def __len__(self):
        return len(self.source_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv', N=-1,
                 target='OT', scale=True, inverse=False,
                 timeenc=1, freq='1D', cols=None, index=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.root_path = root_path
        self.data_path = data_path
        self.source_index = index
        self.__read_data__()

    def __read_data__(self):
        self.scaler = joblib.load(f'scalar/scalar_1D')        # self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        self.source_index = np.arange(0, len(df_raw)-self.seq_len)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
            cols.remove('id')
        if self.features == 'S':
            cols.remove(self.target)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date', 'id'] + cols]
            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            # self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp = df_raw[['date']]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[self.seq_len+1], periods=len(df_raw) - self.seq_len + self.pred_len, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values)
        self.future_dates = list(pred_dates)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.attributes = df_raw.iloc[:, 1].values
        self.time_stamp = df_stamp[['date']]
        self.cols = ['date', 'id'] + cols

        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = self.source_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return len(self.data_x) - self.seq_len + 1
        return len(self.source_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Tide_Pred(Dataset):
    def __init__(self, root_path, size=None,
                 features='M', data_path='ETTh1.csv', N=-1,
                 target='OT', scale=True, timeenc=1, freq='1D', index=None, cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.N = N

        self.root_path = root_path
        self.data_path = data_path
        self.source_index = index
        self.cols = cols
        self.__read_data__()

    def __read_data__(self):
        self.scaler = joblib.load(f'scalar\scalar_train_and_covariates_2')
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.fillna(0, inplace=True)
        cols = list(df_raw.columns)

        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        cols.remove('id')

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date', 'id'] + cols]
            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]
        else:
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        self.time_stamp = df_stamp
        self.cols = ['date', 'id'] + cols[:self.N]

        self.attributes = df_raw.iloc[:, 1].values
        self.seq_x = data[:, :self.N]
        self.covariates = data[:, self.N:]

    def __getitem__(self, index):
        s_begin = self.source_index[index]
        s_end = s_begin + self.seq_len
        seq_x = self.seq_x[s_begin: s_end]
        seq_y = self.seq_x[s_end: s_end + self.pred_len]
        covariates = self.covariates[s_begin: s_end + self.pred_len]
        attributes = self.attributes[s_begin]

        return seq_x, seq_y, covariates, attributes

    def __len__(self):
        return len(self.source_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)