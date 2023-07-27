import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def for_predict_Tide(train_path, covariates_path, predict_table_path, N, seq_len):
    df_for_predict = pd.read_csv(predict_table_path)\
            .rename(columns={'product_pid': 'id', 'transaction_date': 'date'})
    df_for_predict['date'] = pd.to_datetime(df_for_predict['date'], format='%Y%m%d')
    df = pd.read_csv(train_path)
    df_covariates = pd.read_csv(covariates_path)

    # future covariates part
    df_covariates['date'] = pd.to_datetime(df_covariates['date'])
    future_range = pd.date_range(start='2022-11-10', end='2022-11-23', freq='1D')
    future_range = future_range.drop(['2022-11-12', '2022-11-13', '2022-11-19', '2022-11-20'])
    covariates_future = df_covariates[df_covariates['date'].isin(future_range)]

    # predict part
    matrix = [[0] * 2+N for _ in range(10)]
    predict_part = pd.DataFrame(matrix, columns=df.columns.values[:2+N])
    predict_part.loc[:, 'date'] = list(future_range)
    predict_part['date'] = pd.to_datetime(predict_part['date'])

    # fusion
    future = pd.merge(predict_part, covariates_future, on='date', how='left')

    result = pd.DataFrame()
    for i in range(0, len(df_for_predict), 10):
        for_predict = df_for_predict.iloc[i:i+10, :]
        id = for_predict.iloc[0, 0]
        id = int(id[7:])
        history = df[df['id'] == id]

        if len(history) <= seq_len:
            print(f'{id}: len = {len(history)}')
            continue
        history = history.iloc[-seq_len:, :]
        history['date'] = pd.to_datetime(history['date'])
        res = pd.concat([history, future])
        res['id'] = id
        result = pd.concat([result, res])

    result = result.reset_index(drop=True)
    result.to_csv(r'D:\Work\datadata\data\阿里天池-金融场景1\for_predict_len10_9_15.csv', index=False)


def for_predict_DLinear(train_path, predict_table_path, enc_in, seq_len):
    df = pd.read_csv(train_path)
    df_for_predict = pd.read_csv(predict_table_path) \
        .rename(columns={'product_pid': 'id', 'transaction_date': 'date'})
    df_for_predict['date'] = pd.to_datetime(df_for_predict['date'], format='%Y%m%d')

    future_range = pd.date_range(start='2022-11-10', end='2022-11-23', freq='1D')
    future_range = future_range.drop(['2022-11-12', '2022-11-13', '2022-11-19', '2022-11-20'])

    # predict part
    matrix = [[0] * (2 + enc_in) for _ in range(10)]
    predict_part = pd.DataFrame(matrix, columns=df.columns.values[:2 + enc_in])
    predict_part.loc[:, 'date'] = list(future_range)
    predict_part['date'] = pd.to_datetime(predict_part['date'])

    result = pd.DataFrame()
    for i in range(0, len(df_for_predict), 10):
        for_predict = df_for_predict.iloc[i:i + 10, :]
        id = for_predict.iloc[0, 0]
        id = int(id[7:])
        history = df[df['id'] == id]

        if len(history) <= seq_len:
            print(f'{id}: len = {len(history)}')
            continue
        history = history.iloc[-seq_len:, :]
        history['date'] = pd.to_datetime(history['date'])
        res = pd.concat([history, predict_part])
        res['id'] = id
        result = pd.concat([result, res])

    result = result.reset_index(drop=True)
    result.to_csv(r'data\for_predict_len10_14.csv', index=False)


def outlier(path, label_list=[]):
    df = pd.read_csv(path)
    k = 1
    for label in label_list:
        plt.subplot(len(label_list), 1, k)
        m = np.mean(df[label])
        std = np.std(df[label])
        plt.plot(df[label])
        for i in range(len(df)):
            if df.loc[i, label] >= m + 3*std or df.loc[i, label] <= m - std*3:
                df.loc[i, label] = None
        df[label] = df[label].interpolate('linear')
        plt.plot(df[label])
        k += 1
    plt.show()
    df.to_csv(r'data/train_and_covariates_linear.csv', index=False)


if __name__ == '__main__':
    df = pd.read_csv(r'data/train_and_covariates_2.csv')
    df['date'] = pd.to_datetime(df['date'])
    date_range_train = pd.date_range(start='2021-01-04', end='2022-07-22', freq='1D')  # train
    df_train = df[df['date'].isin(date_range_train)]
    date_range_vali = pd.date_range(start='2022-07-23', end='2022-09-10', freq='1D')  # vali
    df_vali = df[df['date'].isin(date_range_vali)]
    print(len(df_train) / len(df))
    print(len(df_vali) / len(df))
    print((len(df) - len(df_train) - len(df_vali)) / len(df))