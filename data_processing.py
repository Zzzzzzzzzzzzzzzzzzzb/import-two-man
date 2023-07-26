import pandas as pd
import numpy as np


def for_predict(train_path, covariates_path, predict_table_path, N, seq_len):
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


if __name__ == '__main__':
    predict_table_path = r'D:\Work\datadata\data\阿里天池-金融场景1\predict_table.csv'
    train_path = r'D:\Work\datadata\data\阿里天池-金融场景1\train_and_covariates_2.csv'
    covariates_path = r'D:\Work\datadata\data\阿里天池-金融场景1\covariates.csv'
    N = 9
    seq_len = 10
    for_predict(
        train_path=train_path, covariates_path=covariates_path, predict_table_path=predict_table_path, N=N, seq_len=seq_len
        )
