import pandas as pd
import os

path = r'D:\Work\DLinear_research\data\3jiaqu'

for file in os.listdir(path):
    df = pd.read_csv(os.path.join(path, file))
    df['date'] = pd.to_datetime(df['date'])
    data_range = pd.date_range(start=df.iloc[0, 0], end=df.iloc[-1, 0], freq='30min')
    df = df[df['date'].isin(data_range)]
    save_path = os.path.join('D:\Work\DLinear_research\data', file)[:-8] + '30min.csv'
    df.to_csv(save_path, index=False)