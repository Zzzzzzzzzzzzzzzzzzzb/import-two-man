import pandas as pd
import numpy as np
import joblib

df = pd.read_csv(r'D:\Work\DLinear_research\results\predict.csv')
predict_table = pd.read_csv(r'D:\Work\DLinear_research\data\tianchi\predict_table.csv')
scalar = joblib.load(r'D:\Work\DLinear_research\scalar\scalar_train_and_covariates')
m = scalar.mean_
std = scalar.scale_
df['apply_amt'] = df['apply_amt'] * std[0] + m[0]
df['redeem_amt'] = df['redeem_amt'] * std[1] + m[1]
df['net_in_amt'] = df['net_in_amt'] * std[2] + m[2]
res = pd.concat([predict_table.iloc[:, :2], df.iloc[:, 2: 5]], axis=1)
res = res.rename(columns={'apply_amt': 'apply_amt_pred', 'redeem_amt': 'redeem_amt_pred', 'net_in_amt': 'net_in_amt_pred'})
res.to_csv(r'D:\Work\DLinear_research\data\tianchi\result.csv', index=False, sep=',')
a = 1