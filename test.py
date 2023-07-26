import joblib
import pandas as pd
import numpy as np
import matplotlib

# read data
df_true = pd.read_csv(r'D:\Work\DLinear_research\data\tianchi\train_and_covariates.csv')
df_pred = pd.read_csv(r'D:\Work\DLinear_research\results\test_Tide_Tide_ftM_sl40_ll20_pl10_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0\real_prediction.csv')
df_true.rename(columns={'apply_amt': 'apply_amt_true', 'redeem_amt': 'redeem_amt_true', 'net_in_amt': 'net_in_amt_true'}, inplace=True)
df_true = df_true.iloc[:, :5]
df_pred = df_pred.iloc[:, :5]
df_pred['date'] = pd.to_datetime(df_pred['date'])
df_true['date'] = pd.to_datetime(df_true['date'])
scalar = joblib.load(r'D:\Work\DLinear_research\scalar\scalar_train_and_covariates')
m = scalar.mean_
std = scalar.scale_
df_pred['apply_amt'] = df_pred['apply_amt'] * std[0] + m[0]
df_pred['redeem_amt'] = df_pred['redeem_amt'] * std[1] + m[1]
df_pred['net_in_amt'] = df_pred['net_in_amt'] * std[2] + m[2]
res = []
for i in range(0, len(df_pred), 10):
    pred_temp = df_pred[i: i+10]
    true_temp = pd.merge(pred_temp, df_true, on=['date', 'id'], how='left')
    wmape = 0
    for j in range(3):
        fenzi = sum(np.abs(list(true_temp.iloc[:, 2+j] - true_temp.iloc[:, 2+j+3])))
        fenmu = sum(np.abs(list(true_temp.iloc[:, 2+j+3])))

        wmape += fenzi / fenmu
    res.append(wmape / 3)
print(np.mean(res))
a = 1