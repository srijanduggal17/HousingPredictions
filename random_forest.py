import pandas as pd
import numpy as np
from dataset_example import get_dataset, get_numpy, check_dfs
import joblib
import os
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
import xgboost as xgb
from matplotlib import pyplot as plt

master_path = '/home/kbhakta/Dropbox (GaTech)/Georgia Tech Classes/Spring 2020/Deep Learning/Final_Project/data/master_county_state_reference.csv'
save_path = os.getcwd()
file_name = '/rf_data'

# ## RUN ONCE TO GET DATA AND SAVE AS LOADABLE FILE
# #_______________________________________________
# dataset = get_dataset(master_path)
# data, dataset.xcols, dataset.target_col, dataset.date_index = get_numpy(dataset) # Refer to function in dataset_example for this function
# print(save_path + file_name)
# joblib.dump([data, dataset], save_path + file_name)
# #_______________________________________________

[data, dataset] = joblib.load(save_path + file_name)

print("# of Counties: ", len(data))
print('-'*100)
print(len(dataset.xcols))
print('-'*100)
print("Target Column: ", dataset.target_col)
print('-'*100)

# # Make a list of all the dataframes
# dfs = []
# for i in range(len(data)):
# 	dfs.append(dataset._get_df(i))
# joblib.dump(dfs, save_path + '/all_data')

dfs = joblib.load(save_path + '/all_data')
merged_df = pd.concat(dfs)
# print(merged_df)

# x is training data, y is testing data
x = merged_df.drop(pd.DatetimeIndex(['2010-12-01', '2011-12-01', '2012-12-01', '2013-12-01','2014-12-01', '2015-12-01', '2016-12-01', '2017-12-01'], dtype='datetime64[ns]'))
y = merged_df.loc[pd.DatetimeIndex(['2010-12-01', '2011-12-01', '2012-12-01', '2013-12-01','2014-12-01', '2015-12-01', '2016-12-01', '2017-12-01'], dtype='datetime64[ns]')]

print(x.shape)
print(y.shape)
print('-'*100)

x_features = x.drop(['Zillow Price Index'], axis = 1)
x_target  = x['Zillow Price Index']
y_features = y.drop(['Zillow Price Index'], axis = 1)
y_target  = y['Zillow Price Index']

print(x_features.shape)
print(x_target.shape)
print(y_features.shape)
print(y_target.shape)
print('-'*100)

print("Most Important Features in Order: ", x_features.columns[[17, 37, 36,155]])

model = RFR(n_jobs = -1)
# model = xgb.XGBRegressor()

model.fit(x_features,x_target)
y_pred = model.predict(y_features)
mse = mean_squared_error(y_target, y_pred)
print("MSE: ", mse)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)
print('-'*100)

print(model.feature_importances_)
num_objects = np.arange(len(model.feature_importances_))
plt.bar(num_objects, model.feature_importances_)

z = model.feature_importances_
# plt.show()

# print(x_features.columns[[17, 37, 36,155]])
