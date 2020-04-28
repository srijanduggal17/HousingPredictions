import pandas as pd
import numpy as np
from dataset_example import get_dataset, get_numpy, check_dfs
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt

master_path = '/home/kbhakta/Dropbox (GaTech)/Georgia Tech Classes/Spring 2020/Deep Learning/Final_Project/data/master_county_state_reference.csv'
save_path = os.getcwd()
file_name = '/saved_data'

## RUN ONCE TO GET DATA AND SAVE AS LOADABLE FILE
_______________________________________________#
dataset = get_dataset(master_path)
data, dataset.xcols, dataset.target_col, dataset.date_index = get_numpy(dataset) # Refer to function in dataset_example for this function
dfs_list = check_dfs(master_path)
print(save_path + file_name)
joblib.dump([data, dataset, dfs_list], save_path + file_name)
_______________________________________________#

[data, dataset, dfs_list] = joblib.load(save_path + file_name)

print("# of Dataframes: ", len(dfs_list))
print('-'*100)
# print(dfs_list[0])
# print('-'*100)
# print(dfs_list[1])
# print(len(data[0][0]))
# print('-'*100)
# print(dataset.xcols[:10])
# print('-'*100)
print("Target Column: ", dataset.target_col)
print('-'*100)
# print(dataset.date_index)
# print('-'*100)

## TODO: Make loop for feature selection
idx_list = []
for i in range(len(dfs_list)):
	print('-'*100)
	print(i)
	print('-'*100)
	x = dfs_list[i].iloc[:,:-1]
	y = dfs_list[i].iloc[:,-1]

	x = dfs_list[i].iloc[0:84,:-1].sort_index()
	y = dfs_list[i].iloc[0:84,-1].sort_index()
	x_test = dfs_list[i].iloc[84:,:-1].sort_index()
	y_test = dfs_list[i].iloc[84:,-1].sort_index()

	# model = xgb.XGBRegressor(objective='reg:squarederror')
	model = MLPRegressor(
        hidden_layer_sizes=(20,), activation='relu', solver='adam', batch_size='auto', 
        learning_rate='constant', learning_rate_init=0.01, max_iter=1000, shuffle = True
        )

	# xgb_model.fit(x,y)
	# y_pred = xgb_model.predict(x_test)
	# mse = mean_squared_error(y_test, y_pred)
	# print("MSE: ", mse)
	# rmse = np.sqrt(mse)
	# print("RMSE: ", rmse)
	# print('-'*100)

	### Mlxtend Implementation of Sequential Feature Selection
	sfs = SFS(model, 
		k_features=20,#x.shape[1], 
		forward=True, 
		floating=False,
		scoring = 'neg_mean_squared_error', 
		cv=0)

	sfs = sfs.fit(x, y)

	# scores = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
	# fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
	# plt.grid()
	# plt.show()

	idx = sfs.k_feature_idx_
	print(type(idx))
	print('Selected features indexes:', sfs.k_feature_idx_)
	idx_list.append(idx)
	# fNames = sfs.k_feature_names_
	# print('Selected features:', sfs.k_feature_names_)
	
	# if (i == 1):
	# 	print(idx_list[1][0])
	# 	fds

joblib.dump(idx_list, save_path + '/idx_features_xgb')

# idx_list = joblib.load(save_path + '/idx_features')

# print(idx_list[1][0])
#########################################################################################

# ## Load in some sample data
# df = pd.read_csv('sample_table.csv')
# df = df.iloc[:,1:]
# # print(df)
# # print(df.shape)
# x_total = df.iloc[:,:-1]
# y_total = df.iloc[:,-1]
# x = df.iloc[0:80,:-1].sort_index()
# y = df.iloc[0:80,-1].sort_index()
# x_test = df.iloc[80:-1,:-1].sort_index()
# y_test = df.iloc[80:-1,-1].sort_index()
# # print(x.shape)
# # print(y.shape)
# # print(x_test.values)
# # print(y_test.values)
# print('-'*100)

# ############# Preliminary XGBoost model
# # xgb_model = xgb.XGBRegressor()
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
# xgb_model.fit(x,y)
# y_pred = xgb_model.predict(x_test)

# # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # print("RMSE: ", rmse)
# mse = mean_squared_error(y_test, y_pred)
# print("MSE: ", mse)
# print('-'*100)
# ###################

# # ### Plotting Feature Importance
# # print("Feature Importance: ", xgb_model.feature_importances_)
# # xgb.plot_importance(xgb_model)
# # plt.show()

# # ### Plot prediction versus groundtruth
# # plt.plot(y_test.values, label = 'GroundTruth')
# # plt.plot(y_pred, label = 'Prediction')
# # plt.legend()
# # plt.show()

# ### Scikit learn implementation
# # Fit model using each importance as a threshold
# # thresholds = np.sort(xgb_model.feature_importances_)
# # for thresh in thresholds:
# # 	# select features using threshold
# # 	selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)

# # 	xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
# # 	xgb_model.fit(x,y)
# # 	y_pred = xgb_model.predict(x_test)

# # 	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # 	# print("RMSE: ", rmse)
# # 	print("Thresh=%.3f, RMSE: %.3f%%" % (thresh, rmse))

# # SHAP Explanation

# # explainer = shap.TreeExplainer(model)
# # shap_values = explainer.shap_values(X)

# # # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# # shap.force_plot(explainer.expected_value, shap_values[0,:], x)

# ### Mlxtend Implementation of Sequential Feature Selection
# # sfs = SFS(xgb_model, 
# # 	k_features=10,#x.shape[1], 
# # 	forward=True, 
# # 	floating=False,
# # 	scoring = 'neg_mean_squared_error', 
# # 	cv=0)

# # sfs = sfs.fit(x, y)

# # scores = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# # fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
# # plt.grid()
# # plt.show()
# # idx = sfs.k_feature_idx_
# # print('Selected features indexes:', sfs.k_feature_idx_)
# # fNames = sfs.k_feature_names_
# # print('Selected features:', sfs.k_feature_names_)

# ### Mlxtend Implementation of Exhaustive Feature Selection -- if we need to try all combinations (takes very long time to run)
# # efs = EFS(xgb_model, 
# # 	min_features=10,
# #     max_features=12,
# # 	print_progress = True,
# # 	scoring = 'neg_mean_squared_error', 
# # 	cv=0)

# # efs = efs.fit(x, y)

# # print('Best MSE score: %.2f' % efs.best_score_ * (-1))
# # print('Best subset:', efs.best_idx_)
