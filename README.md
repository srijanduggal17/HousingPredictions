# HousingPredictions
Using Deep Learning to predict future housing values.</br>
Final Group Project for CS 7643/4803. >/br>
Group members were Krishan Bhakta, Srijan Duggal, Chris Fleisher, Pratyusha Karnati, and Anshul Tusnial

Data files (csv) are in the google drive</br>
Conda environment can be cloned using environment.txt and this link has instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments

Notebooks/Python files (read in order):
Unless indicated in brackets, all files are in master branch

For dataset research: inside master branch under folder "Choosing our Features"
- API Fred: Messing around with all the api functions that might be needed
- Make State Table: Create the table to store the state id and the state name
- Make County Table: Create the table to store the county name, county id, and corresponding state id. It will be used to pull state-level information into our examples for each county
- Make Series Table: Create the table to store the series id, county id, and series metadata. It will be used to make decisions on which features to use and then to request the corresponding series from the Fred API
- Remove Duplicates: This notebook was for removing duplicates from the county and series tables
- Clean Series Table: This notebook was for cleaning the series table and creating an aggregate dataset to make decisions on which features to use
- Exploring Cleaned Series Results: This notebook was for exploring the cleaned series table and removing counties for which there are no targets
- Make State Level Series Table: Create the table to store the series id, state id, and series metadata. Aggregate all state-level series by feature name to make decisions on which features to use
- Get Country Level Features: Create the table to store the series id and series metadata for country level features. Get the full series data for each feature.
- Get Timeline for Chosen Features: Take all our country level features and trim them to a set of features and counties that share a timeline of data
- Trim State Features to Timeline: Take all the state level features and trim them to the features and states that share the timeline dictated by the county-level features
- Get Fred Data: This notebook can be ignored. It was before I realized there was an API for Fred and I was trying to make a web scraping thing (i didn't want to delete it tho)

For dataset creation:
- [county_state_data]Get County Features: Get the actual observations for each county-level feature
- [county_state_data]Get Country Level Features: Get the actual observations for each country-level feature
- [county_state_data]Get-State-Features: Get the actual observations for each state-level feature 
- [data_agg]utils/data/CountyDataset.py: Combining the data into a Pytorch dataset format that can be used with a DataLoader
- [data_agg]dataset_example.py: Example of how to use the CountyDataset
- CountyDataset_KB.py: Combining the data into format that can be used with a Random Forest model
- dataset_example_KB.py: Example of how to use the CountyDataset for a Random Forest model

For model creation/evaluation:
- [data_agg]models/RNN.py: The RNN architecture we built using Pytorch
- [data_agg]models/LSTM.py: The LSTM architecture we built using Pytorch
- [data_agg]model_training_stuff.ipynb: Training, Tuning, and Evaluating the baseline linear regression as well as the RNN and LSTM models
- feature_selection.py: Using sequential forward feature selection and exhaustive sequential search on a random forest model
- random_forest.py: The finalized random forest model
- [results_plotting]Hyperparameter Plots: Plotting of the hyperparameter heatmaps for each dataset model combination

Link to view list of features and a brief description: https://docs.google.com/spreadsheets/d/1yX5qohttdohDgZSLiX7vnL44KqtLH_4VUgqU1Qsvz78/edit?usp=sharing<br>
Link to view model results: https://docs.google.com/spreadsheets/d/1tXWqvpAQH8pWk_dcBkn7xfK9TyiygSPt4dgF_23QuHE/edit?usp=sharing<br>
Link to view learning curves: https://drive.google.com/drive/folders/1QF-U58S6pPQxYaY1iS3O9aoZi610mlmX?usp=sharing<br>
Link to view hyperparameter results: https://drive.google.com/drive/folders/1or1BbhLpo476xmVt6yRT_FfGZvVigRhq?usp=sharing


Relevant Files (in Google Drive folder - this note is for team members) - see notebook notes/contents to understand these files
- state_table.csv: result from Make State Table notebook
- county_table_dedup.csv: result from Remove Duplicates notebook
- series_table_dedup.csv: result from Remove Duplicates notebook
- aggregated_feature_info.csv: result from Clean Series Table notebook
- agg_feat_info_clipped.csv: result from Exploring Cleaned Series Results notebook
- clipped_county_table.csv: result from Exploring Cleaned Series Results notebook
- clipped_series_table.csv: result from Exploring Cleaned Series Results notebook
- state_series_table_all.csv: result from Make State Level Series Table notebook
- agg_state_feat_info.csv: result from Make State Level Series Table notebook
- country_series_table.csv: result from Get Country Level Features notebook
- country_features.csv: result from Get Country Level Features notebook
- county_features_trimmed.csv: result from Get Timeline for Chosen Features notebook
- county_features_final.csv: the list of county-level features we will use and the corresponding FRED series that need to be retrieved from their API. Result from Trim State Features to Timeline notebook
- state_features_final.csv: the list of state-level features we will use and the corresponding FRED series that need to be retrieved from their API. Result from Trim State Features to Timeline notebook

This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
