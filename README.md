# HousingPredictions
Final Group Project for CS7643/4803. Using Deep Learning to predict future housing values.</br>
Data files (csv) are in the google drive</br>
Conda environment can be cloned using environment.txt and this link has instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments

Notebooks (read in order):
- API Fred: Messing around with all the api functions that might be needed
- Make State Table: Create the table to store the state id and the state name
- Make County Table: Create the table to store the county name, county id, and corresponding state id. It will be used to pull state-level information into our examples for each county
- Make Series Table: Create the table to store the series id, county id, and series metadata. It will be used to make decisions on which features to use and then to request the corresponding series from the Fred API
- Remove Duplicates: This notebook was for removing duplicates from the county and series tables
- Clean Series Table: This notebook was for cleaning the series table and creating an aggregate dataset to make decisions on which features to use
- Exploring Cleaned Series Results: This notebook was for exploring the cleaned series table and removing counties for which there are no targets
- Make State Level Series Table: Create the table to store the series id, state id, and series metadata. Aggregate all state-level series by feature name to make decisions on which features to use
- Get Fred Data: This notebook can be ignored. It was before I realized there was an API for Fred and I was trying to make a web scraping thing (i didn't want to delete it tho)

Relevant Files (in Google Drive folder) - see notebook notes/contents to understand these files
- state_table.csv: result from Make State Table notebook
- county_table_dedup.csv: result from Remove Duplicates notebook
- series_table_dedup.csv: result from Remove Duplicates notebook
- aggregated_feature_info.csv: result from Clean Series Table notebook
- agg_feat_info_clipped.csv: result from Exploring Cleaned Series Results notebook
- clipped_county_table.csv: result from Exploring Cleaned Series Results notebook
- clipped_series_table.csv: result from Exploring Cleaned Series Results notebook
- state_series_table_all.csv: result from Make State Level Series Table notebook
- agg_state_feat_info.csv: result from Make State Level Series Table notebook

Next Steps
- Get all target series
- Get country level features
- ~~Get state level features~~
- Decide which county-level features to use (meeting on Monday)
