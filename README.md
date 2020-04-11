# HousingPredictions
Final Group Project for CS7643/4803. Using Deep Learning to predict future housing values.</br>
Data files (csv) are in the google drive</br>
Conda environment can be cloned using environment.txt and this link has instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments

Notebooks:
- API Fred: Messing around with all the api functions that might be needed
- Make State Table: Create the table to store the state id and the state name
- Make County Table: Create the table to store the county name, county id, and corresponding state id. It will be used to pull state-level information into our examples for each county
- Make Series Table: Create the table to store the series id, county id, and series metadata. It will be used to make decisions on which features to use and then to request the corresponding series from the Fred API
- Remove Duplicates: This notebook is for removing duplicates from the county and series tables
- Clean Series Table: This notebook was for cleaning the series table and creating an aggregate dataset to make decisions on which features to use
- Get Fred Data: This notebook can be ignored. It was before I realized there was an API for Fred and I was trying to make a web scraping thing (i didn't want to delete it tho)
