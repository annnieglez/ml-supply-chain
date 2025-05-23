# Data Folder

This folder contains various CSV files used throughout the project, including raw data, preprocessed datasets for database construction, exploratory data analysis (EDA), and machine learning (ML) models.

## Files and Descriptions

### 1. `DataCoSupplyChainDataSet.csv`

- **Source**: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS from Kaggle.
- **Download Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/data)
- **Description**: The original dataset, containing supply chain data used for analysis and modeling.

### 2. `DescriptionDataCoSupplyChain.csv`

- **Description**: A file containing explanations for each column in the dataset.
- **Usage**: Used as a reference for understanding the dataset's variables and their meanings.

### 3. `orders_cleaned_db.csv`

- **Description**: A preprocessed dataset specifically formatted for the database construction.
- **Usage**: Used as based datafile for the data cleaning.

### 4. `numerical.csv`

- **Description**: A subset of the dataset containing only numerical columns.
- **Usage**: Used for exploratory data analysis (EDA) focusing on numerical data.

### 5. `numerical_with_id.csv`

- **Description**: Similar to `numerical.csv` but includes ID information.
- **Usage**: Used for customer clustering analysis in EDA.

### 6. `categorical.csv`

- **Description**: A dataset containing only categorical variables.
- **Usage**: Used for categorical data analysis in EDA.

### 7. `df_ml.csv`

- **Description**: The final cleaned and processed dataset used for training machine learning models.
- **Usage**: Input dataset for various ML classification models.

## Notes

- `orders_cleaned_db` is generated during the database processing files in `../notebooks/data_cleaning_for_database.ipynb`
- `numerical.csv`, `numerical_with_id.csv`, `categorical.csv` and `df_ml.csv` are generated during the data cleaning at `../notebooks/data_cleaning.ipynb`
