import pandas as pd

# Loading the dataset as a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Data/garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Displaying the initial rows of the dataset
print(data.head())

# Displaying general information about the dataset
print(data.info())

# Showing statistical summary for numerical columns
print(data.describe())

# Identifying both categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# Checking the dataset for missing values
print(data.isnull().sum())
