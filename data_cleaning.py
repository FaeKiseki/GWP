import pandas as pd

# Load the dataset from a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Data/garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Fill missing values in the 'wip' column with the median value
if 'wip' in data.columns:
    median_wip = data['wip'].median()
    data['wip'] = data['wip'].fillna(median_wip)
    print(f"Replaced missing values in 'wip' column with median: {median_wip}")
else:
    print("'wip' column is missing in the dataset.")

# Define a function to handle outliers using the interquartile range (IQR) method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap values to handle outliers
    df[column] = df[column].apply(
        lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
    )
    print(f"Handled outliers in the '{column}' column.")

# Apply the outlier handling function to specific columns
columns_to_check = ['idle_time', 'incentive', 'actual_productivity']
for column in columns_to_check:
    if column in data.columns:
        handle_outliers(data, column)
    else:
        print(f"The column '{column}' is missing in the dataset.")

# Save the cleaned dataset to a new CSV file
output_path = '/Users/macbookpro/Documents/GWP/Data/cleaned_garments_worker_productivity.csv'
data.to_csv(output_path, index=False)
print(f"The cleaned dataset has been saved to {output_path}")
