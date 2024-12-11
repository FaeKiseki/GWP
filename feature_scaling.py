import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Data/engineered_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Specify the numerical columns to be scaled
numerical_features = [
    'team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive',
    'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers', 'actual_productivity'
]

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])
print(f"Applied scaling to these numerical features: {numerical_features}")

# Save the scaled dataset to a new CSV file
output_path = '/Users/macbookpro/Documents/GWP/Data/scaled_garments_worker_productivity.csv'
data.to_csv(output_path, index=False)
print(f"Saved the scaled dataset to {output_path}")
