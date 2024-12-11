import pandas as pd

# Load the dataset from a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Data/cleaned_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Apply one-hot encoding to categorical features
categorical_features = ['quarter', 'department', 'day']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
print("Performed one-hot encoding on these categorical features:", categorical_features)

# Convert the 'date' column to datetime and extract additional time-based features
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Parse the 'date' column
    data['month'] = data['date'].dt.month  # Add a 'month' feature
    data['day_of_week'] = data['date'].dt.dayofweek  # Add a 'day_of_week' feature (0=Monday to 6=Sunday)
    print("Extracted date-related features: month, day_of_week")
else:
    print("'date' column is missing from the dataset.")

# Remove the 'date' column after extracting features
data.drop(columns=['date'], inplace=True)
print("Dropped the 'date' column after feature extraction.")

# Save the feature-engineered dataset to a new CSV file
output_path = '/Users/macbookpro/Documents/GWP/Data/engineered_garments_worker_productivity.csv'
data.to_csv(output_path, index=False)
print(f"Saved the feature-engineered dataset to {output_path}")
