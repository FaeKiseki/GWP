import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Datas/scaled_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
X = data.drop(columns=['actual_productivity'])  # All features except the target variable
y = data['actual_productivity']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print(f"Training set (X_train): {X_train.shape}")
print(f"Training labels (y_train): {y_train.shape}")
print(f"Test set (X_test): {X_test.shape}")
print(f"Test labels (y_test): {y_test.shape}")

# Save the splits as CSV files
X_train.to_csv('/Users/macbookpro/Documents/GWP/Datas/X_train.csv', index=False)
X_test.to_csv('/Users/macbookpro/Documents/GWP/Datas/X_test.csv', index=False)
y_train.to_csv('/Users/macbookpro/Documents/GWP/Datas/y_train.csv', index=False)
y_test.to_csv('/Users/macbookpro/Documents/GWP/Datas/y_test.csv', index=False)

print("Data splits saved to CSV files.")
