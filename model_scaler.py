import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
file_path = '/Users/macbookpro/Documents/GWP/Datas/scaled_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Split data into features and target
X = data.drop(columns=['actual_productivity'])
y = data['actual_productivity']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler and scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the best model (e.g., Random Forest Regressor)
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Save the model
model_path = '/Users/macbookpro/Documents/GWP/Models/best_model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)

print(f"Best model saved at: {model_path}")

# Save the scaler
scaler_path = '/Users/macbookpro/Documents/GWP/Models/scaler.pkl'
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Scaler saved at: {scaler_path}")
