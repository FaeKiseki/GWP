import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Load the dataset
file_path = '/Users/macbookpro/Documents/GWP/Datas/scaled_garments_worker_productivity.csv'
visual_save_path = '/Users/macbookpro/Documents/GWP/Visuals/'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
X = data.drop(columns=['actual_productivity'])
y = data['actual_productivity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
    "Support Vector Regressor": SVR(kernel='rbf')
}

# Initialize results dictionary
results = []

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "MSE": mse, "R2": r2})
    
    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='blue', line_kws={'color': 'red', 'lw': 2})
    plt.title(f"Residual Plot for {name}", fontsize=16)
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{visual_save_path}{name.replace(' ', '_')}_residual_plot.png")
    plt.close()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('/Users/macbookpro/Documents/GWP/Datas/model_evaluation_results.csv', index=False)

# Visualization: Bar Chart Comparing Metrics
metrics = ['MAE', 'MSE', 'R2']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=results_df, palette="viridis")
    plt.title(f"Model Comparison: {metric}", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{visual_save_path}model_comparison_{metric.lower()}.png")
    plt.close()

print("Model evaluation metrics and visualizations saved.")
