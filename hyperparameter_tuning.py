import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the dataset
file_path = '/Users/macbookpro/Documents/GWP/Datas/scaled_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
X = data.drop(columns=['actual_productivity'])
y = data['actual_productivity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grids for non-XGB models
param_grids = {
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Random Forest Regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Gradient Boosting Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10]
    }
}

# Define and tune non-XGB models
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

# Store results
tuned_results = []

for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    if model_name in ["Ridge", "Lasso"]:
        # GridSearchCV for Ridge and Lasso
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='r2', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        # RandomizedSearchCV for tree-based models
        from sklearn.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_name], n_iter=20, cv=3, scoring='r2', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
    
    y_pred = best_model.predict(X_test)
    tuned_results.append({
        "Model": model_name,
        "Best Parameters": best_params,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    })

# Custom tuning for XGBoost
print("Tuning XGBoost Regressor...")
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_score = -float('inf')
best_params = {}

for n_estimators in xgb_param_grid["n_estimators"]:
    for learning_rate in xgb_param_grid["learning_rate"]:
        for max_depth in xgb_param_grid["max_depth"]:
            params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth
            }
            scores = []
            for train_idx, val_idx in kf.split(X_train):
                x_train, x_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                xgb_model.set_params(**params)
                xgb_model.fit(x_train, y_train_fold)
                y_val_pred = xgb_model.predict(x_val)
                scores.append(r2_score(y_val, y_val_pred))
            mean_score = sum(scores) / len(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

xgb_model.set_params(**best_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

tuned_results.append({
    "Model": "XGBoost Regressor",
    "Best Parameters": best_params,
    "MAE": mean_absolute_error(y_test, y_pred_xgb),
    "MSE": mean_squared_error(y_test, y_pred_xgb),
    "R2": r2_score(y_test, y_pred_xgb)
})

# Save results
tuned_results_df = pd.DataFrame(tuned_results)
tuned_results_df.to_csv('/Users/macbookpro/Documents/GWP/Datas/tuned_model_results.csv', index=False)
print("Hyperparameter tuning results saved to CSV.")
