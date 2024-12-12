import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the tuned results and initial results
tuned_results_path = '/Users/macbookpro/Documents/GWP/Datas/tuned_model_results.csv'
initial_results_path = '/Users/macbookpro/Documents/GWP/Datas/model_evaluation_results.csv'

tuned_results_df = pd.read_csv(tuned_results_path)
initial_results_df = pd.read_csv(initial_results_path)

# Ensure the models match in both files
tuned_results_df = tuned_results_df.set_index("Model")
initial_results_df = initial_results_df.set_index("Model")

# Combine results for comparison
comparison_df = initial_results_df[["MAE", "MSE", "R2"]].join(
    tuned_results_df[["MAE", "MSE", "R2"]], lsuffix="_Before", rsuffix="_After"
)
comparison_df.reset_index(inplace=True)

# Plot performance before and after tuning
metrics = ["MAE", "MSE", "R2"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=comparison_df,
        x="Model",
        y=f"{metric}_Before",
        label=f"{metric} Before Tuning",
        marker="o"
    )
    sns.lineplot(
        data=comparison_df,
        x="Model",
        y=f"{metric}_After",
        label=f"{metric} After Tuning",
        marker="o"
    )
    plt.title(f"{metric} Before and After Tuning", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"/Users/macbookpro/Documents/GWP/Visuals/{metric}_Comparison.png")
    plt.show()

# Identify the best-performing model after tuning
best_model_row = tuned_results_df.loc[tuned_results_df["R2"].idxmax()]
best_model_name = best_model_row.name

print(f"Best Model After Tuning: {best_model_name}")
print(f"Best Parameters: {best_model_row['Best Parameters']}")
print(f"R2: {best_model_row['R2']:.4f}, MSE: {best_model_row['MSE']:.4f}, MAE: {best_model_row['MAE']:.4f}")

# Load predictions from the best model
# Assuming predictions for the best model are saved (you can integrate this step earlier)
# Generate final residual plot
residuals = pd.read_csv('/Users/macbookpro/Documents/GWP/Datas/best_model_residuals.csv')

plt.figure(figsize=(10, 6))
sns.residplot(x=residuals["Predicted"], y=residuals["Residuals"], lowess=True, color="blue", line_kws={"color": "red", "lw": 2})
plt.title(f"Residual Plot for {best_model_name}", fontsize=16)
plt.xlabel("Predicted Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"/Users/macbookpro/Documents/GWP/Visuals/{best_model_name}_Residual_Plot.png")
plt.show()
