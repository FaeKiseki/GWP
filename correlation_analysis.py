import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Datas/scaled_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Compute the correlation matrix
correlation_matrix = data.corr()

# Set the path to save the heatmap
save_path = '/Users/macbookpro/Documents/GWP/Visuals/correlation_heatmap.png'

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)

# Adjust the x-axis label rotation and alignment
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()

# Save the heatmap
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Correlation heatmap saved to {save_path}")
