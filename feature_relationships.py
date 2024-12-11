import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
file_path = '/Users/macbookpro/Documents/GWP/Datas/scaled_garments_worker_productivity.csv'
data = pd.read_csv(file_path)

# Path to save the visualizations
visual_save_path = '/Users/macbookpro/Documents/GWP/Visuals/'

# Scatter plots to visualize relationships
features_to_plot = ['over_time', 'incentive', 'smv']

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature, y='actual_productivity', alpha=0.6, color='blue')
    plt.title(f'Relationship between Actual Productivity and {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Actual Productivity', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the scatter plot
    save_path = f"{visual_save_path}scatter_{feature}_vs_actual_productivity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved scatter plot: {save_path}")

# Binning continuous variables for better analysis
data['over_time_bin'] = pd.cut(data['over_time'], bins=10, labels=False)
data['incentive_bin'] = pd.cut(data['incentive'], bins=10, labels=False)
data['smv_bin'] = pd.cut(data['smv'], bins=10, labels=False)

# Features to analyze using box plots
features_to_bin_plot = ['over_time_bin', 'incentive_bin', 'smv_bin']

# Generate box plots for binned features
for feature in features_to_bin_plot:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data, x=feature, y='actual_productivity', palette='Blues')

    # Rotate x-axis labels
    plt.xticks(rotation=0, ha='center')  # Labels are short, so 0 degrees is sufficient

    plt.title(f'Distribution of Actual Productivity by {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Actual Productivity', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout to prevent clipping

    # Save the box plot
    save_path = f"{visual_save_path}boxplot_{feature}_vs_actual_productivity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved box plot: {save_path}")
