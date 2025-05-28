import pandas as pd
import numpy as np
from scipy import stats

# Read GP and MLP results from clean files
gp_df = pd.read_csv('gp_clean_results.csv')
mlp_df = pd.read_csv('mlp_clean_results.csv')

# Extract metrics for comparison
metrics = ['accuracy', 'precision', 'recall', 'f1']

# Open file for writing
with open('wilcoxon_results.txt', 'w') as f:
    f.write("Data values:\n")
    for metric in metrics:
        f.write(f"\n{metric.upper()}:\n")
        f.write(f"GP values: {gp_df[metric].values}\n")
        f.write(f"MLP values: {mlp_df[metric].values}\n")

    # Perform Wilcoxon signed-rank test for each metric
    f.write("\nWilcoxon Signed-Rank Test Results\n")
    f.write("=" * 40 + "\n")

    for metric in metrics:
        statistic, p_value = stats.wilcoxon(gp_df[metric].values, mlp_df[metric].values)
        
        f.write(f"\nMetric: {metric.upper()}\n")
        f.write(f"GP mean: {np.mean(gp_df[metric].values):.4f} ± {np.std(gp_df[metric].values):.4f}\n")
        f.write(f"MLP mean: {np.mean(mlp_df[metric].values):.4f} ± {np.std(mlp_df[metric].values):.4f}\n")
        f.write(f"Wilcoxon statistic: {statistic:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Significant difference (a=0.05): {'Yes' if p_value < 0.05 else 'No'}\n")

# Also print to console for immediate feedback
with open('wilcoxon_results.txt', 'r') as f:
    print(f.read()) 