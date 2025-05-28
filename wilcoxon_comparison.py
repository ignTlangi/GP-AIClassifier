import pandas as pd
import numpy as np
from scipy import stats

def load_results():
    """Load both MLP and GP results from clean CSV files."""
    mlp_df = pd.read_csv('mlp_clean_results.csv')
    gp_df = pd.read_csv('gp_clean_results.csv')
    
    # Sort both DataFrames by seed to ensure alignment
    mlp_df = mlp_df.sort_values('seed').reset_index(drop=True)
    gp_df = gp_df.sort_values('seed').reset_index(drop=True)
    
    return mlp_df, gp_df

def create_comparison_table(f, mlp_df, gp_df):
    """Create a formatted comparison table for the report."""
    f.write("\nComparison Table for GP vs MLP:\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Seed':<6} | {'Metric':<10} | {'GP Value':>10} | {'MLP Value':>10} | {'Difference':>10}\n")
    f.write("-" * 80 + "\n")
    
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    for i in range(len(mlp_df)):
        seed = mlp_df.iloc[i]['seed']
        mlp_row = mlp_df.iloc[i]
        gp_row = gp_df.iloc[i]
        
        for metric in metrics:
            gp_val = gp_row[metric] * 100  # Convert to percentage
            mlp_val = mlp_row[metric] * 100  # Convert to percentage
            diff = mlp_val - gp_val
            f.write(f"{seed:<6} | {metric:<10} | {gp_val:>10.4f} | {mlp_val:>10.4f} | {diff:>+10.4f}\n")
        f.write("-" * 80 + "\n")

def perform_wilcoxon_test(f, mlp_df, gp_df):
    """Perform Wilcoxon signed-rank test between GP and MLP results."""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    f.write("\nWilcoxon Signed-Rank Test Results:\n")
    f.write("=" * 80 + "\n")
    
    for metric in metrics:
        mlp_values = mlp_df[metric].values * 100  # Convert to percentage
        gp_values = gp_df[metric].values * 100  # Convert to percentage
        
        try:
            statistic, p_value = stats.wilcoxon(mlp_values, gp_values)
            f.write(f"\nMetric: {metric}\n")
            f.write(f"Statistic: {statistic:.4f}\n")
            f.write(f"p-value: {p_value:.4f}\n")
            f.write(f"Significant difference (alpha=0.05): {'Yes' if p_value < 0.05 else 'No'}\n")
            f.write(f"Mean GP: {np.mean(gp_values):.4f}%\n")
            f.write(f"Mean MLP: {np.mean(mlp_values):.4f}%\n")
            f.write(f"Mean Difference: {np.mean(mlp_values - gp_values):+.4f}%\n")
        except Exception as e:
            f.write(f"\nMetric: {metric}\n")
            f.write(f"Could not perform Wilcoxon test: {str(e)}\n")
        f.write("-" * 80 + "\n")

def main():
    # Load results from clean CSV files
    mlp_results, gp_results = load_results()
    
    # Write results to file
    with open('comparison_results.txt', 'w', encoding='utf-8') as f:
        # Create comparison table
        create_comparison_table(f, mlp_results, gp_results)
        
        # Perform Wilcoxon test
        perform_wilcoxon_test(f, mlp_results, gp_results)
    
    # Print the results file
    with open('comparison_results.txt', 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == "__main__":
    main() 