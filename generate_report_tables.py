import pandas as pd
import numpy as np
from scipy import stats

def load_results():
    """Load both MLP and GP results from clean CSV files."""
    mlp_df = pd.read_csv('mlp_clean_results.csv')
    gp_df = pd.read_csv('gp_clean_results.csv')
    return mlp_df, gp_df

def create_seed_comparison_table():
    """Create a table comparing results for each seed."""
    mlp_df, gp_df = load_results()
    
    # Create a DataFrame for the comparison
    rows = []
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    for seed in [1, 314, 999]:
        mlp_row = mlp_df[mlp_df['seed'] == seed].iloc[0]
        gp_row = gp_df[gp_df['seed'] == seed].iloc[0]
        
        # Add GP results
        rows.append({
            'Seed': seed,
            'Model': 'GP',
            'Accuracy': f"{gp_row['accuracy']*100:.2f}%",
            'F1 Score': f"{gp_row['f1']*100:.2f}%",
            'Precision': f"{gp_row['precision']*100:.2f}%",
            'Recall': f"{gp_row['recall']*100:.2f}%"
        })
        
        # Add MLP results
        rows.append({
            'Seed': seed,
            'Model': 'MLP',
            'Accuracy': f"{mlp_row['accuracy']*100:.2f}%",
            'F1 Score': f"{mlp_row['f1']*100:.2f}%",
            'Precision': f"{mlp_row['precision']*100:.2f}%",
            'Recall': f"{mlp_row['recall']*100:.2f}%"
        })
    
    comparison_df = pd.DataFrame(rows)
    return comparison_df

def create_summary_table():
    """Create a summary table with mean values and statistical significance."""
    mlp_df, gp_df = load_results()
    
    rows = []
    metrics = {
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    for metric, metric_name in metrics.items():
        # Calculate means and standard deviations
        gp_mean = np.mean(gp_df[metric]) * 100
        gp_std = np.std(gp_df[metric]) * 100
        mlp_mean = np.mean(mlp_df[metric]) * 100
        mlp_std = np.std(mlp_df[metric]) * 100
        diff = mlp_mean - gp_mean
        
        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(gp_df[metric].values, mlp_df[metric].values)
        
        # Create two versions of the mean strings - one for text and one for LaTeX
        text_gp_mean = f"{gp_mean:.2f}% +/- {gp_std:.2f}%"
        text_mlp_mean = f"{mlp_mean:.2f}% +/- {mlp_std:.2f}%"
        latex_gp_mean = f"{gp_mean:.2f}\\% ± {gp_std:.2f}\\%"
        latex_mlp_mean = f"{mlp_mean:.2f}\\% ± {mlp_std:.2f}\\%"
        
        rows.append({
            'Metric': metric_name,
            'GP Mean': text_gp_mean,
            'MLP Mean': text_mlp_mean,
            'GP Mean LaTeX': latex_gp_mean,
            'MLP Mean LaTeX': latex_mlp_mean,
            'Difference': f"{'+' if diff > 0 else ''}{diff:.2f}%",
            'Difference LaTeX': f"{'+' if diff > 0 else ''}{diff:.2f}\\%",
            'p-value': f"{p_value:.4f}",
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    summary_df = pd.DataFrame(rows)
    return summary_df

def save_tables():
    """Save tables in both text and LaTeX formats."""
    comparison_df = create_seed_comparison_table()
    summary_df = create_summary_table()
    
    # Save text format
    with open('report_tables.txt', 'w') as f:
        f.write("Overall, the results suggest that:\n")
        f.write("MLP shows more consistent performance across different seeds\n")
        f.write("GP can achieve excellent results (as seen in seed 1) but has higher variance\n")
        f.write("The differences between the models are not statistically significant at a=0.05\n")
        f.write("\n\n")
        
        f.write("Table 1: Comparison of GP and MLP Results by Seed\n")
        f.write("=" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Table 2: Summary Statistics and Statistical Significance\n")
        f.write("=" * 80 + "\n")
        # Create a copy of summary_df with only the text columns for display
        display_df = summary_df.copy()
        display_df = display_df[['Metric', 'GP Mean', 'MLP Mean', 'Difference', 'p-value', 'Significant']]
        f.write(display_df.to_string(index=False))
        
        # Add key findings
        f.write("\n\nKey Findings:\n")
        f.write("Accuracy:\n")
        f.write("- GP performed better for seed 1 (99.62% vs 88.59%)\n")
        f.write("- MLP performed better for seeds 314 and 999 (96.58% vs 83.65% and 65.40%)\n")
        f.write("- The difference is not statistically significant (p-value = 0.5000)\n\n")
        
        f.write("F1 Score:\n")
        f.write("- GP performed better for seed 1 (99.63% vs 88.46%)\n")
        f.write("- MLP performed better for seeds 314 and 999 (96.57% vs 80.72% and 49.72%)\n\n")
        
        f.write("Precision:\n")
        f.write("- GP achieved high precision across all seeds (99.25%, 100%, 93.75%)\n")
        f.write("- MLP had more consistent precision (90.73%, 96.79%, 96.79%)\n\n")
        
        f.write("Recall:\n")
        f.write("- GP showed high variance in recall (100%, 67.67%, 33.83%)\n")
        f.write("- MLP was more consistent (88.59%, 96.58%, 96.58%)\n")
    
    # Save LaTeX format
    with open('report_tables.tex', 'w') as f:
        # Document preamble
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\usepackage{caption}\n")
        f.write("\\usepackage[table]{xcolor}\n")
        f.write("\n")
        f.write("\\title{GP vs MLP Performance Comparison}\n")
        f.write("\\author{Statistical Analysis Results}\n")
        f.write("\n")
        f.write("\\begin{document}\n")
        f.write("\\maketitle\n\n")
        
        # Add description of statistical testing
        f.write("\\section*{Statistical Analysis Details}\n")
        f.write("The comparison between GP and MLP models was conducted using the Wilcoxon signed-rank test ")
        f.write("with significance level $a = 0.05$. This non-parametric test was chosen due to the small ")
        f.write("sample size (3 seeds) and no assumption of normal distribution. A p-value $< 0.05$ would indicate ")
        f.write("a statistically significant difference between the models.\n\n")
        
        # Table 1
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of GP and MLP Results by Seed}\n")
        f.write("\\begin{tabular}{cccccc}\n")
        f.write("\\toprule\n")
        f.write("Seed & Model & Accuracy & F1 Score & Precision & Recall \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"{int(row['Seed'])} & {row['Model']} & {row['Accuracy']} & {row['F1 Score']} & {row['Precision']} & {row['Recall']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Table 2
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics and Statistical Significance}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Metric & GP Mean & MLP Mean & Difference & p-value & Significant \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['Metric']} & {row['GP Mean LaTeX']} & {row['MLP Mean LaTeX']} & {row['Difference LaTeX']} & {row['p-value']} & {row['Significant']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        f.write("\\end{document}\n")

def main():
    save_tables()
    print("Tables have been updated with actual Wilcoxon test results and saved to report_tables.txt and report_tables.tex")

if __name__ == "__main__":
    main() 