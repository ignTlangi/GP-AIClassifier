import pandas as pd
import json

def extract_mlp_results():
    """Extract MLP results from JSON file for seeds 1, 314, and 999."""
    with open('experiment_results.json', 'r') as f:
        content = f.read().strip()
        if not content.startswith('{'):
            content = '{' + content
        if not content.endswith('}'):
            content = content + '}'
        data = json.loads(content)
    
    target_seeds = ['1', '314', '999']
    results = []
    
    for seed in target_seeds:
        if seed in data and 'metrics' in data[seed] and 'test' in data[seed]['metrics']:
            metrics = data[seed]['metrics']['test']
            results.append({
                'seed': int(seed),
                'accuracy': metrics['accuracy'] / 100,  # Convert to decimal for consistency
                'f1': metrics['f1'],  # Already in decimal
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })
    
    df = pd.DataFrame(results)
    df.to_csv('mlp_clean_results.csv', index=False)
    print("MLP results extracted to mlp_clean_results.csv")

def extract_gp_results():
    """Extract GP results from CSV file for seeds 1, 314, and 999."""
    # Read the raw GP results
    gp_df = pd.read_csv('gp_results.csv')
    
    # Extract test metrics for each seed
    results = []
    for _, row in gp_df.iterrows():
        results.append({
            'seed': int(row['Seed']),
            'accuracy': row['Test_Accuracy'],
            'precision': row['Test_Precision'],
            'recall': row['Test_Recall'],
            'f1': row['Test_F1']
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('gp_clean_results.csv', index=False)
    print("GP results extracted to gp_clean_results.csv")

def main():
    print("Extracting results...")
    extract_mlp_results()
    extract_gp_results()
    print("Done! Both results are now in clean CSV format.")

if __name__ == "__main__":
    main() 