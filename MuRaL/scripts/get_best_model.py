import os
import pandas as pd
from pathlib import Path

def get_best_model(experiment_path):
    """
    Find the model with minimum validation loss from Ray Tune experiment directories.
    
    Args:
        experiment_path (str): Path containing multiple Train_* subdirectories with progress.csv files
        
    Returns:
        pd.DataFrame: Sorted DataFrame containing:
            - checkpoint_path: Path to model checkpoint
            - validation_loss: Corresponding validation loss
            Sorted by validation_loss (ascending)
    """
    # Collect all progress.csv files from Train_* subdirectories
    csv_files = []
    for item in os.listdir(experiment_path):
        if item.startswith('Train_'):
            csv_path = os.path.join(experiment_path, item, 'progress.csv')
            if os.path.exists(csv_path):
                csv_files.append(csv_path)
            else:
                print(f"Warning: Missing progress.csv in {item}")

    if not csv_files:
        raise FileNotFoundError(f"No valid progress.csv files found in {experiment_path}")

    results = []
    
    for csv_file in csv_files:
        try:
            # Read CSV with tab separator and standardized column names
            df = pd.read_csv(csv_file, sep='\t')
            df.columns = ['validation_loss', 'validation_loss_after_calibra']
            
            # Find epoch with minimum validation loss
            min_loss_idx = df['validation_loss'].idxmin()
            min_loss_row = df.loc[min_loss_idx]
            
            # Construct checkpoint path
            checkpoint_path = str(Path(csv_file).parent / f"checkpoint_{min_loss_idx}")
            
            results.append({
                'checkpoint_path': checkpoint_path,
                'validation_loss': min_loss_row['validation_loss'],
                'trial_dir': os.path.basename(os.path.dirname(csv_file))  # Add trial identifier
            })
            
        except Exception as e:
            print(f"Skipping {csv_file} due to error: {str(e)}")
            continue

    if not results:
        raise ValueError("No processable data found in any CSV file")

    # Create sorted DataFrame and print all results
    results_df = (
        pd.DataFrame(results)
        .sort_values('validation_loss')
        .reset_index(drop=True)
    )
    
    # Print formatted results (tab-separated)
    for _, row in results_df.iterrows():
        print(f"{row['checkpoint_path']}\t{row['validation_loss']:.6f}")
    