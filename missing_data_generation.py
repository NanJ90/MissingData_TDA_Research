
# %%
#reproducibility 
import random
import numpy as np
import copy
#generating 30% missingness for MCAR, MAR and MNAR

#set random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

import pandas as pd

# Remove hardcoded data loading - data will be loaded in the generate_missing_data function
# X = pd.read_csv('data/eeg_eye_state_full.csv')
# features_v1 = copy.deepcopy(X)
# print(f'Original data NaNs: {np.isnan(features_v1).sum().sum()}')

#%%
# --- MCAR (Missing Completely at Random) ---
def generate_mcar_mask(data, missing_rate):
    """Generates a mask for MCAR missing data."""
    mask = np.random.rand(*data.shape) < missing_rate
    return mask

# --- MAR (Missing at Random) ---
def generate_mar_mask(data, missing_rate, dependent_feature_index):
    """Generates a mask for MAR missing data with a target missing rate."""
    mask = np.zeros_like(data, dtype=bool)

    # Identify rows meeting the condition
    condition_met = data[:, dependent_feature_index] > np.mean(data[:, dependent_feature_index])

    # Calculate required missing values for these rows
    num_missing_needed = int(missing_rate * data.shape[0])
    num_eligible_rows = np.sum(condition_met)

    # Adjust missing rate for eligible rows to reach target
    adjusted_missing_rate = min(1.0, num_missing_needed / num_eligible_rows)

    # Introduce missing values for eligible rows
    for i in range(data.shape[0]):
        if condition_met[i]:
            mask[i, :] = np.random.rand(data.shape[1]) < adjusted_missing_rate
    return mask

# --- MNAR (Missing Not at Random) ---
def generate_mnar_mask(data, missing_rate, feature_index):
    """Generates a mask for MNAR missing data with a target missing rate."""
    mask = np.zeros_like(data, dtype=bool)

    # Calculate required missing values for these rows
    num_missing_needed = int(missing_rate * data.shape[0])
    total_rows = data.shape[0]

    # Find indices to introduce missingness randomly with higher probabilities above the percentile
    sorted_indices = np.argsort(data[:, feature_index])
    num_to_mask = min(num_missing_needed, total_rows)  # Ensure we don't exceed total rows

    # Create probabilities: higher probability for higher values (MNAR characteristic)
    # Use a more robust approach
    num_low_prob = int(total_rows * (1 - missing_rate))
    num_high_prob = total_rows - num_low_prob
    
    # Ensure the arrays have the right size
    probs = np.concatenate([
        np.full(num_low_prob, 0.1),      # Low probability for lower values
        np.full(num_high_prob, 0.9)      # High probability for higher values
    ])
    
    # Ensure probs array matches sorted_indices length exactly
    if len(probs) != len(sorted_indices):
        # Adjust to exact length needed
        if len(probs) > len(sorted_indices):
            probs = probs[:len(sorted_indices)]
        else:
            # Pad with the last value
            probs = np.pad(probs, (0, len(sorted_indices) - len(probs)), 'edge')
    
    # Normalize probabilities to sum to 1
    probs = probs / probs.sum()
    
    # Verify sizes match before calling np.random.choice
    assert len(sorted_indices) == len(probs), f"Size mismatch: indices={len(sorted_indices)}, probs={len(probs)}"

    mask_indices = np.random.choice(sorted_indices, size=num_to_mask,
                                   replace=False,
                                   p=probs)

    # Assign missingness based on chosen indices
    mask[mask_indices, :] = True

    return mask



# mcar_mask = generate_mcar_mask(features_v1, 0.3)
# mar_mask = generate_mar_mask(features_v1.to_numpy(), 0.3, dependent_feature_index=2)
# mnar_mask = generate_mnar_mask(features_v1.to_numpy(), 0.3, feature_index=2)

# #%%
# # Apply masks to create data with missing values
# data_mcar = np.ma.masked_array(features_v1, mask=mcar_mask)
# data_mar = np.ma.masked_array(features_v1.to_numpy(), mask=mar_mask)
# data_mnar = np.ma.masked_array(features_v1.to_numpy(), mask=mnar_mask)
# #%%
# #checking missing value by counting mask
# print(f'MCAR missing values: {np.sum(mcar_mask)}')
# print(f'MAR missing values: {np.sum(mar_mask)}')
# print(f'MNAR missing values: {np.sum(mnar_mask)}')
# #%%
# #converting masked arrays to NaN
# data_mcar = data_mcar.filled(np.nan)
# data_mar = data_mar.filled(np.nan)
# data_mnar = data_mnar.filled(np.nan)
# #%%
# #checking missing value
# # print(mcar_mask)
# print(np.isnan(data_mcar).sum())
# print(np.isnan(data_mar).sum())
# print(np.isnan(data_mnar).sum())
# %%
def generate_missing_data(file_path, missing_rate=0.3, dependent_feature_index=2, feature_index=2):
    """
    Generates datasets with MCAR, MAR, and MNAR missing values.

    Args:
        file_path (str): Path to the input CSV file.
        missing_rate (float): Percentage of missingness to introduce.
        dependent_feature_index (int): Index of the feature for MAR missingness.
        feature_index (int): Index of the feature for MNAR missingness.

    Returns:
        tuple: Datasets with MCAR, MAR, and MNAR missing values as pandas DataFrames.
    """
    # Try to automatically detect datetime columns for index
    try:
        # First, try reading normally to detect structure
        X_temp = pd.read_csv(file_path)
        
        # Check for common datetime column names and patterns
        datetime_candidates = []
        for col in X_temp.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
                datetime_candidates.append(col)
        
        # If we found datetime candidates, try to use the first one as index
        if datetime_candidates:
            datetime_col = datetime_candidates[0]
            print(f"Found potential datetime column: '{datetime_col}'. Setting as index.")
            X = pd.read_csv(file_path, index_col=datetime_col, parse_dates=True)
        else:
            # Check if first column looks like a datetime
            first_col = X_temp.columns[0]
            sample_values = X_temp[first_col].head().astype(str)
            if any(char in str(sample_values.iloc[0]) for char in ['-', '/', ':']):
                print(f"First column '{first_col}' appears to be datetime. Setting as index.")
                X = pd.read_csv(file_path, index_col=0, parse_dates=True)
            else:
                X = X_temp
    except:
        # If anything fails, just read normally
        X = pd.read_csv(file_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Columns: {list(X.columns)}")
    print(f"Index: {X.index.name if X.index.name else 'RangeIndex'}")
    
    # Separate features and target - do not add missing values to target column
    if 'target' in X.columns:
        target_col = X['target'].copy()
        features_v1 = copy.deepcopy(X.drop(columns=['target']))
    else:
        target_col = None
        features_v1 = copy.deepcopy(X)

    print(f"Features for missing data generation: {list(features_v1.columns)}")
    print(f"Features shape: {features_v1.shape}")
    print(f"Feature indices - MAR: {dependent_feature_index}, MNAR: {feature_index}")

    mcar_mask = generate_mcar_mask(features_v1, missing_rate)
    mar_mask = generate_mar_mask(features_v1.to_numpy(), missing_rate, dependent_feature_index)
    mnar_mask = generate_mnar_mask(features_v1.to_numpy(), missing_rate, feature_index)

    data_mcar = np.ma.masked_array(features_v1, mask=mcar_mask).filled(np.nan)
    data_mar = np.ma.masked_array(features_v1.to_numpy(), mask=mar_mask).filled(np.nan)
    data_mnar = np.ma.masked_array(features_v1.to_numpy(), mask=mnar_mask).filled(np.nan)

    # Convert back to DataFrames with correct feature columns and preserve index
    data_mcar_df = pd.DataFrame(data_mcar, columns=features_v1.columns, index=features_v1.index)
    data_mar_df = pd.DataFrame(data_mar, columns=features_v1.columns, index=features_v1.index)
    data_mnar_df = pd.DataFrame(data_mnar, columns=features_v1.columns, index=features_v1.index)
    
    # Add target column back (without missing values)
    if target_col is not None:
        data_mcar_df['target'] = target_col
        data_mar_df['target'] = target_col
        data_mnar_df['target'] = target_col

    return ( 
        data_mcar_df, 
        data_mar_df, 
        data_mnar_df, 
        mcar_mask,
        mar_mask, 
        mnar_mask
    )
# %%
if __name__ == "__main__":
    import os
    import glob 
    
    # List available datasets
    print("Available dataset options:")
    data_dir = 'data'
    available_files = []

    if os.path.exists(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
#         pkl_files = glob.glob(os.path.join(data_dir, '*.pkl'))
        available_files = csv_files 
        # + pkl_files

    if available_files:
        for i, file in enumerate(available_files, 1):
            print(f"{i}. {file}")
        print(f"{len(available_files) + 1}. Custom path")
        print(f"{len(available_files) + 2}. Default (data/eeg_eye_state_full.csv)")
        
        choice = input(f"\nSelect dataset (1-{len(available_files) + 2}) or press Enter for default: ").strip()
        try:
            choice_num = int(choice) if choice else len(available_files) + 2
            if 1 <= choice_num <= len(available_files):
                file_path = available_files[choice_num - 1]
            elif choice_num == len(available_files) + 1:
                file_path = input("Enter the path to your file: ").strip()
            else:  # Default option
                file_path = 'data/eeg_eye_state_full.csv'
        except ValueError:
            file_path = 'data/eeg_eye_state_full.csv'
    else:
        print("No files found in 'data' directory.")
        print("1. Enter custom path")
        print("2. Use default (data/eeg_eye_state_full.csv)")
        
        choice = input("Select option (1/2) or press Enter for default: ").strip()
        if choice == "1":
            file_path = input("Enter the path to your file: ").strip()
        else:
            file_path = 'data/eeg_eye_state_full.csv'

    print(f"Selected file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        exit(1)
        
    # Get user input for missing rate
    missing_rate_input = input("Enter missing rate (0.0-1.0, default: 0.3): ").strip()
    try:
        missing_rate = float(missing_rate_input) if missing_rate_input else 0.3
        if not 0.0 <= missing_rate <= 1.0:
            print("Warning: Missing rate should be between 0.0 and 1.0. Using default 0.3")
            missing_rate = 0.3
    except ValueError:
        print("Invalid input for missing rate. Using default 0.3")
        missing_rate = 0.3
    
    # Get user input for dependent feature index (for MAR)
    dep_feature_input = input("Enter dependent feature index for MAR (default: 2): ").strip()
    try:
        dependent_feature_index = int(dep_feature_input) if dep_feature_input else 2
    except ValueError:
        print("Invalid input for dependent feature index. Using default 2")
        dependent_feature_index = 2
    
    # Get user input for feature index (for MNAR)
    feature_input = input("Enter feature index for MNAR (default: 2): ").strip()
    try:
        feature_index = int(feature_input) if feature_input else 2
    except ValueError:
        print("Invalid input for feature index. Using default 2")
        feature_index = 2
    
    print(f"\nGenerating missing data with:")
    print(f"File: {file_path}")
    print(f"Missing rate: {missing_rate}")
    print(f"MAR dependent feature index: {dependent_feature_index}")
    print(f"MNAR feature index: {feature_index}")
    print("Processing...")
    
    # Generate missing data
    data_mcar, data_mar, data_mnar, mcar_mask, mar_mask, mnar_mask = generate_missing_data(
        file_path, missing_rate, dependent_feature_index, feature_index
    )
    #saving to incomplete data directory
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    datasets_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f'incomplete_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, datasets_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    #save datasets to pickle file
    datasets = {
        'mcar': data_mcar,
        'mar': data_mar,
        'mnar': data_mnar,
        'mcar_mask': mcar_mask,
        'mar_mask': mar_mask,
        'mnar_mask': mnar_mask
    }
    # for key, df in datasets.items():
    #     print(key)
        # print(df[0:5])  # Print first 5 rows for verification
    with open(os.path.join(output_dir, f'{datasets_name}.pkl'), 'wb') as f:
        import pickle
        pickle.dump(datasets, f)
    

    # Print statistics
    print(f'\nResults:')
    print(f'MCAR missing values: {np.isnan(data_mcar).sum().sum()}')
    print(f'MAR missing values: {np.isnan(data_mar).sum().sum()}')
    print(f'MNAR missing values: {np.isnan(data_mnar).sum().sum()}')
    
    # # Ask if user wants to save the results
    # save_input = input("\nSave the generated datasets? (y/N): ").strip().lower()
    # if save_input in ['y', 'yes']:
    #     output_dir = input("Enter output directory (default: 'incomplete_data'): ").strip()
    #     if not output_dir:
    #         output_dir = 'incomplete_data'
        
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
        
    #     # Get base filename for output
    #     base_name = os.path.splitext(os.path.basename(file_path))[0]
        
    #     # Save datasets
    #     mcar_path = os.path.join(output_dir, f"{base_name}_mcar.csv")
    #     mar_path = os.path.join(output_dir, f"{base_name}_mar.csv")
    #     mnar_path = os.path.join(output_dir, f"{base_name}_mnar.csv")
        
    #     data_mcar.to_csv(mcar_path, index=False)
    #     data_mar.to_csv(mar_path, index=False)
    #     data_mnar.to_csv(mnar_path, index=False)
        
    #     print(f"\nDatasets saved:")
    #     print(f"MCAR: {mcar_path}")
    #     print(f"MAR: {mar_path}")
    #     print(f"MNAR: {mnar_path}")
    
    print("Done!")
# %%
