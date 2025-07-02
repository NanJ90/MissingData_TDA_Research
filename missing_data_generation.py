
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

X = pd.read_csv('data/eeg_eye_state_full.csv')

features_v1 = copy.deepcopy(X)
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

    # Find indices to introduce missingness randomly with higher probabilities above the percentile
    # but still allowing some to below it.
    sorted_indices = np.argsort(data[:, feature_index])
    num_to_mask = num_missing_needed

    # Create probabilities with zeros and ones, then normalize to sum to 1
    probs = np.concatenate([np.zeros(int(len(sorted_indices)*(1-missing_rate))),np.ones(int(len(sorted_indices)*missing_rate))])
    probs /= probs.sum() # Normalize probabilities to sum to 1

    mask_indices = np.random.choice(sorted_indices, size=num_to_mask,
                                   replace=False,
                                   # Probability weighted for above percentile
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
    X = pd.read_csv(file_path)
    
    # Separate features and target - do not add missing values to target column
    if 'target' in X.columns:
        target_col = X['target'].copy()
        features_v1 = copy.deepcopy(X.drop(columns=['target']))
    else:
        target_col = None
        features_v1 = copy.deepcopy(X)

    mcar_mask = generate_mcar_mask(features_v1, missing_rate)
    mar_mask = generate_mar_mask(features_v1.to_numpy(), missing_rate, dependent_feature_index)
    mnar_mask = generate_mnar_mask(features_v1.to_numpy(), missing_rate, feature_index)

    data_mcar = np.ma.masked_array(features_v1, mask=mcar_mask).filled(np.nan)
    data_mar = np.ma.masked_array(features_v1.to_numpy(), mask=mar_mask).filled(np.nan)
    data_mnar = np.ma.masked_array(features_v1.to_numpy(), mask=mnar_mask).filled(np.nan)

    # Convert back to DataFrames with correct feature columns
    data_mcar_df = pd.DataFrame(data_mcar, columns=features_v1.columns)
    data_mar_df = pd.DataFrame(data_mar, columns=features_v1.columns)
    data_mnar_df = pd.DataFrame(data_mnar, columns=features_v1.columns)
    
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
    data_mcar, data_mar, data_mnar, mcar_mask, mar_mask, mnar_mask = generate_missing_data('data/eeg_eye_state_full.csv')
    # print(f'MCAR missing values: {np.isnan(data_mcar).sum().sum()}')
    # print(f'MAR missing values: {np.isnan(data_mar).sum().sum()}')
    # print(f'MNAR missing values: {np.isnan(data_mnar).sum().sum()}')
# %%
