# %%
import numpy as np
import pandas as pd
import time 
import os 
import sys
import pickle
import importlib
import copy
import missing_data_generation
importlib.reload(missing_data_generation)

# Configuration section - easily modifiable parameters
CONFIG = {
    'default_data_path': 'data/eeg_eye_state_full.csv',
    'missing_rate': 0.3,
    'mar_dependent_feature_idx': 2,
    'mnar_feature_idx': 2,
    'random_state': 42,
    'knn_neighbors': 3,
    'interpolation_method': 'linear',
    'gan_epochs': 1000,
    'gan_learning_rate': 0.001,
    'output_dir': 'imp_data'
}

from missing_data_generation import generate_missing_data

# %%
if __name__ == "__main__":
    import os 
    import glob
    root_dir = 'incomplete_data'
    print("Available missing datasets are:")
    # List all CSV files in the data directory
    data_files = glob.glob('data/*.csv')
    
    if not data_files:
        print("No data files found in 'data/' directory.")
        sys.exit(1)
    for i, file in enumerate(data_files):
        print(f"{i+1}. {os.path.basename(file)}")
        #using number to select the file and locate the incomplete data path = imcomplete_data/{dataset_name}/{dataset_name}.pkl
    choice = input(f"Select a dataset by number (1-{len(data_files)}): ").strip()
    try:
        choice_num = int(choice) if choice else len(data_files) + 2
        if 1 <= choice_num <= len(data_files):
            dataset_name = os.path.basename(data_files[choice_num - 1]).replace('.csv', '')

            print(f"Selected dataset: {dataset_name}")
            data_path = f"{root_dir}/{dataset_name}/{dataset_name}.pkl"
            if os.path.exists(data_path):
                print(f"Loading data from: {data_path}")
                print("📦 Loading data...")
                with open(data_path, 'rb') as f:
                    datasets = pickle.load(f)
                    data_mcar = datasets['mcar']
                    data_mar = datasets['mar']
                    data_mnar = datasets['mnar']
                    mcar_mask = datasets['mcar_mask']
                    mar_mask = datasets['mar_mask']
                    mnar_mask = datasets['mnar_mask']
                    print(data_mcar.head())
    except ValueError:
        print("Invalid input. Using default dataset.")
        

    # # Option 1: Use command line arguments
    # if len(sys.argv) > 1:
    #     data_path = sys.argv[1]
    #     print(f"Using data path from command line: {data_path}")
    # # Option 2: Use environment variable
    # elif 'DATA_PATH' in os.environ:
    #     data_path = os.environ['DATA_PATH']
    #     print(f"Using data path from environment variable: {data_path}")
    # # Option 3: Interactive input (default)
    # else:
    #     data_path = None
    #     print("No data path specified. Will prompt for input.")
    #incomplete data path = imcomplete_data/{dataset_name}/{dataset_name}.pkl

    # Generate missing data
#     data_mcar, data_mar, data_mnar, mcar_mask, mar_mask, mnar_mask = load_and_generate_missing_data(data_path)

#     # Set random state for reproducibility
#     random_state = CONFIG['random_state']
#     np.random.seed(random_state)
# else:
#     # When imported as module, don't generate data automatically
#     print("📦 Imputation module imported successfully!")
#     print("💡 Use load_and_generate_missing_data() function to load your data")
#     data_mcar = data_mar = data_mnar = None
#     mcar_mask = mar_mask = mnar_mask = None
def load_and_generate_missing_data(data_path=None, missing_rate=None, dep_feature_idx=None, feature_idx=None):
    """
    Load data and generate missing data patterns.
    
    Args:
        data_path (str): Path to the CSV file. If None, will use CONFIG or prompt user.
        missing_rate (float): Percentage of missing data. If None, uses CONFIG.
        dep_feature_idx (int): Feature index for MAR dependency. If None, uses CONFIG.
        feature_idx (int): Feature index for MNAR. If None, uses CONFIG.
    
    Returns:
        tuple: Generated missing data (mcar, mar, mnar) and masks.
    """
    # Use CONFIG defaults if not provided
    missing_rate = missing_rate or CONFIG['missing_rate']
    dep_feature_idx = dep_feature_idx or CONFIG['mar_dependent_feature_idx']
    feature_idx = feature_idx or CONFIG['mnar_feature_idx']
    
    if data_path is None:
        # Get user input for data path
        default_path = CONFIG['default_data_path']
        data_path = input(f"Enter the path to your CSV file (default: '{default_path}'): ").strip()
        if not data_path:
            data_path = default_path
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    print(f"Missing rate: {missing_rate}")
    print(f"MAR dependent feature index: {dep_feature_idx}")
    print(f"MNAR feature index: {feature_idx}")
    
    return generate_missing_data(data_path, missing_rate, dep_feature_idx, feature_idx)

# Only run data loading when script is executed directly, not when imported

# %%
# Import required libraries for imputation methods
from sklearn.impute import KNNImputer
import torch, torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import Adam

# Set PyTorch random seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# cuda = True if torch.cuda.is_available() else False

def knn_imputer(data, mask, k=None):
    """Apply KNN imputation to features only, preserving target column."""
    k = k or CONFIG['knn_neighbors']
    
    # Separate features and target
    if 'target' in data.columns:
        target_col = data['target'].copy()
        features_data = data.drop(columns=['target']).to_numpy()
    else:
        target_col = None
        features_data = data.to_numpy()
    
    # Apply mask and imputation to features only
    data_with_nan = np.where(mask, np.nan, features_data)
    knn_imputer_model = KNNImputer(n_neighbors=k)
    knn_data = knn_imputer_model.fit_transform(data_with_nan)
    
    # Convert back to DataFrame and add target column
    if 'target' in data.columns:
        result_df = pd.DataFrame(knn_data, columns=data.drop(columns=['target']).columns, index=data.index)
        result_df['target'] = target_col
        return result_df
    else:
        return pd.DataFrame(knn_data, columns=data.columns, index=data.index)

# Note: The following execution code is commented out to allow module import
# Uncomment and run these sections when using the script directly
start_time_knn = time.time()
knn_data_mcar = knn_imputer(data_mcar, mcar_mask)
knn_data_mar = knn_imputer(data_mar, mar_mask)
knn_data_mnar = knn_imputer(data_mnar, mnar_mask)
end_time_knn = time.time()
# %%

def interpolate_imputer(data, method='linear'):
    """
    Fills missing values in features using pandas interpolate, preserving target column.

    Args:
        data: The DataFrame with missing values.
        method: The interpolation method ('linear', 'nearest', 'cubic', etc.).

    Returns:
        The imputed DataFrame with missing values filled in features only.
    """
    df = copy.deepcopy(data)
    
    # Separate features and target
    if 'target' in df.columns:
        target_col = df['target'].copy()
        features_df = df.drop(columns=['target'])
    else:
        target_col = None
        features_df = df
    
    print(f'Before imp data has missing value of {features_df.isnull().sum().sum()}')
    
    # Apply interpolation to features only
    features_interpolated = features_df.interpolate(method=method, axis=0, limit_direction='both')
    
    # Add target column back
    if target_col is not None:
        features_interpolated['target'] = target_col
    
    return features_interpolated

interpolated_data_mcar = interpolate_imputer(data_mcar)
interpolated_data_mar = interpolate_imputer(data_mar)
interpolated_data_mnar = interpolate_imputer(data_mnar)
end_time_interpolate = time.time()
# %%
start_time_locf = time.time()
def locf_imputer(data, mask):
    """
    Fills missing values in features using Last Observation Carried Forward (LOCF), preserving target column.

    Args:
        data: The DataFrame with missing values.
        mask: The mask indicating missing values (not used in this implementation).

    Returns:
        The imputed DataFrame with missing values filled using LOCF in features only.
    """
    df = copy.deepcopy(data)
    
    # Separate features and target
    if 'target' in df.columns:
        target_col = df['target'].copy()
        features_df = df.drop(columns=['target'])
    else:
        target_col = None
        features_df = df
    
    # Apply LOCF to features only
    features_filled = features_df.fillna(method='ffill').fillna(method='bfill')
    
    # Add target column back
    if target_col is not None:
        features_filled['target'] = target_col
    
    return features_filled

locf_data_mcar = locf_imputer(data_mcar, mcar_mask)
locf_data_mar = locf_imputer(data_mar, mar_mask)
locf_data_mnar = locf_imputer(data_mnar, mnar_mask)
end_time_locf = time.time()

# %%

start_time_gan = time.time()
class GAIN(nn.Module):
    def __init__(self, dim, hint_rate=0.9):
        super().__init__()
        self.dim = dim
        self.generator = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
            nn.Sigmoid()
        )
        self.hint_rate = hint_rate

    def forward(self, data_x, mask):
        z = torch.rand_like(data_x)
        data_hat = mask * data_x + (1 - mask) * z
        hint = torch.rand_like(mask)
        hint = (hint < self.hint_rate).float() * mask
        g_input = torch.cat([data_hat, mask], dim=1)
        d_input = torch.cat([data_hat, hint], dim=1)

        g_sample = self.generator(g_input)
        x_hat = mask * data_x + (1 - mask) * g_sample
        d_prob = self.discriminator(d_input)
        return x_hat, d_prob, g_sample

def gain_impute(data_missing, epochs=1000, lr=0.001):
    """Apply GAIN imputation to features only, preserving target column."""
    original_data = data_missing.copy()
    
    # Separate features and target
    if 'target' in data_missing.columns:
        target_col = data_missing['target'].copy()
        features_data = data_missing.drop(columns=['target'])
        print(f'Initial NaN count in features: {features_data.isnull().sum().sum()}')
        features_numpy = features_data.to_numpy()
    else:
        target_col = None
        features_numpy = data_missing.to_numpy()
        print(f'Initial NaN count: {data_missing.isnull().sum().sum()}')
    
    mask = ~np.isnan(features_numpy)

    #Normalized data to [0, 1] range and fill NaNs with 0
    data_normalized = np.copy(features_numpy)
    for col in range(features_numpy.shape[1]):
        column_data = features_numpy[:, col]
        valid_data = column_data[~np.isnan(column_data)]
        if len(valid_data) > 0:
            min_val, max_val = np.min(valid_data), np.max(valid_data)
            data_normalized[:, col] = (column_data - min_val) / (max_val - min_val + 1e-6)
        else:
            # If column entirely NaN, fill with random numbers or zeros
            data_normalized[:, col] = np.random.uniform(0, 1, size=features_numpy.shape[0])

    
    random_data = np.random.uniform(0,1,size=features_numpy.shape)
    data_normalized[np.isnan(data_normalized)] = random_data[np.isnan(data_normalized)]
    assert not np.isnan(data_normalized).any(), "NaNs still exist after random filling!"

    #convert to tensor
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    assert not torch.isnan(data_tensor).any(), "NaNs in data_tensor!"
    mask_tensor = torch.tensor(mask.astype(float), dtype=torch.float32)
    print(f'Missing values: {(~mask).sum()}')
    print(f'Data range: [{data_tensor.min().item():.2f}, {data_tensor.max().item():.2f}]')

    model = GAIN(data_tensor.shape[1])
    opt = Adam(model.parameters(), lr=lr)

    for _ in tqdm(range(epochs)):
        model.train()
        x_hat, d_prob, g_sample = model(data_tensor, mask_tensor)
        d_loss = torch.mean(mask_tensor * torch.log(d_prob + 1e-8))
        g_loss = torch.mean((mask_tensor * data_tensor - mask_tensor * g_sample)**2)
        loss = -d_loss + g_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    x_hat, _, _ = model(data_tensor, mask_tensor)
    imputed_data = x_hat.detach().numpy()

    for col in range(imputed_data.shape[1]):
        column_data = features_numpy[:, col]
        valid_data = column_data[~np.isnan(column_data)]
        if len(valid_data) > 0:
            min_val, max_val = np.min(valid_data), np.max(valid_data)
            imputed_data[:, col] = imputed_data[:, col] * (max_val - min_val) + min_val
    
    # Convert back to DataFrame and add target column
    if target_col is not None:
        result_df = pd.DataFrame(imputed_data, columns=features_data.columns)
        result_df['target'] = target_col
        return result_df
    else:
        return pd.DataFrame(imputed_data, columns=data_missing.columns)

def gain_impute_old(data_missing, epochs=1000, lr=0.001):
    # Convert DataFrame to NumPy array if necessary
    if isinstance(data_missing, pd.DataFrame):
        print(f'Initial NaN count: {data_missing.isnull().sum().sum()}')
        data_missing = data_missing.to_numpy()
    # print(f"Data shape: {data_missing[:5]}")
    print(f'NaN count after conversion: {np.isnan(data_missing).sum()}')
    data_tensor = torch.tensor(data_missing, dtype=torch.float32)
    print(f'NaN count in data tensor: {torch.isnan(data_tensor).sum().item()}')
    mask_tensor = (~np.isnan(data_missing)).astype(float)
    mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)
    print(f'Number of masked values: {(mask_tensor==0).sum()}')
    model = GAIN(data_tensor.shape[1])
    opt = Adam(model.parameters(), lr=lr)

    for _ in tqdm(range(epochs)):
        model.train()
        x_hat, d_prob, g_sample = model(data_tensor, mask_tensor)
        d_loss = torch.mean(mask_tensor * torch.log(d_prob + 1e-8))
        g_loss = torch.mean((mask_tensor * data_tensor - mask_tensor * g_sample)**2)
        loss = -d_loss + g_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    x_hat, _, _ = model(data_tensor, mask_tensor)
    return x_hat.detach().numpy()

# Example usage:
# imputed_gain = gain_impute(data_missing)
# bottleneck_gain = tda_distance(data, imputed_gain)
gan_data_mcar = gain_impute(data_mcar)
gan_data_mar = gain_impute(data_mar)
gan_data_mnar = gain_impute(data_mnar)
# print(f"Bottleneck distance (GAIN Imputation): {bottleneck_gain:.4f}")
end_time_gan = time.time()
# %%
# print(gan_data_mcar[:10])
# %%
print(f"KNN Imputation Time: {end_time_knn - start_time_knn:.2f} seconds")
# print(f"Interpolation Imputation Time: {end_time_interpolate - start:.2f} seconds")x
print(f"LOCF Imputation Time: {end_time_locf - start_time_locf:.2f} seconds")
print(f"GAIN Imputation Time: {end_time_gan - start_time_gan:.2f} seconds")

#%%
#verifying there is no masked data in each imputation method
# knn
print(f' KNN imputation check')
print(f'MCAR: {knn_data_mcar.isnull().sum().sum()}')
print(f'MAR: {knn_data_mar.isnull().sum().sum()}')
print(f'MNAR: {knn_data_mnar.isnull().sum().sum()}')
#interpolation
print(f'\n Interpolation imputation check')
print(f'MCAR: {interpolated_data_mcar.isnull().sum().sum()}')
print(f'MAR: {interpolated_data_mar.isnull().sum().sum()}')
print(f'MNAR: {interpolated_data_mnar.isnull().sum().sum()}')
# #last observation forward carry
print(f'\n LOCF imputation check')
print(f'MCAR: {locf_data_mcar.isnull().sum().sum()}')
print(f'MAR: {locf_data_mar.isnull().sum().sum()}')
print(f'MNAR: {locf_data_mnar.isnull().sum().sum()}')
# #GAN
print(f'\n GAN imputation check')
print(f'MCAR: {gan_data_mcar.isnull().sum().sum()}')
print(f'MAR: {gan_data_mar.isnull().sum().sum()}')
print(f'MNAR: {gan_data_mnar.isnull().sum().sum()}')
# %%
imp_data_dir = 'imp_data'
if not os.path.exists(imp_data_dir):
    os.makedirs(imp_data_dir, exist_ok=True)

# Define a mapping of data and filenames - now all are DataFrames
datasets = {
    "knn_data_mcar": knn_data_mcar,
    "knn_data_mar": knn_data_mar,
    "knn_data_mnar": knn_data_mnar,
    "interpolated_data_mcar": interpolated_data_mcar,
    "interpolated_data_mar": interpolated_data_mar,
    "interpolated_data_mnar": interpolated_data_mnar,
    "locf_data_mcar": locf_data_mcar,
    "locf_data_mar": locf_data_mar,
    "locf_data_mnar": locf_data_mnar,
    "gan_data_mcar": gan_data_mcar,
    "gan_data_mar": gan_data_mar,
    "gan_data_mnar": gan_data_mnar,
}

# Loop through the datasets and save them to CSV
# for filename, data in datasets.items():
    # data.to_csv(f"{root_dir}/{filename}.csv", index=False)
# saving datasets to a pickle file
time_stamp = time.strftime("%Y%m%d_%H%M%S")
with open(f"{imp_data_dir}/imputed_datasets_{time_stamp}.pkl", 'wb') as f:
    pickle.dump(datasets, f)

# if __name__ == "__main__":
#     print(f"Imputed data saved to {root_dir} directory.")
#     print(f"Total time taken for all imputations: {end_time_gan - start_time_knn:.2f} seconds")
#     print("All imputations completed successfully!")
# %%
#saving gan imputed data
# pd.DataFrame(gan_data_mcar, columns=data_mcar.columns).to_csv(f"imp_data/gan_data_mcar.csv", index=False)
# pd.DataFrame(gan_data_mar, columns=data_mar.columns).to_csv(f"imp_data/gan_data_mar.csv", index=False)
# pd.DataFrame(gan_data_mnar, columns=data_mnar.columns).to_csv(f"imp_data/gan_data_mnar.csv", index=False)
#%%
# with open(f"imp_data/imputed_datasets_20250724_002055.pkl", 'rb') as f:
#     datasets = pickle.load(f)
#     for key, df in datasets.items():
#         print(key)
#         print(df.head())  # Print first 5 rows for verification