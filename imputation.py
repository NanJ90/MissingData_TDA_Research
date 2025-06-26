# %%
import numpy as np
import pandas as pd
import time 
import os 
import sys
import importlib
import missing_data_generation
importlib.reload(missing_data_generation)

from missing_data_generation import generate_missing_data

data_root = 'data'
if not os.path.exists(data_root):
    os.makedirs(data_root, exist_ok=True)

data_mcar, data_mar, data_mnar, mcar_mask, mar_mask, mnar_mask = generate_missing_data(f'{data_root}/eeg_eye_state_full.csv')
print(data_mcar.head())

# %%
start_time_knn = time.time()
from sklearn.impute import KNNImputer
def knn_imputer(data, mask, k=3):
    #convert mask to nan for KNNimputer to recognize

    # dd= data.copy()
    # print(dd[:10])
    data_with_nan = np.where(mask, np.nan, data)
    knn_imputer = KNNImputer(n_neighbors=k)
    knn_data = knn_imputer.fit_transform(data_with_nan)
    return knn_data

knn_data_mcar = knn_imputer(data_mcar, mcar_mask)
knn_data_mar = knn_imputer(data_mar, mar_mask)
knn_data_mnar = knn_imputer(data_mnar, mcar_mask)
end_time_knn = time.time()
# %%
import copy
start_time_interpolate = time.time()
def interpolate_imputer(data, method='linear'):
    """
    Fills missing values in a NumPy array using pandas interpolate.

    Args:
        data: The NumPy array with missing values (represented as np.nan).
        method: The interpolation method ('linear', 'nearest', 'cubic', etc.).
                See pandas.DataFrame.interpolate for options.

    Returns:
        The imputed array with missing values filled.
    """
    # Create a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # print(df)
    df = copy.deepcopy(data)  # Ensure we don't modify the original data
    # Impute missing values using pandas interpolate
    print(f'Before imp data has missign value of {df.isnull().sum()}')
    df_interpolated = df.interpolate(method=method, axis=0, limit_direction='both') #axis=0 to impute column-wise
    
    # Return the imputed data as a NumPy array
    return df_interpolated.to_numpy()

interpolated_data_mcar = interpolate_imputer(data_mcar)
interpolated_data_mar = interpolate_imputer(data_mar)
interpolated_data_mnar = interpolate_imputer(data_mnar)
end_time_interpolate = time.time()
# %%
start_time_locf = time.time()
def locf_imputer(data,mask):
    """
    Fills missing values in a masked NumPy array using Last Observation Carried Forward (LOCF).

    Args:
        data: The masked NumPy array with missing values (represented by the mask).

    Returns:
        The imputed array with missing values filled using LOCF.
    """
    dd = pd.DataFrame(data)
    # print(dd.iloc[:10])
    # print(f'Before imp data has missign value of {dd.isnull().sum().sum()}')
    # Fill trailing NaNs with the last valid observation
    df = dd.fillna(method='ffill').fillna(method='bfill')
    # print(df.iloc[:10])
    return df.to_numpy()

locf_data_mcar = locf_imputer(data_mcar,mcar_mask)
locf_data_mar = locf_imputer(data_mar, mar_mask)
locf_data_mnar = locf_imputer(data_mnar,mnar_mask)
end_time_locf = time.time()
#%%
import torch, torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch
import torch.nn as nn
# from gain import GAIN
# import torch
# from torch.optim import Adam
from tqdm import tqdm
from torch.optim import Adam
cuda = True if torch.cuda.is_available() else False
print(f"Using CUDA: {cuda}")
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
    if isinstance(data_missing, pd.DataFrame):
        print(f'Initial NaN count: {data_missing.isnull().sum().sum()}')
        data_missing = data_missing.to_numpy()
    
    mask = ~np.isnan(data_missing)

    #Normalized data to [0, 1] range and fill NaNs with 0
    data_normalized = np.copy(data_missing)
    for col in range(data_missing.shape[1]):
        column_data = data_missing[:, col]
        valid_data = column_data[~np.isnan(column_data)]
    if len(valid_data) > 0:
        min_val, max_val = np.min(valid_data), np.max(valid_data)
        data_normalized[:, col] = (column_data - min_val) / (max_val - min_val + 1e-6)
    else:
        # If column entirely NaN, fill with random numbers or zeros
        data_normalized[:, col] = np.random.uniform(0, 1, size=data_missing.shape[0])

    
    random_data = np.random.uniform(0,1,size=data_missing.shape)
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
        column_data = data_missing[:, col]
        valid_data = column_data[~np.isnan(column_data)]
        if len(valid_data) > 0:
            min_val, max_val = np.min(valid_data), np.max(valid_data)
            imputed_data[:, col] = imputed_data[:, col] * (max_val - min_val) + min_val
    return imputed_data

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
print(f"Interpolation Imputation Time: {end_time_interpolate - start_time_interpolate:.2f} seconds")
print(f"LOCF Imputation Time: {end_time_locf - start_time_locf:.2f} seconds")
print(f"GAIN Imputation Time: {end_time_gan - start_time_gan:.2f} seconds")

#%%
#verifying there is no masked data in each imputation method
# knn
print(f' KNN imputation check')
# print(f'Original data NaNs: {np.isnan(feature1.to_numpy()).sum()}')
print(f'MCAR: {np.isnan(knn_data_mcar).sum()}')
print(f'MAR: {np.isnan(knn_data_mar).sum()}')
print(f'MNAR: {np.isnan(knn_data_mnar).sum()}')
#interpolation
print(f'\n Interpolation imputation check')
print(f'MCAR: {np.isnan(interpolated_data_mcar).sum()}')
print(f'MAR: {np.isnan(interpolated_data_mar).sum()}')
print(f'MNAR: {np.isnan(interpolated_data_mnar).sum()}')
# #last observation forward carry
print(f'\n LOCF imputation check')
print(f'MCAR: {np.isnan(locf_data_mcar).sum()}')
print(f'MAR: {np.isnan(locf_data_mar).sum()}')
print(f'MNAR: {np.isnan(locf_data_mnar).sum()}')
# #GAN
print(f'\n GAN imputation check')
print(f'MCAR: {np.isnan(gan_data_mcar).sum()}')
print(f'MAR: {np.isnan(gan_data_mar).sum()}')
print(f'MNAR: {np.isnan(gan_data_mnar).sum()}')
# %%
root_dir = 'imp_data'
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)
# Define a mapping of data and filenames
datasets = {
    "knn_data_mcar": (knn_data_mcar, data_mcar.columns),
    "knn_data_mar": (knn_data_mar, data_mar.columns),
    "knn_data_mnar": (knn_data_mnar, data_mnar.columns),
    "interpolated_data_mcar": (interpolated_data_mcar, data_mcar.columns),
    "interpolated_data_mar": (interpolated_data_mar, data_mar.columns),
    "interpolated_data_mnar": (interpolated_data_mnar, data_mnar.columns),
    "locf_data_mcar": (locf_data_mcar, data_mcar.columns),
    "locf_data_mar": (locf_data_mar, data_mar.columns),
    "locf_data_mnar": (locf_data_mnar, data_mnar.columns),
    "gan_data_mcar": (gan_data_mcar, data_mcar.columns),
    "gan_data_mar": (gan_data_mar, data_mar.columns),
    "gan_data_mnar": (gan_data_mnar, data_mnar.columns),
}

# Loop through the datasets and save them to CSV
for filename, (data, columns) in datasets.items():
    pd.DataFrame(data, columns=columns).to_csv(f"{root_dir}/{filename}.csv", index=False)

if __name__ == "__main__":
    print(f"Imputed data saved to {root_dir} directory.")
    print(f"Total time taken for all imputations: {end_time_gan - start_time_knn:.2f} seconds")
    print("All imputations completed successfully!")
# %%
#saving gan imputed data
# pd.DataFrame(gan_data_mcar, columns=data_mcar.columns).to_csv(f"imp_data/gan_data_mcar.csv", index=False)
# pd.DataFrame(gan_data_mar, columns=data_mar.columns).to_csv(f"imp_data/gan_data_mar.csv", index=False)
# pd.DataFrame(gan_data_mnar, columns=data_mnar.columns).to_csv(f"imp_data/gan_data_mnar.csv", index=False)