#%%
import pandas as pd
import numpy as np
import os
import pickle
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration for multiple seed experiments - Updated to match Kaggle
EXPERIMENT_CONFIG = {
    'seeds': [42, 123, 456, 789, 999],  # Multiple seeds for robust evaluation
    'seq_length': 24,        # Kaggle uses 24-hour windows
    'hidden_size': 256,      # Kaggle uses 256 hidden units
    'num_epochs': 150,       # Kaggle uses 150 epochs
    'batch_size': 32,        # Kaggle uses batch_size=32
    'learning_rate': 0.001   # Standard Adam learning rate
}

print("🚀 Starting Multi-Seed LSTM Traffic Flow Forecasting for PEMS08 Dataset")
print(f"🎲 Running experiments with {len(EXPERIMENT_CONFIG['seeds'])} different seeds")
print("="*70)

# Load imputed datasets
with open("imp_data/imputed_datasets_20250724_002055.pkl", 'rb') as f:
    datasets = pickle.load(f)

# Load original dataset
orig_dataset = pd.read_csv("data/PEMS08.csv")

# Prepare all datasets (original + imputed)
all_datasets = {'original': orig_dataset}
for key, df in datasets.items():
    all_datasets[key] = copy.deepcopy(df)

print(f"📊 Loaded {len(all_datasets)} datasets for comparison:")
for name, df in all_datasets.items():
    print(f"  - {name}: {df.shape}")

#%%
# LSTM Model Definition - Updated to match Kaggle architecture
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Two LSTM layers like Kaggle model
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layers matching Kaggle architecture
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First LSTM layer (return_sequences=True equivalent)
        lstm1_out, _ = self.lstm1(x)
        
        # Second LSTM layer (return_sequences=False equivalent - take last output)
        lstm2_out, _ = self.lstm2(lstm1_out)
        out = lstm2_out[:, -1, :]  # Take last timestep
        
        # Dropout
        out = self.dropout1(out)
        
        # First Dense layer with ReLU
        out = self.fc1(out)
        out = self.relu(out)
        
        # Second Dropout
        out = self.dropout2(out)
        
        # Output layer (linear activation)
        out = self.fc2(out)
        
        return out

class TrafficLSTMSimple(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, output_size=1, dropout=0.2):
        super(TrafficLSTMSimple, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Larger LSTM like Kaggle (256 vs your 50)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        # Larger dense layer like Kaggle
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 256 -> 256
        self.fc2 = nn.Linear(hidden_size, output_size)   # 256 -> 1
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take last time step
        out = self.dropout(out[:, -1, :])
        
        # Dense layers with ReLU (like Kaggle)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

#%%
# Data preprocessing functions
# def create_sequences(data, seq_length, target_col_idx=2):
#     """
#     Create sequences for LSTM training
#     Args:
#         data: numpy array of shape (n_samples, n_features)
#         seq_length: length of input sequences
#         target_col_idx: index of target column (default 2 for 'cost')
#     Returns:
#         X: input sequences, y: target values
#     """
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         # Use all features as input
#         X.append(data[i:(i + seq_length), :])  
#         # Predict the 'cost' column (traffic flow)
#         y.append(data[i + seq_length, target_col_idx])  
#     return np.array(X), np.array(y)
# creating 3-dimensional array for [timestep, timeframe, features]
def create_dataset(traffic, location, WINDOW_SIZE):

    # mask a certain location
    location_current = traffic[traffic["location"]==location].reset_index()
    
    # group to hour and average 12 (5-minute) timesteps
    location_current["hour"] = ((location_current["timestep"] - 1) // 12)
    grouped = location_current.groupby("hour").mean().reset_index()
    
    # add hour features as mod 24 cycle (0...23)
    grouped['day'] = (grouped['hour'] // 24) % 7
    grouped['hour'] %= 24
    
    one_hot_hour = pd.get_dummies(grouped['hour'])
    one_hot_hour = one_hot_hour.add_prefix('hour_')
    
    # merge all the features together to get a total of 27 features
    hour_grouped = pd.concat([grouped[["occupy", "flow", "speed"]], one_hot_hour], axis=1)
    hour_grouped = np.array(hour_grouped)
    
    X, Y = [], []
    
    # add lag features (in reverse time order)
    for i in range(len(hour_grouped) - WINDOW_SIZE):
        X.append(hour_grouped[i:(i + WINDOW_SIZE)][::-1]) # reverse the order
        Y.append(hour_grouped[i + WINDOW_SIZE, 0]) # index 0 is occupy
    
    return X,Y # returns (timestep, timeframe, features) and (target)
# creating 4-th dimension for the locations
X, Y = [], []

for location in range(170):
    a,b = create_dataset(all_datasets['original'],location, WINDOW_SIZE=24)
    X.append(a)
    Y.append(b)
    
X = np.moveaxis(X,0,-1)
Y = np.moveaxis(Y,0,-1)

print(X.shape)
print(Y.shape)
#%%
def prepare_data(df, seq_length=10, train_ratio=0.8):
    """
    Prepare data for LSTM training - Updated to match Kaggle scaling approach
    """
    # Remove any rows with NaN values
    df_clean = df.dropna()
    
    # Create sequences first (before scaling)
    data_array = df_clean.to_numpy()
    X, y = create_sequences(data_array, seq_length)
    
    # Split into train and test sets BEFORE scaling
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Separate scalers for X and y (like Kaggle)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Scale X data: reshape to 2D, scale, then back to 3D
    original_X_train_shape = X_train.shape
    original_X_test_shape = X_test.shape
    
    X_train_reshaped = X_train.reshape(X_train.shape[0] * X_train.shape[1], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0] * X_test.shape[1], -1)
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(original_X_train_shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(original_X_test_shape)
    
    # Scale y data
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train_scaled)
    y_train = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test_scaled)
    y_test = torch.FloatTensor(y_test_scaled).reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test, scaler_X, scaler_y

#%%
# Training function
def train_lstm(model, train_loader, criterion, optimizer, num_epochs=100):
    """
    Train the LSTM model
    """
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return train_losses

#%%
# Evaluation function
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        
    # Convert to numpy for metrics calculation
    y_true = y_test.numpy().flatten()
    y_pred = predictions.numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'predictions': y_pred,
        'actual': y_true
    }

#%%
# Single seed experiment function
def run_single_seed_experiment(datasets_dict, seed, seq_length=10, hidden_size=256, num_epochs=150, batch_size=32):
    """
    Run LSTM forecasting experiment for a single seed - Updated for Kaggle-style model
    """
    # Set seeds for this experiment
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    results = {}
    
    for dataset_name, df in datasets_dict.items():
        try:
            # Prepare data with separate scalers
            X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(df, seq_length)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model - Use Kaggle-style architecture
            input_size = X_train.shape[2]  # Number of features
            
            # Choose between the two model architectures
            # model = TrafficLSTM(input_size=input_size, hidden_size=hidden_size)  # Exact Kaggle replica
            model = TrafficLSTMSimple(input_size=input_size, hidden_size=hidden_size)  # Simplified but larger
            
            # Loss and optimizer (matching Kaggle: MSE + Adam)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=EXPERIMENT_CONFIG['learning_rate'])
            
            # Train model (reduced verbosity for multiple seeds)
            train_losses = train_lstm(model, train_loader, criterion, optimizer, num_epochs)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)
            
            # Store results with both scalers
            results[dataset_name] = {
                'metrics': metrics,
                'train_losses': train_losses,
                'data_shape': df.shape,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
            
        except Exception as e:
            results[dataset_name] = {'error': str(e)}
    
    return results

results = run_single_seed_experiment(all_datasets, seed=42, seq_length=EXPERIMENT_CONFIG['seq_length'])
#%%
# Multi-seed forecasting pipeline
def run_multi_seed_experiment(datasets_dict, seeds, seq_length=10, hidden_size=50, num_epochs=100, batch_size=32):
    """
    Run LSTM forecasting experiment with multiple seeds and compute averages
    """
    all_results = {}
    
    print(f"\n🔄 Starting multi-seed LSTM forecasting experiments...")
    print(f"🎲 Seeds: {seeds}")
    print("="*70)
    
    # Run experiments for each seed
    for i, seed in enumerate(seeds, 1):
        print(f"\n🌱 Experiment {i}/{len(seeds)} - Seed: {seed}")
        print("-" * 50)
        
        seed_results = run_single_seed_experiment(
            datasets_dict, seed, seq_length, hidden_size, num_epochs, batch_size
        )
        
        # Progress update for each dataset
        for dataset_name, result in seed_results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"  ✅ {dataset_name:<20} - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
            else:
                print(f"  ❌ {dataset_name:<20} - Error: {result.get('error', 'Unknown')}")
        
        all_results[seed] = seed_results
    
    # Compute averaged results
    print(f"\n📊 Computing averaged results across {len(seeds)} seeds...")
    averaged_results = compute_averaged_results(all_results)
    
    return averaged_results, all_results

#%%
# Function to compute averaged results
def compute_averaged_results(all_results):
    """
    Compute mean and standard deviation of metrics across all seeds
    """
    dataset_names = set()
    for seed_results in all_results.values():
        dataset_names.update(seed_results.keys())
    
    averaged_results = {}
    
    for dataset_name in dataset_names:
        metrics_list = []
        
        # Collect metrics from all seeds
        for seed, seed_results in all_results.items():
            if dataset_name in seed_results and 'metrics' in seed_results[dataset_name]:
                metrics_list.append(seed_results[dataset_name]['metrics'])
        
        if metrics_list:
            # Compute averages and standard deviations
            avg_metrics = {}
            std_metrics = {}
            
            for metric_name in ['RMSE', 'MAE', 'MSE', 'R2']:
                values = [m[metric_name] for m in metrics_list]
                avg_metrics[metric_name] = np.mean(values)
                std_metrics[metric_name] = np.std(values)
            
            averaged_results[dataset_name] = {
                'mean_metrics': avg_metrics,
                'std_metrics': std_metrics,
                'num_successful_seeds': len(metrics_list),
                'sample_shape': metrics_list[0].get('predictions', []).shape if metrics_list else None
            }
        else:
            averaged_results[dataset_name] = {'error': 'No successful runs'}
    
    return averaged_results

#%%
# Run the multi-seed experiment
print("🚀 Starting comprehensive multi-seed LSTM forecasting comparison...")
averaged_results, all_seed_results = run_multi_seed_experiment(
    all_datasets, 
    seeds=EXPERIMENT_CONFIG['seeds'],
    seq_length=EXPERIMENT_CONFIG['seq_length'], 
    hidden_size=EXPERIMENT_CONFIG['hidden_size'],
    num_epochs=EXPERIMENT_CONFIG['num_epochs'],
    batch_size=EXPERIMENT_CONFIG['batch_size']
)

#%%
# Create performance comparison for averaged results
def create_averaged_performance_comparison(averaged_results):
    """
    Create a comprehensive performance comparison table with error bars
    """
    print("\n📊 MULTI-SEED PERFORMANCE COMPARISON RESULTS")
    print("="*90)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for dataset_name, result in averaged_results.items():
        if 'mean_metrics' in result:
            mean_metrics = result['mean_metrics']
            std_metrics = result['std_metrics']
            comparison_data.append({
                'Dataset': dataset_name,
                'RMSE_mean': mean_metrics['RMSE'],
                'RMSE_std': std_metrics['RMSE'],
                'MAE_mean': mean_metrics['MAE'], 
                'MAE_std': std_metrics['MAE'],
                'R2_mean': mean_metrics['R2'],
                'R2_std': std_metrics['R2'],
                'Successful_Seeds': result['num_successful_seeds']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE_mean')  # Sort by RMSE (lower is better)
        
        # Format the display
        for _, row in comparison_df.iterrows():
            print(f"{row['Dataset']:<25} | "
                  f"RMSE: {row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f} | "
                  f"MAE: {row['MAE_mean']:.4f}±{row['MAE_std']:.4f} | " 
                  f"R²: {row['R2_mean']:.4f}±{row['R2_std']:.4f} | "
                  f"Seeds: {row['Successful_Seeds']}")
        
        # Find best performing dataset
        best_dataset = comparison_df.iloc[0]['Dataset']
        print(f"\n🏆 Best performing dataset: {best_dataset}")
        print(f"   RMSE: {comparison_df.iloc[0]['RMSE_mean']:.4f} ± {comparison_df.iloc[0]['RMSE_std']:.4f}")
        print(f"   R²: {comparison_df.iloc[0]['R2_mean']:.4f} ± {comparison_df.iloc[0]['R2_std']:.4f}")
        
        return comparison_df
    else:
        print("❌ No successful results to compare")
        return None

# Generate comparison report
comparison_results = create_averaged_performance_comparison(averaged_results)

#%%
# Enhanced visualization functions for multi-seed results
def plot_multi_seed_results(averaged_results, all_seed_results):
    """
    Plot comprehensive multi-seed results with error bars
    """
    plt.figure(figsize=(16, 12))
    
    # Prepare data for plotting
    dataset_names = []
    rmse_means = []
    rmse_stds = []
    mae_means = []
    mae_stds = []
    r2_means = []
    r2_stds = []
    
    for dataset_name, result in averaged_results.items():
        if 'mean_metrics' in result:
            dataset_names.append(dataset_name)
            rmse_means.append(result['mean_metrics']['RMSE'])
            rmse_stds.append(result['std_metrics']['RMSE'])
            mae_means.append(result['mean_metrics']['MAE'])
            mae_stds.append(result['std_metrics']['MAE'])
            r2_means.append(result['mean_metrics']['R2'])
            r2_stds.append(result['std_metrics']['R2'])
    
    # RMSE comparison with error bars
    plt.subplot(2, 3, 1)
    bars = plt.bar(range(len(dataset_names)), rmse_means, yerr=rmse_stds, 
                   alpha=0.7, capsize=5, error_kw={'elinewidth': 2})
    plt.title('RMSE Comparison (Lower is Better)\nWith Standard Deviation')
    plt.ylabel('RMSE')
    plt.xticks(range(len(dataset_names)), dataset_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, rmse_means, rmse_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    # R² comparison with error bars
    plt.subplot(2, 3, 2)
    bars = plt.bar(range(len(dataset_names)), r2_means, yerr=r2_stds, 
                   alpha=0.7, color='green', capsize=5, error_kw={'elinewidth': 2})
    plt.title('R² Score Comparison (Higher is Better)\nWith Standard Deviation')
    plt.ylabel('R² Score')
    plt.xticks(range(len(dataset_names)), dataset_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, r2_means, r2_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    # MAE comparison with error bars
    plt.subplot(2, 3, 3)
    bars = plt.bar(range(len(dataset_names)), mae_means, yerr=mae_stds, 
                   alpha=0.7, color='orange', capsize=5, error_kw={'elinewidth': 2})
    plt.title('MAE Comparison (Lower is Better)\nWith Standard Deviation')
    plt.ylabel('MAE')
    plt.xticks(range(len(dataset_names)), dataset_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, mae_means, mae_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Box plots for RMSE across seeds
    plt.subplot(2, 3, 4)
    rmse_data_by_dataset = []
    labels = []
    for dataset_name in dataset_names:
        rmse_values = []
        for seed_results in all_seed_results.values():
            if dataset_name in seed_results and 'metrics' in seed_results[dataset_name]:
                rmse_values.append(seed_results[dataset_name]['metrics']['RMSE'])
        if rmse_values:
            rmse_data_by_dataset.append(rmse_values)
            labels.append(dataset_name)
    
    if rmse_data_by_dataset:
        plt.boxplot(rmse_data_by_dataset, labels=labels)
        plt.title('RMSE Distribution Across Seeds')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    
    # Method ranking comparison
    plt.subplot(2, 3, 5)
    method_performance = {}
    for dataset_name, result in averaged_results.items():
        if 'mean_metrics' in result:
            method = dataset_name.split('_')[0] if '_' in dataset_name else 'original'
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(result['mean_metrics']['RMSE'])
    
    methods = []
    avg_rmse = []
    for method, rmse_list in method_performance.items():
        methods.append(method)
        avg_rmse.append(np.mean(rmse_list))
    
    # Sort by performance
    sorted_data = sorted(zip(methods, avg_rmse), key=lambda x: x[1])
    methods, avg_rmse = zip(*sorted_data)
    
    bars = plt.bar(methods, avg_rmse, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple'][:len(methods)])
    plt.title('Average RMSE by Imputation Method')
    plt.ylabel('Average RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rmse in zip(bars, avg_rmse):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{rmse:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Variance analysis
    plt.subplot(2, 3, 6)
    plt.scatter(rmse_means, rmse_stds, s=100, alpha=0.7)
    for i, name in enumerate(dataset_names):
        plt.annotate(name, (rmse_means[i], rmse_stds[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Mean RMSE')
    plt.ylabel('RMSE Standard Deviation')
    plt.title('Variance vs Performance Trade-off')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_seed_lstm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate enhanced visualization
plot_multi_seed_results(averaged_results, all_seed_results)

#%%
# Summary and final output
def create_final_summary(averaged_results, num_seeds):
    """
    Create a final summary report
    """
    print(f"\n🎉 MULTI-SEED LSTM FORECASTING ANALYSIS COMPLETED!")
    print("="*70)
    print(f"📊 Total seeds used: {num_seeds}")
    print(f"📊 Total datasets compared: {len(averaged_results)}")
    
    # Find best dataset
    best_rmse = float('inf')
    best_dataset = None
    for dataset_name, result in averaged_results.items():
        if 'mean_metrics' in result:
            rmse = result['mean_metrics']['RMSE']
            if rmse < best_rmse:
                best_rmse = rmse
                best_dataset = dataset_name
    
    if best_dataset:
        best_result = averaged_results[best_dataset]
        print(f"\n🏆 BEST PERFORMING DATASET: {best_dataset}")
        print(f"   RMSE: {best_result['mean_metrics']['RMSE']:.4f} ± {best_result['std_metrics']['RMSE']:.4f}")
        print(f"   MAE:  {best_result['mean_metrics']['MAE']:.4f} ± {best_result['std_metrics']['MAE']:.4f}")
        print(f"   R²:   {best_result['mean_metrics']['R2']:.4f} ± {best_result['std_metrics']['R2']:.4f}")
        print(f"   Successful seeds: {best_result['num_successful_seeds']}/{num_seeds}")
    
    # Method ranking
    print(f"\n📈 IMPUTATION METHOD RANKING (by average RMSE):")
    method_performance = {}
    for dataset_name, result in averaged_results.items():
        if 'mean_metrics' in result:
            if dataset_name == 'original':
                method = 'Original Data'
            else:
                method_parts = dataset_name.split('_')
                if len(method_parts) >= 2:
                    method = method_parts[0].upper()
                else:
                    method = 'Unknown'
            
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(result['mean_metrics']['RMSE'])
    
    method_ranking = []
    for method, rmse_list in method_performance.items():
        avg_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list) if len(rmse_list) > 1 else 0
        method_ranking.append((method, avg_rmse, std_rmse))
    
    method_ranking.sort(key=lambda x: x[1])
    
    for i, (method, avg_rmse, std_rmse) in enumerate(method_ranking, 1):
        if std_rmse > 0:
            print(f"   {i}. {method:<15}: {avg_rmse:.4f} ± {std_rmse:.4f} RMSE")
        else:
            print(f"   {i}. {method:<15}: {avg_rmse:.4f} RMSE")
    
    print(f"\n📁 Visualizations saved:")
    print(f"   - multi_seed_lstm_comparison.png")
    print(f"   - Multi-seed forecasting analysis report")

# Generate final summary
create_final_summary(averaged_results, len(EXPERIMENT_CONFIG['seeds']))

print("\n" + "="*70)
print("🚀 All analyses completed successfully!")
print("� Check the generated PNG files for detailed visualizations")
print("="*70)
