"""
EEG Eye State Dataset - Point Cloud Representation using Takens' Embedding

This script demonstrates how to:
1. Load the EEG Eye State dataset
2. Apply Takens' embedding to convert time series to point clouds
3. Visualize the point cloud representations
4. Analyze the topological structure

Author: Data Scientist
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# For Takens' embedding
try:
    from gtda.time_series import SingleTakensEmbedding
    from gtda.plotting import plot_point_cloud
    GTDA_AVAILABLE = True
except ImportError:
    print("Warning: giotto-tda not available. Install with: pip install giotto-tda")
    GTDA_AVAILABLE = False

# For data fetching
try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    print("Warning: ucimlrepo not available. Install with: pip install ucimlrepo")
    UCI_AVAILABLE = False


# ============================================================================
# STEP 1: Load EEG Eye State Dataset
# ============================================================================
def load_eeg_data():
    """
    Load the EEG Eye State dataset from UCI repository.
    
    Returns:
        X: Features (EEG signals from 14 channels)
        y: Target (eye state: 0=closed, 1=open)
    """
    print("=" * 70)
    print("STEP 1: Loading EEG Eye State Dataset")
    print("=" * 70)
    
    if not UCI_AVAILABLE:
        # Try to load from local file if available
        try:
            data = pd.read_csv('data/eeg_eye_state_full.csv')
            X = data.drop('target', axis=1)
            y = data['target']
            print(f"✓ Loaded from local file: {X.shape[0]} samples, {X.shape[1]} features")
        except FileNotFoundError:
            raise ImportError("Please install ucimlrepo: pip install ucimlrepo")
    else:
        # Fetch from UCI repository
        print("Fetching dataset from UCI repository...")
        eeg_eye_state = fetch_ucirepo(id=264)
        X = eeg_eye_state.data.features
        y = eeg_eye_state.data.targets
        print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features (EEG channels): {X.shape[1]}")
    print(f"  - Target distribution: {y.value_counts().to_dict()}")
    print(f"    0 = Eye closed, 1 = Eye open\n")
    
    return X, y


# ============================================================================
# STEP 2: Preprocess the Data
# ============================================================================
def preprocess_data(X, y, normalize=True):
    """
    Preprocess the EEG data.
    
    Args:
        X: Feature matrix
        y: Target vector
        normalize: Whether to normalize the data
    
    Returns:
        X_processed: Preprocessed features
        y_processed: Target vector
    """
    print("=" * 70)
    print("STEP 2: Preprocessing Data")
    print("=" * 70)
    
    X_processed = X.copy()
    y_processed = y.copy()
    
    # Convert to numpy if needed
    if isinstance(X_processed, pd.DataFrame):
        X_processed = X_processed.values
    if isinstance(y_processed, pd.DataFrame):
        y_processed = y_processed.values.flatten()
    
    # Normalize the data
    if normalize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed)
        print("✓ Data normalized using StandardScaler")
    
    print(f"  - Data shape: {X_processed.shape}")
    print(f"  - Data range: [{X_processed.min():.2f}, {X_processed.max():.2f}]\n")
    
    return X_processed, y_processed


# ============================================================================
# STEP 3: Apply Takens' Embedding
# ============================================================================
def apply_takens_embedding(time_series, dimension=3, time_delay=1, stride=1):
    """
    Apply Takens' embedding to convert a 1D time series into a point cloud.
    
    Takens' embedding theorem states that we can reconstruct the attractor
    of a dynamical system by using delayed coordinates.
    
    For a time series x(t), the embedding creates points:
    [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
    where:
    - d = embedding dimension
    - τ = time delay
    
    Args:
        time_series: 1D array of time series values
        dimension: Embedding dimension (default: 3)
        time_delay: Time delay parameter τ (default: 1)
        stride: Stride for sampling (default: 1)
    
    Returns:
        point_cloud: Array of shape (n_points, dimension)
    """
    print("=" * 70)
    print("STEP 3: Applying Takens' Embedding")
    print("=" * 70)
    print(f"  - Embedding dimension (d): {dimension}")
    print(f"  - Time delay (τ): {time_delay}")
    print(f"  - Stride: {stride}")
    
    n = len(time_series)
    max_index = n - (dimension - 1) * time_delay
    
    if max_index <= 0:
        raise ValueError(f"Time series too short for dimension {dimension} and delay {time_delay}")
    
    # Create point cloud using Takens' embedding
    point_cloud = []
    for i in range(0, max_index, stride):
        point = [time_series[i + j * time_delay] for j in range(dimension)]
        point_cloud.append(point)
    
    point_cloud = np.array(point_cloud)
    
    print(f"✓ Point cloud created: {point_cloud.shape[0]} points in {dimension}D space")
    print(f"  - Point cloud shape: {point_cloud.shape}\n")
    
    return point_cloud


def apply_takens_embedding_gtda(time_series, dimension=3, time_delay=1):
    """
    Apply Takens' embedding using giotto-tda library (if available).
    
    Args:
        time_series: 1D array of time series values
        dimension: Embedding dimension
        time_delay: Time delay parameter
    
    Returns:
        point_cloud: Array of shape (n_points, dimension)
    """
    if not GTDA_AVAILABLE:
        return None
    
    # Reshape for giotto-tda (expects 2D array)
    time_series_2d = time_series.reshape(-1, 1)
    
    # Create Takens embedding transformer
    takens = SingleTakensEmbedding(
        parameters_type='search',
        dimension=dimension,
        time_delay=time_delay,
        n_jobs=1
    )
    
    # Fit and transform
    point_cloud = takens.fit_transform(time_series_2d)
    
    return point_cloud[0]  # Return first (and only) time series


# ============================================================================
# STEP 4: Create Point Clouds for Multiple EEG Channels
# ============================================================================
def create_multichannel_point_clouds(X, y, dimension=3, time_delay=1, 
                                     max_samples_per_class=500):
    """
    Create point clouds for multiple EEG channels, separated by eye state.
    
    Args:
        X: Feature matrix (n_samples, n_channels)
        y: Target vector (eye state)
        dimension: Embedding dimension
        time_delay: Time delay parameter
        max_samples_per_class: Maximum samples per class for visualization
    
    Returns:
        point_clouds_closed: List of point clouds for closed eyes
        point_clouds_open: List of point clouds for open eyes
    """
    print("=" * 70)
    print("STEP 4: Creating Point Clouds for Multiple EEG Channels")
    print("=" * 70)
    
    # Separate data by eye state
    closed_indices = np.where(y == 0)[0]
    open_indices = np.where(y == 1)[0]
    
    # Sample if needed
    if len(closed_indices) > max_samples_per_class:
        closed_indices = np.random.choice(closed_indices, max_samples_per_class, replace=False)
    if len(open_indices) > max_samples_per_class:
        open_indices = np.random.choice(open_indices, max_samples_per_class, replace=False)
    
    print(f"  - Closed eye samples: {len(closed_indices)}")
    print(f"  - Open eye samples: {len(open_indices)}")
    
    # Create point clouds for each channel
    n_channels = X.shape[1]
    point_clouds_closed = []
    point_clouds_open = []
    
    for channel_idx in range(n_channels):
        # Closed eyes
        closed_series = X[closed_indices, channel_idx]
        pc_closed = apply_takens_embedding(closed_series, dimension, time_delay, stride=1)
        point_clouds_closed.append(pc_closed)
        
        # Open eyes
        open_series = X[open_indices, channel_idx]
        pc_open = apply_takens_embedding(open_series, dimension, time_delay, stride=1)
        point_clouds_open.append(pc_open)
    
    print(f"✓ Created point clouds for {n_channels} channels")
    print(f"  - Each point cloud has dimension: {dimension}\n")
    
    return point_clouds_closed, point_clouds_open


# ============================================================================
# STEP 5: Visualize Point Clouds
# ============================================================================
def visualize_point_cloud_3d(point_cloud, title="Point Cloud", color='blue', 
                             alpha=0.6, figsize=(10, 8)):
    """
    Visualize a 3D point cloud.
    
    Args:
        point_cloud: Array of shape (n_points, 3)
        title: Plot title
        color: Point color
        alpha: Transparency
        figsize: Figure size
    """
    if point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must be 3D for this visualization")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               c=color, alpha=alpha, s=20)
    
    ax.set_xlabel('X (t)')
    ax.set_ylabel('X (t+τ)')
    ax.set_zlabel('X (t+2τ)')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def visualize_multiple_point_clouds(point_clouds_closed, point_clouds_open, 
                                    channel_names=None, n_channels_to_show=3):
    """
    Visualize point clouds for multiple channels side by side.
    
    Args:
        point_clouds_closed: List of point clouds for closed eyes
        point_clouds_open: List of point clouds for open eyes
        channel_names: Names of EEG channels
        n_channels_to_show: Number of channels to visualize
    """
    print("=" * 70)
    print("STEP 5: Visualizing Point Clouds")
    print("=" * 70)
    
    n_channels = min(len(point_clouds_closed), n_channels_to_show)
    
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(len(point_clouds_closed))]
    
    fig = plt.figure(figsize=(6 * n_channels, 10))
    
    for i in range(n_channels):
        # Closed eyes
        ax1 = fig.add_subplot(2, n_channels, i + 1, projection='3d')
        pc_closed = point_clouds_closed[i]
        ax1.scatter(pc_closed[:, 0], pc_closed[:, 1], pc_closed[:, 2],
                   c='blue', alpha=0.5, s=10, label='Closed')
        ax1.set_title(f'{channel_names[i]} - Eye Closed')
        ax1.set_xlabel('X(t)')
        ax1.set_ylabel('X(t+τ)')
        ax1.set_zlabel('X(t+2τ)')
        
        # Open eyes
        ax2 = fig.add_subplot(2, n_channels, i + n_channels + 1, projection='3d')
        pc_open = point_clouds_open[i]
        ax2.scatter(pc_open[:, 0], pc_open[:, 1], pc_open[:, 2],
                   c='red', alpha=0.5, s=10, label='Open')
        ax2.set_title(f'{channel_names[i]} - Eye Open')
        ax2.set_xlabel('X(t)')
        ax2.set_ylabel('X(t+τ)')
        ax2.set_zlabel('X(t+2τ)')
    
    plt.tight_layout()
    print(f"✓ Visualized {n_channels} channels\n")
    return fig


def visualize_point_cloud_comparison(point_cloud_closed, point_cloud_open, 
                                     channel_name="EEG Channel"):
    """
    Compare point clouds for closed vs open eyes in a single plot.
    
    Args:
        point_cloud_closed: Point cloud for closed eyes
        point_cloud_open: Point cloud for open eyes
        channel_name: Name of the channel
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Closed eyes
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(point_cloud_closed[:, 0], point_cloud_closed[:, 1], 
               point_cloud_closed[:, 2], c='blue', alpha=0.5, s=10)
    ax1.set_title(f'{channel_name} - Eye Closed')
    ax1.set_xlabel('X(t)')
    ax1.set_ylabel('X(t+τ)')
    ax1.set_zlabel('X(t+2τ)')
    
    # Open eyes
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(point_cloud_open[:, 0], point_cloud_open[:, 1], 
               point_cloud_open[:, 2], c='red', alpha=0.5, s=10)
    ax2.set_title(f'{channel_name} - Eye Open')
    ax2.set_xlabel('X(t)')
    ax2.set_ylabel('X(t+τ)')
    ax2.set_zlabel('X(t+2τ)')
    
    # Combined
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(point_cloud_closed[:, 0], point_cloud_closed[:, 1], 
               point_cloud_closed[:, 2], c='blue', alpha=0.3, s=10, label='Closed')
    ax3.scatter(point_cloud_open[:, 0], point_cloud_open[:, 1], 
               point_cloud_open[:, 2], c='red', alpha=0.3, s=10, label='Open')
    ax3.set_title(f'{channel_name} - Comparison')
    ax3.set_xlabel('X(t)')
    ax3.set_ylabel('X(t+τ)')
    ax3.set_zlabel('X(t+2τ)')
    ax3.legend()
    
    plt.tight_layout()
    return fig


# ============================================================================
# STEP 6: Analyze Point Cloud Properties
# ============================================================================
def analyze_point_cloud_properties(point_clouds_closed, point_clouds_open):
    """
    Analyze and compare properties of point clouds.
    
    Args:
        point_clouds_closed: List of point clouds for closed eyes
        point_clouds_open: List of point clouds for open eyes
    """
    print("=" * 70)
    print("STEP 6: Analyzing Point Cloud Properties")
    print("=" * 70)
    
    print("\nPoint Cloud Statistics:")
    print("-" * 70)
    print(f"{'Channel':<15} {'State':<10} {'N Points':<12} {'Mean':<12} {'Std':<12} {'Range':<12}")
    print("-" * 70)
    
    for i, (pc_closed, pc_open) in enumerate(zip(point_clouds_closed, point_clouds_open)):
        # Closed eyes
        print(f"{f'Channel {i+1}':<15} {'Closed':<10} {len(pc_closed):<12} "
              f"{pc_closed.mean():<12.4f} {pc_closed.std():<12.4f} "
              f"{pc_closed.max()-pc_closed.min():<12.4f}")
        
        # Open eyes
        print(f"{f'Channel {i+1}':<15} {'Open':<10} {len(pc_open):<12} "
              f"{pc_open.mean():<12.4f} {pc_open.std():<12.4f} "
              f"{pc_open.max()-pc_open.min():<12.4f}")
        print("-" * 70)
    
    # Compute pairwise distances (sample for efficiency)
    print("\nComputing distance statistics (sampled)...")
    sample_size = min(100, len(point_clouds_closed[0]))
    sample_indices = np.random.choice(len(point_clouds_closed[0]), sample_size, replace=False)
    
    pc_closed_sample = point_clouds_closed[0][sample_indices]
    pc_open_sample = point_clouds_open[0][sample_indices]
    
    from scipy.spatial.distance import pdist
    dist_closed = pdist(pc_closed_sample)
    dist_open = pdist(pc_open_sample)
    
    print(f"  - Mean pairwise distance (closed): {dist_closed.mean():.4f}")
    print(f"  - Mean pairwise distance (open): {dist_open.mean():.4f}")
    print(f"  - Std pairwise distance (closed): {dist_closed.std():.4f}")
    print(f"  - Std pairwise distance (open): {dist_open.std():.4f}\n")


# ============================================================================
# STEP 7: PCA Visualization (for higher dimensional embeddings)
# ============================================================================
def visualize_pca_projection(point_clouds_closed, point_clouds_open, 
                             n_components=2, channel_idx=0):
    """
    Project point clouds to 2D using PCA for visualization.
    
    Args:
        point_clouds_closed: List of point clouds for closed eyes
        point_clouds_open: List of point clouds for open eyes
        n_components: Number of PCA components
        channel_idx: Index of channel to visualize
    """
    print("=" * 70)
    print("STEP 7: PCA Projection Visualization")
    print("=" * 70)
    
    # Combine point clouds
    pc_closed = point_clouds_closed[channel_idx]
    pc_open = point_clouds_open[channel_idx]
    
    # Apply PCA
    combined = np.vstack([pc_closed, pc_open])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined)
    
    # Split back
    n_closed = len(pc_closed)
    closed_pca = combined_pca[:n_closed]
    open_pca = combined_pca[n_closed:]
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2D projection
    axes[0].scatter(closed_pca[:, 0], closed_pca[:, 1], c='blue', alpha=0.5, s=10, label='Closed')
    axes[0].scatter(open_pca[:, 0], open_pca[:, 1], c='red', alpha=0.5, s=10, label='Open')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title(f'PCA Projection - Channel {channel_idx + 1}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Explained variance
    if len(pca.explained_variance_ratio_) > 2:
        axes[1].bar(range(1, min(10, len(pca.explained_variance_ratio_)) + 1),
                   pca.explained_variance_ratio_[:10])
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Explained Variance Ratio')
        axes[1].set_title('PCA Explained Variance')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"✓ PCA visualization created for channel {channel_idx + 1}\n")
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function to run the complete pipeline.
    """
    print("\n" + "=" * 70)
    print("EEG EYE STATE DATASET - TAKENS' EMBEDDING ANALYSIS")
    print("=" * 70 + "\n")
    
    # Step 1: Load data
    X, y = load_eeg_data()
    
    # Step 2: Preprocess
    X_processed, y_processed = preprocess_data(X, y, normalize=True)
    
    # Step 3: Apply Takens' embedding to a single channel as example
    print("=" * 70)
    print("STEP 3: Example - Takens' Embedding for Single Channel")
    print("=" * 70)
    channel_idx = 0  # First EEG channel
    time_series = X_processed[:, channel_idx]
    
    # Manual implementation
    dimension = 3
    time_delay = 1
    point_cloud_example = apply_takens_embedding(time_series, dimension, time_delay, stride=5)
    
    # Try giotto-tda if available
    if GTDA_AVAILABLE:
        try:
            point_cloud_gtda = apply_takens_embedding_gtda(time_series, dimension, time_delay)
            if point_cloud_gtda is not None:
                print(f"✓ Giotto-tda embedding: {point_cloud_gtda.shape}")
        except Exception as e:
            print(f"  Note: Giotto-tda embedding failed: {e}")
    
    # Step 4: Create point clouds for multiple channels
    point_clouds_closed, point_clouds_open = create_multichannel_point_clouds(
        X_processed, y_processed, dimension=dimension, time_delay=time_delay,
        max_samples_per_class=500
    )
    
    # Step 5: Visualize
    print("Creating visualizations...")
    fig1 = visualize_multiple_point_clouds(
        point_clouds_closed, point_clouds_open, 
        n_channels_to_show=3
    )
    plt.savefig('eeg_point_clouds_multichannel.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: eeg_point_clouds_multichannel.png")
    
    # Comparison plot for first channel
    fig2 = visualize_point_cloud_comparison(
        point_clouds_closed[0], point_clouds_open[0],
        channel_name="Channel 1"
    )
    plt.savefig('eeg_point_cloud_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: eeg_point_cloud_comparison.png")
    
    # Step 6: Analyze properties
    analyze_point_cloud_properties(point_clouds_closed, point_clouds_open)
    
    # Step 7: PCA visualization
    fig3 = visualize_pca_projection(point_clouds_closed, point_clouds_open, channel_idx=0)
    plt.savefig('eeg_point_cloud_pca.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: eeg_point_cloud_pca.png")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Processed {X_processed.shape[0]} samples")
    print(f"  - Created point clouds for {len(point_clouds_closed)} channels")
    print(f"  - Embedding dimension: {dimension}")
    print(f"  - Time delay: {time_delay}")
    print("\nVisualizations saved:")
    print("  - eeg_point_clouds_multichannel.png")
    print("  - eeg_point_cloud_comparison.png")
    print("  - eeg_point_cloud_pca.png")
    print("\n")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()



