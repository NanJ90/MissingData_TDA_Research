"""
Takens' Embedding Framework for Time Series Point Cloud Analysis

A reusable framework for applying Takens' embedding to time series datasets,
including support for imputed datasets and multiple imputation methods.

Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from gtda.time_series import SingleTakensEmbedding
    from gtda.plotting import plot_point_cloud
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False


class TakensEmbeddingFramework:
    """
    Framework for applying Takens' embedding to time series datasets.
    
    Supports:
    - Single or multiple datasets
    - Original and imputed datasets
    - Multiple imputation methods
    - Comparison analysis
    - Visualization
    """
    
    def __init__(
        self,
        dimension: int = 3,
        time_delay: int = 1,
        stride: int = 1,
        normalize: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Takens' Embedding Framework.
        
        Args:
            dimension: Embedding dimension (default: 3)
            time_delay: Time delay parameter τ (default: 1)
            stride: Stride for sampling (default: 1)
            normalize: Whether to normalize data (default: True)
            random_state: Random seed for reproducibility
        """
        self.dimension = dimension
        self.time_delay = time_delay
        self.stride = stride
        self.normalize = normalize
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.scaler = StandardScaler() if normalize else None
        self.datasets = {}
        self.point_clouds = {}
        self.processed_data = {}
        
    def apply_takens_embedding(
        self,
        time_series: np.ndarray,
        dimension: Optional[int] = None,
        time_delay: Optional[int] = None,
        stride: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply Takens' embedding to convert a 1D time series into a point cloud.
        
        For a time series x(t), the embedding creates points:
        [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
        
        Args:
            time_series: 1D array of time series values
            dimension: Embedding dimension (uses self.dimension if None)
            time_delay: Time delay parameter τ (uses self.time_delay if None)
            stride: Stride for sampling (uses self.stride if None)
        
        Returns:
            point_cloud: Array of shape (n_points, dimension)
        """
        dimension = dimension or self.dimension
        time_delay = time_delay or self.time_delay
        stride = stride or self.stride
        
        n = len(time_series)
        max_index = n - (dimension - 1) * time_delay
        
        if max_index <= 0:
            raise ValueError(
                f"Time series too short for dimension {dimension} and delay {time_delay}. "
                f"Need at least {(dimension - 1) * time_delay + 1} points, got {n}."
            )
        
        # Create point cloud using Takens' embedding
        point_cloud = []
        for i in range(0, max_index, stride):
            point = [time_series[i + j * time_delay] for j in range(dimension)]
            point_cloud.append(point)
        
        return np.array(point_cloud)
    
    def load_dataset(
        self,
        data: Union[pd.DataFrame, np.ndarray, str],
        name: str,
        target_column: Optional[str] = None,
        target_indices: Optional[np.ndarray] = None
    ):
        """
        Load a dataset into the framework.
        
        Args:
            data: Dataset as DataFrame, numpy array, or path to CSV/pickle file
            name: Name identifier for this dataset
            target_column: Name of target column (for classification/grouping)
            target_indices: Array of target indices (alternative to target_column)
        """
        # Load data if path provided
        if isinstance(data, str):
            if data.endswith('.pkl'):
                import pickle
                with open(data, 'rb') as f:
                    data = pickle.load(f)
            elif data.endswith('.csv'):
                data = pd.read_csv(data)
        
        # Convert to numpy if DataFrame
        if isinstance(data, pd.DataFrame):
            if target_column and target_column in data.columns:
                X = data.drop(columns=[target_column]).values
                y = data[target_column].values
            else:
                X = data.values
                y = target_indices
        else:
            X = np.array(data)
            y = target_indices
        
        # Store dataset
        self.datasets[name] = {
            'X': X,
            'y': y,
            'original_shape': X.shape
        }
        
        print(f"✓ Loaded dataset '{name}': {X.shape}")
    
    def load_imputed_datasets(
        self,
        imputed_data_path: str,
        original_data: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        original_name: str = 'original',
        target_column: Optional[str] = None
    ):
        """
        Load imputed datasets from a pickle file.
        
        Expected format: Dictionary with keys like:
        - 'knn_data_mcar', 'knn_data_mar', 'knn_data_mnar'
        - 'interpolated_data_mcar', etc.
        - 'gan_data_mcar', etc.
        
        Args:
            imputed_data_path: Path to pickle file containing imputed datasets
            original_data: Original complete dataset (optional)
            original_name: Name for original dataset
            target_column: Name of target column if present
        """
        import pickle
        
        print(f"Loading imputed datasets from: {imputed_data_path}")
        with open(imputed_data_path, 'rb') as f:
            imputed_datasets = pickle.load(f)
        
        # Load original if provided
        if original_data is not None:
            self.load_dataset(original_data, original_name, target_column)
        
        # Load all imputed datasets
        for name, data in imputed_datasets.items():
            if isinstance(data, pd.DataFrame):
                self.load_dataset(data, name, target_column)
            else:
                self.load_dataset(data, name)
        
        print(f"✓ Loaded {len(imputed_datasets)} imputed datasets")
        print(f"  Available datasets: {list(self.datasets.keys())}")
    
    def preprocess_dataset(self, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a dataset (normalize if enabled).
        
        Args:
            name: Dataset name
        
        Returns:
            X_processed: Preprocessed features
            y: Target vector (if available)
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")
        
        X = self.datasets[name]['X'].copy()
        y = self.datasets[name]['y']
        
        # Normalize if enabled
        if self.normalize:
            if name not in self.processed_data:
                # Fit scaler on this dataset
                X_processed = self.scaler.fit_transform(X)
                self.processed_data[name] = {'scaler': self.scaler}
            else:
                # Use existing scaler
                X_processed = self.processed_data[name]['scaler'].transform(X)
        else:
            X_processed = X
        
        return X_processed, y
    
    def create_point_clouds(
        self,
        dataset_name: str,
        channel_idx: Optional[int] = None,
        group_by_target: bool = False,
        max_samples_per_group: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create point clouds for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            channel_idx: Index of channel/feature to use (None = all channels)
            group_by_target: Whether to separate by target labels
            max_samples_per_group: Maximum samples per group (for efficiency)
        
        Returns:
            Dictionary of point clouds
        """
        X_processed, y = self.preprocess_dataset(dataset_name)
        
        point_clouds = {}
        
        # Determine channels to process
        if channel_idx is None:
            channels = range(X_processed.shape[1])
        else:
            channels = [channel_idx]
        
        # Group by target if requested
        if group_by_target and y is not None:
            unique_targets = np.unique(y)
            for target in unique_targets:
                target_indices = np.where(y == target)[0]
                
                # Sample if needed
                if max_samples_per_group and len(target_indices) > max_samples_per_group:
                    target_indices = np.random.choice(
                        target_indices, max_samples_per_group, replace=False
                    )
                
                for ch_idx in channels:
                    time_series = X_processed[target_indices, ch_idx]
                    pc = self.apply_takens_embedding(time_series)
                    
                    key = f"{dataset_name}_ch{ch_idx}_target{target}"
                    point_clouds[key] = pc
        else:
            # Process all data together
            for ch_idx in channels:
                time_series = X_processed[:, ch_idx]
                pc = self.apply_takens_embedding(time_series)
                
                key = f"{dataset_name}_ch{ch_idx}"
                point_clouds[key] = pc
        
        # Store point clouds
        if dataset_name not in self.point_clouds:
            self.point_clouds[dataset_name] = {}
        self.point_clouds[dataset_name].update(point_clouds)
        
        return point_clouds
    
    def compare_datasets(
        self,
        dataset_names: List[str],
        channel_idx: int = 0,
        group_by_target: bool = False,
        max_samples_per_group: Optional[int] = 500
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compare point clouds across multiple datasets.
        
        Args:
            dataset_names: List of dataset names to compare
            channel_idx: Channel index to use
            group_by_target: Whether to separate by target labels
            max_samples_per_group: Maximum samples per group
        
        Returns:
            Dictionary of point clouds organized by dataset
        """
        comparison_results = {}
        
        for name in dataset_names:
            if name not in self.datasets:
                print(f"⚠ Warning: Dataset '{name}' not found, skipping...")
                continue
            
            point_clouds = self.create_point_clouds(
                name, channel_idx, group_by_target, max_samples_per_group
            )
            comparison_results[name] = point_clouds
        
        return comparison_results
    
    def analyze_point_cloud(
        self,
        point_cloud: np.ndarray,
        name: str = "point_cloud"
    ) -> Dict[str, float]:
        """
        Analyze properties of a point cloud.
        
        Args:
            point_cloud: Point cloud array
            name: Name for reporting
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_points': len(point_cloud),
            'dimension': point_cloud.shape[1],
            'mean': point_cloud.mean(),
            'std': point_cloud.std(),
            'min': point_cloud.min(),
            'max': point_cloud.max(),
            'range': point_cloud.max() - point_cloud.min()
        }
        
        # Compute pairwise distances (sample for efficiency)
        sample_size = min(100, len(point_cloud))
        if sample_size > 1:
            sample_indices = np.random.choice(len(point_cloud), sample_size, replace=False)
            sample_pc = point_cloud[sample_indices]
            distances = pdist(sample_pc)
            stats['mean_pairwise_distance'] = distances.mean()
            stats['std_pairwise_distance'] = distances.std()
        
        return stats
    
    def visualize_point_cloud_3d(
        self,
        point_cloud: np.ndarray,
        title: str = "Point Cloud",
        color: str = 'blue',
        alpha: float = 0.6,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[Axes3D] = None
    ):
        """
        Visualize a 3D point cloud.
        
        Args:
            point_cloud: Point cloud array (must be 3D)
            title: Plot title
            color: Point color
            alpha: Transparency
            figsize: Figure size
            ax: Existing axes (if None, creates new figure)
        
        Returns:
            fig, ax: Figure and axes objects
        """
        if point_cloud.shape[1] != 3:
            raise ValueError(f"Point cloud must be 3D, got shape {point_cloud.shape}")
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        ax.scatter(
            point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
            c=color, alpha=alpha, s=20
        )
        
        ax.set_xlabel('X(t)')
        ax.set_ylabel('X(t+τ)')
        ax.set_zlabel('X(t+2τ)')
        ax.set_title(title)
        
        return fig, ax
    
    def visualize_comparison(
        self,
        point_clouds: Dict[str, np.ndarray],
        titles: Optional[Dict[str, str]] = None,
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Visualize multiple point clouds for comparison.
        
        Args:
            point_clouds: Dictionary of point clouds {name: point_cloud}
            titles: Dictionary of titles (uses keys if None)
            colors: Dictionary of colors (uses default if None)
            figsize: Figure size
        """
        n_clouds = len(point_clouds)
        if n_clouds == 0:
            print("No point clouds to visualize")
            return
        
        fig = plt.figure(figsize=(figsize[0] * n_clouds, figsize[1]))
        
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx, (name, pc) in enumerate(point_clouds.items()):
            ax = fig.add_subplot(1, n_clouds, idx + 1, projection='3d')
            
            title = titles.get(name, name) if titles else name
            color = colors.get(name, default_colors[idx % len(default_colors)]) if colors else default_colors[idx % len(default_colors)]
            
            self.visualize_point_cloud_3d(pc, title, color, ax=ax)
        
        plt.tight_layout()
        return fig
    
    def visualize_pca_projection(
        self,
        point_clouds: Dict[str, np.ndarray],
        n_components: int = 2,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Project point clouds to 2D using PCA for visualization.
        
        Args:
            point_clouds: Dictionary of point clouds
            n_components: Number of PCA components
            figsize: Figure size
        """
        # Combine all point clouds
        all_pcs = []
        labels = []
        for name, pc in point_clouds.items():
            all_pcs.append(pc)
            labels.extend([name] * len(pc))
        
        combined = np.vstack(all_pcs)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        combined_pca = pca.fit_transform(combined)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 2D projection
        ax = axes[0]
        colors_map = {'blue', 'red', 'green', 'orange', 'purple'}
        color_list = list(colors_map)
        
        start_idx = 0
        for idx, (name, pc) in enumerate(point_clouds.items()):
            end_idx = start_idx + len(pc)
            color = color_list[idx % len(color_list)]
            ax.scatter(
                combined_pca[start_idx:end_idx, 0],
                combined_pca[start_idx:end_idx, 1],
                c=color, alpha=0.5, s=10, label=name
            )
            start_idx = end_idx
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('PCA Projection Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Explained variance
        axes[1].bar(range(1, min(10, len(pca.explained_variance_ratio_)) + 1),
                   pca.explained_variance_ratio_[:10])
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Explained Variance Ratio')
        axes[1].set_title('PCA Explained Variance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compute_distance_matrix(
        self,
        point_clouds: Dict[str, np.ndarray],
        metric: str = 'euclidean'
    ) -> pd.DataFrame:
        """
        Compute pairwise distances between point clouds.
        
        Args:
            point_clouds: Dictionary of point clouds
            metric: Distance metric ('euclidean', 'hausdorff', etc.)
        
        Returns:
            Distance matrix as DataFrame
        """
        names = list(point_clouds.keys())
        n = len(names)
        distance_matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    pc1 = point_clouds[name1]
                    pc2 = point_clouds[name2]
                    
                    if metric == 'euclidean':
                        # Use centroid distance
                        centroid1 = pc1.mean(axis=0)
                        centroid2 = pc2.mean(axis=0)
                        distance_matrix[i, j] = np.linalg.norm(centroid1 - centroid2)
                    elif metric == 'hausdorff':
                        # Approximate Hausdorff distance
                        distances = cdist(pc1, pc2)
                        distance_matrix[i, j] = max(distances.min(axis=0).max(), 
                                                   distances.min(axis=1).max())
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
        
        return pd.DataFrame(distance_matrix, index=names, columns=names)
    
    def generate_report(
        self,
        dataset_names: Optional[List[str]] = None,
        channel_idx: int = 0,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a text report comparing datasets.
        
        Args:
            dataset_names: List of dataset names (uses all if None)
            channel_idx: Channel index to analyze
            output_file: Optional file path to save report
        
        Returns:
            Report as string
        """
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("TAKENS' EMBEDDING ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"\nConfiguration:")
        report_lines.append(f"  - Embedding dimension: {self.dimension}")
        report_lines.append(f"  - Time delay: {self.time_delay}")
        report_lines.append(f"  - Stride: {self.stride}")
        report_lines.append(f"  - Normalization: {self.normalize}")
        report_lines.append(f"\nAnalyzing channel: {channel_idx}")
        report_lines.append("\n" + "-" * 70)
        
        # Analyze each dataset
        all_stats = {}
        for name in dataset_names:
            if name not in self.datasets:
                continue
            
            # Create point clouds if not exists
            if name not in self.point_clouds:
                self.create_point_clouds(name, channel_idx)
            
            # Get point cloud
            key = f"{name}_ch{channel_idx}"
            if key in self.point_clouds.get(name, {}):
                pc = self.point_clouds[name][key]
                stats = self.analyze_point_cloud(pc, name)
                all_stats[name] = stats
                
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  Points: {stats['n_points']}")
                report_lines.append(f"  Mean: {stats['mean']:.4f}")
                report_lines.append(f"  Std: {stats['std']:.4f}")
                report_lines.append(f"  Range: {stats['range']:.4f}")
                if 'mean_pairwise_distance' in stats:
                    report_lines.append(f"  Mean pairwise distance: {stats['mean_pairwise_distance']:.4f}")
        
        # Distance matrix
        if len(all_stats) > 1:
            point_clouds_dict = {
                name: self.point_clouds[name][f"{name}_ch{channel_idx}"]
                for name in all_stats.keys()
            }
            distance_matrix = self.compute_distance_matrix(point_clouds_dict)
            report_lines.append("\n" + "-" * 70)
            report_lines.append("\nDistance Matrix (Centroid Distance):")
            report_lines.append(str(distance_matrix))
        
        report_lines.append("\n" + "=" * 70)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"✓ Report saved to: {output_file}")
        
        return report


def example_usage():
    """
    Example usage of the framework.
    """
    # Initialize framework
    framework = TakensEmbeddingFramework(
        dimension=3,
        time_delay=1,
        normalize=True,
        random_state=42
    )
    
    # Example 1: Load single dataset
    # framework.load_dataset('data/eeg_eye_state_full.csv', 'eeg_original', target_column='target')
    
    # Example 2: Load imputed datasets
    # framework.load_imputed_datasets(
    #     'imp_data/imputed_datasets_20250724_002055.pkl',
    #     original_data='data/eeg_eye_state_full.csv',
    #     original_name='original',
    #     target_column='target'
    # )
    
    # Example 3: Create point clouds
    # point_clouds = framework.create_point_clouds('original', channel_idx=0, group_by_target=True)
    
    # Example 4: Compare datasets
    # comparison = framework.compare_datasets(
    #     ['original', 'knn_data_mcar', 'gan_data_mcar'],
    #     channel_idx=0,
    #     group_by_target=True
    # )
    
    # Example 5: Visualize
    # framework.visualize_comparison(comparison['original'])
    
    # Example 6: Generate report
    # report = framework.generate_report(['original', 'knn_data_mcar'])
    
    print("Framework initialized. See example_usage() for usage patterns.")


if __name__ == "__main__":
    example_usage()



