"""
Simple Example: Using Takens' Embedding for EEG Eye State Dataset

This is a simplified example showing the basic steps.
For the full implementation, see eeg_takens_embedding.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import functions from the main script
from eeg_takens_embedding import (
    load_eeg_data,
    preprocess_data,
    apply_takens_embedding,
    visualize_point_cloud_3d,
    visualize_point_cloud_comparison
)


def simple_example():
    """
    Simple step-by-step example of Takens' embedding.
    """
    print("=" * 70)
    print("SIMPLE EXAMPLE: Takens' Embedding for EEG Data")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading EEG Eye State dataset...")
    X, y = load_eeg_data()
    
    # Step 2: Preprocess
    print("\n[Step 2] Preprocessing data...")
    X_processed, y_processed = preprocess_data(X, y, normalize=True)
    
    # Step 3: Select a single channel and create point cloud
    print("\n[Step 3] Applying Takens' embedding...")
    channel_idx = 0  # First EEG channel
    
    # Separate by eye state
    closed_indices = np.where(y_processed == 0)[0][:200]  # Sample 200 points
    open_indices = np.where(y_processed == 1)[0][:200]
    
    # Create time series for closed and open eyes
    closed_series = X_processed[closed_indices, channel_idx]
    open_series = X_processed[open_indices, channel_idx]
    
    # Apply Takens' embedding
    # Parameters: dimension=3, time_delay=1
    dimension = 3
    time_delay = 1
    
    point_cloud_closed = apply_takens_embedding(closed_series, dimension, time_delay)
    point_cloud_open = apply_takens_embedding(open_series, dimension, time_delay)
    
    print(f"  ✓ Created point cloud for closed eyes: {point_cloud_closed.shape}")
    print(f"  ✓ Created point cloud for open eyes: {point_cloud_open.shape}")
    
    # Step 4: Visualize
    print("\n[Step 4] Creating visualizations...")
    
    # Individual plots
    fig1, ax1 = visualize_point_cloud_3d(
        point_cloud_closed, 
        title="EEG Channel 1 - Eye Closed (Takens' Embedding)",
        color='blue'
    )
    plt.savefig('example_closed.png', dpi=150, bbox_inches='tight')
    
    fig2, ax2 = visualize_point_cloud_3d(
        point_cloud_open,
        title="EEG Channel 1 - Eye Open (Takens' Embedding)",
        color='red'
    )
    plt.savefig('example_open.png', dpi=150, bbox_inches='tight')
    
    # Comparison plot
    fig3 = visualize_point_cloud_comparison(
        point_cloud_closed,
        point_cloud_open,
        channel_name="EEG Channel 1"
    )
    plt.savefig('example_comparison.png', dpi=150, bbox_inches='tight')
    
    print("  ✓ Saved visualizations:")
    print("    - example_closed.png")
    print("    - example_open.png")
    print("    - example_comparison.png")
    
    # Step 5: Explain what we see
    print("\n[Step 5] Interpretation:")
    print("  - Each point in the 3D space represents a state of the EEG signal")
    print("  - The coordinates are: [x(t), x(t+τ), x(t+2τ)]")
    print("  - The shape of the point cloud reveals the underlying dynamics")
    print("  - Differences between closed/open states can be seen in the cloud structure")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    simple_example()



