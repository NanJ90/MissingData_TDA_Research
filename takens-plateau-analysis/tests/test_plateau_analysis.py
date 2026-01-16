import numpy as np
import matplotlib.pyplot as plt
from src.plateau_analysis import compute_mean_pairwise_distances, detect_plateau

def test_plateau_analysis():
    # Sample parameters for testing
    channel_idx = 0
    time_delay = 1
    max_dim_limit = 12
    sample_size = 500
    R_th_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different threshold values to test

    # Prepare series (use available samples)
    closed_series = X_processed[closed_indices[:sample_size], channel_idx]
    open_series = X_processed[open_indices[:sample_size], channel_idx]

    # Store results for each R_th
    results = {}

    for R_th in R_th_values:
        mean_dist_closed, mean_dist_open, valid_dims = compute_mean_pairwise_distances(closed_series, open_series, time_delay, max_dim_limit, R_th)
        plateau_dim = detect_plateau(mean_dist_closed, valid_dims)

        results[R_th] = {
            'mean_dist_closed': mean_dist_closed,
            'mean_dist_open': mean_dist_open,
            'valid_dims': valid_dims,
            'plateau_dim': plateau_dim
        }

        # Plotting the results
        plt.figure(figsize=(8, 5))
        plt.plot(valid_dims, mean_dist_closed, marker='o', label='Closed (mean pairwise dist)')
        plt.plot(valid_dims, mean_dist_open, marker='s', label='Open (mean pairwise dist)')
        if plateau_dim is not None:
            plt.axvline(plateau_dim, color='gray', linestyle='--', alpha=0.7)
            plt.text(plateau_dim + 0.2, max(mean_dist_closed.max(), mean_dist_open.max()) * 0.95,
                     f'Candidate d≈{plateau_dim}', rotation=90, va='top', color='gray')
        plt.xlabel('Embedding dimension (d)')
        plt.ylabel('Mean pairwise distance')
        plt.title(f'Mean pairwise distance vs embedding dimension (R_th={R_th})')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Analyze common plateau dimension across different R_th values
    common_plateau_dims = [result['plateau_dim'] for result in results.values() if result['plateau_dim'] is not None]
    if common_plateau_dims:
        common_plateau_dim = np.mode(common_plateau_dims)
        print(f"Common dimensionality at which all R_th values plateau: {common_plateau_dim}")
    else:
        print("No common plateau dimension found across the tested R_th values.")

test_plateau_analysis()