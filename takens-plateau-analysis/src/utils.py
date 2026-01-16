def compute_plateau_analysis(point_clouds_closed, point_clouds_open, max_dim_limit=12, plateau_threshold=0.05, plateau_consecutive=2):
    """
    Compute mean pairwise distances for different embedding dimensions and determine the common dimensionality
    at which the distances plateau for robustness in the analysis of the attractor's geometry.

    Args:
        point_clouds_closed: List of point clouds for closed eye states.
        point_clouds_open: List of point clouds for open eye states.
        max_dim_limit: Maximum dimension to try.
        plateau_threshold: Relative change threshold for plateau detection.
        plateau_consecutive: Number of consecutive small changes required to detect a plateau.

    Returns:
        candidate_dim: The common dimensionality at which the distances plateau.
        mean_dist_closed: List of mean pairwise distances for closed eye states.
        mean_dist_open: List of mean pairwise distances for open eye states.
    """
    from scipy.spatial.distance import pdist
    import numpy as np

    valid_dims = []
    mean_dist_closed = []
    mean_dist_open = []

    for d in range(1, max_dim_limit + 1):
        try:
            pc_c = point_clouds_closed[d - 1]
            pc_o = point_clouds_open[d - 1]
            if len(pc_c) < 2 or len(pc_o) < 2:
                continue
            mdc = pdist(pc_c).mean()
            mdo = pdist(pc_o).mean()
            mean_dist_closed.append(mdc)
            mean_dist_open.append(mdo)
            valid_dims.append(d)
        except Exception:
            continue

    # Detect plateau for closed eye states
    rel_changes = np.abs(np.diff(mean_dist_closed) / (mean_dist_closed[:-1] + 1e-12))
    candidate_dim = None
    if len(rel_changes) >= plateau_consecutive:
        for i in range(len(rel_changes) - (plateau_consecutive - 1)):
            window = rel_changes[i:i + plateau_consecutive]
            if np.all(window < plateau_threshold):
                candidate_dim = int(valid_dims[i + 1])  # plateau detected at the latter dimension in the window
                break

    return candidate_dim, mean_dist_closed, mean_dist_open