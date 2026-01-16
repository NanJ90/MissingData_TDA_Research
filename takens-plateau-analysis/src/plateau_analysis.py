from scipy.spatial.distance import pdist
import numpy as np


def compute_mean_pairwise_distances(arg1, arg2=None, arg3=None, arg4=None, R_th=None):
    """Compute mean pairwise distances.

    Supports two call signatures for backward compatibility:
      1) compute_mean_pairwise_distances(point_clouds, max_dim, time_delay)
         - `point_clouds` is a dict with keys 'closed' and 'open' containing series.
         - Returns: (valid_dims, mean_distances_array) where mean_distances_array has shape (n_dims, 2)

      2) compute_mean_pairwise_distances(closed_series, open_series, time_delay, max_dim, R_th=None)
         - Returns: (mean_dist_closed_array, mean_dist_open_array, valid_dims)
    """

    # Case A: old signature: (point_clouds, max_dim, time_delay)
    if isinstance(arg1, dict) and arg2 is not None and arg3 is not None and arg4 is None:
        point_clouds = arg1
        max_dim = arg2
        time_delay = arg3

        mean_distances = []
        valid_dims = []

        for d in range(1, max_dim + 1):
            try:
                pc_closed = apply_takens_embedding(point_clouds['closed'], d, time_delay)
                pc_open = apply_takens_embedding(point_clouds['open'], d, time_delay)

                if len(pc_closed) < 2 or len(pc_open) < 2:
                    continue

                mdc = pdist(pc_closed).mean()
                mdo = pdist(pc_open).mean()

                mean_distances.append((mdc, mdo))
                valid_dims.append(d)

            except Exception:
                continue

        return valid_dims, np.array(mean_distances)

    # Case B: new signature: (closed_series, open_series, time_delay, max_dim, R_th=None)
    if arg2 is not None and arg3 is not None and arg4 is not None:
        closed_series = arg1
        open_series = arg2
        time_delay = arg3
        max_dim = arg4

        mean_dist_closed = []
        mean_dist_open = []
        valid_dims = []

        for d in range(1, max_dim + 1):
            try:
                pc_closed = apply_takens_embedding(closed_series, d, time_delay)
                pc_open = apply_takens_embedding(open_series, d, time_delay)

                if len(pc_closed) < 2 or len(pc_open) < 2:
                    continue

                mdc = pdist(pc_closed).mean()
                mdo = pdist(pc_open).mean()

                mean_dist_closed.append(mdc)
                mean_dist_open.append(mdo)
                valid_dims.append(d)

            except Exception:
                continue

        return np.array(mean_dist_closed), np.array(mean_dist_open), valid_dims

    # If we reach here, the arguments didn't match expected patterns
    raise TypeError('compute_mean_pairwise_distances: unsupported call signature')


def detect_plateau(mean_distances, valid_dims, plateau_threshold=0.05, plateau_consecutive=2):
    rel_changes = np.abs(np.diff(mean_distances[:, 0]) / (mean_distances[:-1, 0] + 1e-12))
    candidate_dim = None

    if len(rel_changes) >= plateau_consecutive:
        for i in range(len(rel_changes) - (plateau_consecutive - 1)):
            window = rel_changes[i:i + plateau_consecutive]
            if np.all(window < plateau_threshold):
                candidate_dim = int(valid_dims[i + 1])
                break

    return candidate_dim


def analyze_plateau_for_multiple_Rth(point_clouds, R_th_values, max_dim_limit, time_delay):
    results = {}

    for R_th in R_th_values:
        valid_dims, mean_distances = compute_mean_pairwise_distances(point_clouds, max_dim_limit, time_delay)
        candidate_dim = detect_plateau(mean_distances, valid_dims)

        results[R_th] = {
            'valid_dims': valid_dims,
            'mean_distances': mean_distances,
            'candidate_dim': candidate_dim
        }

    return results


def apply_takens_embedding(time_series, dimension=3, time_delay=1):
    n = len(time_series)
    max_index = n - (dimension - 1) * time_delay

    if max_index <= 0:
        raise ValueError(f"Time series too short for dimension {dimension} and delay {time_delay}")

    point_cloud = []
    for i in range(max_index):
        point = [time_series[i + j * time_delay] for j in range(dimension)]
        point_cloud.append(point)

    return np.array(point_cloud)