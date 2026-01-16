def apply_takens_embedding(time_series, dimension=3, time_delay=1, stride=1):
    """
    Apply Takens' embedding to convert a 1D time series into a point cloud.

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
    n = len(time_series)
    max_index = n - (dimension - 1) * time_delay

    if max_index <= 0:
        raise ValueError(f"Time series too short for dimension {dimension} and delay {time_delay}")

    # Create point cloud using Takens' embedding
    point_cloud = []
    for i in range(0, max_index, stride):
        point = [time_series[i + j * time_delay] for j in range(dimension)]
        point_cloud.append(point)

    return np.array(point_cloud)

def compute_mean_pairwise_distances(point_cloud):
    """
    Compute the mean pairwise distances for a given point cloud.

    Args:
        point_cloud: Array of shape (n_points, dimension)

    Returns:
        mean_distance: Mean of pairwise distances
    """
    return pdist(point_cloud).mean()

def analyze_plateau_for_multiple_R_ths(series, dimensions, time_delay, R_th_values):
    """
    Analyze the mean pairwise distances for multiple R_th values and determine common dimensionality.

    Args:
        series: Time series data
        dimensions: List of embedding dimensions to test
        time_delay: Time delay parameter τ
        R_th_values: List of R_th values to analyze

    Returns:
        common_dim: Dimensionality at which distances plateau for all R_th values
    """
    mean_distances = {R_th: [] for R_th in R_th_values}
    
    for R_th in R_th_values:
        for d in dimensions:
            try:
                point_cloud = apply_takens_embedding(series, dimension=d, time_delay=time_delay)
                mean_distance = compute_mean_pairwise_distances(point_cloud)
                mean_distances[R_th].append(mean_distance)
            except Exception as e:
                mean_distances[R_th].append(None)  # Handle errors gracefully

    # Determine common dimensionality where all mean distances plateau
    common_dim = None
    for i in range(1, len(dimensions)):
        if all(mean_distances[R_th][i] is not None and mean_distances[R_th][i] == mean_distances[R_th][i - 1] for R_th in R_th_values):
            common_dim = dimensions[i]
            break

    return common_dim, mean_distances

# Example usage
# series = ...  # Load your time series data here
# dimensions = list(range(1, 13))  # Test dimensions from 1 to 12
# time_delay = 1
# R_th_values = [0.1, 0.2, 0.3]  # Example R_th values
# common_dim, mean_distances = analyze_plateau_for_multiple_R_ths(series, dimensions, time_delay, R_th_values)
# print(f"Common dimensionality at which distances plateau: {common_dim}")