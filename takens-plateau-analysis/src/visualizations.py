def plot_mean_pairwise_distances(mean_distances, valid_dims, r_th_values):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    
    for i, r_th in enumerate(r_th_values):
        plt.plot(valid_dims, mean_distances[i], marker='o', label=f'R_th = {r_th}')
    
    plt.xlabel('Embedding Dimension (d)')
    plt.ylabel('Mean Pairwise Distance')
    plt.title('Mean Pairwise Distance vs Embedding Dimension for Different R_th Values')
    plt.axhline(y=mean_distances.mean(), color='gray', linestyle='--', label='Overall Mean Distance')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_plateau(mean_distances, valid_dims, plateau_threshold=0.05, plateau_consecutive=2):
    candidate_dims = []
    
    for distances in mean_distances:
        rel_changes = np.abs(np.diff(distances) / (distances[:-1] + 1e-12))
        if len(rel_changes) >= plateau_consecutive:
            for i in range(len(rel_changes) - (plateau_consecutive - 1)):
                window = rel_changes[i:i + plateau_consecutive]
                if np.all(window < plateau_threshold:
                    candidate_dims.append(valid_dims[i + 1])
                    break
    
    return candidate_dims

def visualize_plateau_analysis(mean_distances, valid_dims, r_th_values):
    candidate_dims = analyze_plateau(mean_distances, valid_dims)
    plot_mean_pairwise_distances(mean_distances, valid_dims, r_th_values)
    
    if candidate_dims:
        print(f"Common dimensionalities at which distances plateau: {set(candidate_dims)}")
    else:
        print("No common dimensionality found where distances plateau.")