"""
Simple Example: Using Takens' Embedding Framework

Quick example showing how to use the framework with imputed datasets.
"""

from takens_framework import TakensEmbeddingFramework
import os
import glob

def main():
    """Example usage of the framework."""
    
    # Step 1: Initialize framework
    print("Initializing framework...")
    framework = TakensEmbeddingFramework(
        dimension=3,
        time_delay=1,
        normalize=True,
        random_state=42
    )
    
    # Step 2: Load datasets
    print("\nLoading datasets...")
    
    # Find imputed datasets
    imp_data_dir = 'imp_data'
    pickle_files = glob.glob(f'{imp_data_dir}/imputed_datasets_*.pkl')
    
    if pickle_files and os.path.exists('data/eeg_eye_state_full.csv'):
        imputed_data_path = sorted(pickle_files)[-1]
        original_data_path = 'data/eeg_eye_state_full.csv'
        
        # Load all datasets at once
        framework.load_imputed_datasets(
            imputed_data_path=imputed_data_path,
            original_data=original_data_path,
            original_name='original',
            target_column='target'
        )
        
        print(f"✓ Loaded {len(framework.datasets)} datasets")
        
        # Step 3: Compare datasets
        print("\nCreating point clouds...")
        
        # Compare original with a few imputed methods
        datasets_to_compare = ['original']
        
        # Add one example from each imputation method
        for name in framework.datasets.keys():
            if 'knn' in name and 'mcar' in name:
                datasets_to_compare.append(name)
            elif 'gan' in name and 'mcar' in name:
                datasets_to_compare.append(name)
            elif len(datasets_to_compare) >= 3:
                break
        
        comparison = framework.compare_datasets(
            dataset_names=datasets_to_compare,
            channel_idx=0,
            group_by_target=True,
            max_samples_per_group=500
        )
        
        # Step 4: Visualize
        print("\nGenerating visualizations...")
        
        # Visualize comparison
        target_key = '_ch0_target0'  # Closed eyes
        comparison_pcs = {}
        for name, pcs in comparison.items():
            key = f"{name}{target_key}"
            if key in pcs:
                display_name = name.replace('_data_', '_').title()
                comparison_pcs[display_name] = pcs[key]
        
        if len(comparison_pcs) > 1:
            framework.visualize_comparison(comparison_pcs)
            import matplotlib.pyplot as plt
            plt.suptitle('Original vs Imputed Datasets', y=1.02)
            plt.savefig('takens_comparison.png', dpi=150, bbox_inches='tight')
            print("✓ Saved: takens_comparison.png")
        
        # Step 5: Generate report
        print("\nGenerating report...")
        report = framework.generate_report(
            dataset_names=datasets_to_compare,
            channel_idx=0,
            output_file='takens_report.txt'
        )
        print("✓ Report generated")
        
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)
        
    else:
        print("⚠ Datasets not found. Please ensure:")
        print("  1. Run imputation.py to generate imputed datasets")
        print("  2. Original dataset exists at: data/eeg_eye_state_full.csv")


if __name__ == "__main__":
    main()



