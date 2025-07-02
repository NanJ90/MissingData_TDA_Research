#%%
import pandas as pd
import numpy as np
import xgboost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report
import wandb
import os
import time
from datetime import datetime

# Initialize wandb
wandb.init(project="eeg-eye-state-imputation", name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
# %%
# Generate some example binary classification data

def classification_data_nocv(imp_name,df, root_path):
    """Generate synthetic binary classification data."""

    X = df.drop(columns=['target'])
    y = df['target']
    

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the XGBoost classifier
    model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss',random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    acc = classification_report(y_test, y_pred)
    # print(f"Test Accuracy: {acc:.4f}")
    return acc
#%%
def classification_data(imp_name, df, root_path):
    """Evaluate model performance using cross-validation."""
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Define the cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create the XGBoost classifier
    model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
        'f1_weighted': 'f1_weighted'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    # Calculate mean and std of scores
    results = {
        'Accuracy': f"{cv_results['test_accuracy'].mean():.4f} (±{cv_results['test_accuracy'].std():.4f})",
        'Precision': f"{cv_results['test_precision_weighted'].mean():.4f} (±{cv_results['test_precision_weighted'].std():.4f})",
        'Recall': f"{cv_results['test_recall_weighted'].mean():.4f} (±{cv_results['test_recall_weighted'].std():.4f})",
        'F1': f"{cv_results['test_f1_weighted'].mean():.4f} (±{cv_results['test_f1_weighted'].std():.4f})"
    }
    
    # Log to wandb
    wandb.log({
        f"{imp_name}_accuracy_mean": cv_results['test_accuracy'].mean(),
        f"{imp_name}_accuracy_std": cv_results['test_accuracy'].std(),
        f"{imp_name}_precision_mean": cv_results['test_precision_weighted'].mean(),
        f"{imp_name}_precision_std": cv_results['test_precision_weighted'].std(),
        f"{imp_name}_recall_mean": cv_results['test_recall_weighted'].mean(),
        f"{imp_name}_recall_std": cv_results['test_recall_weighted'].std(),
        f"{imp_name}_f1_mean": cv_results['test_f1_weighted'].mean(),
        f"{imp_name}_f1_std": cv_results['test_f1_weighted'].std(),
        f"{imp_name}_data_shape": f"{df.shape[0]}x{df.shape[1]}"
    })
    
    return results
#%%
#import datasets
import os
import sys
root_dir = 'imp_data'
root_dir_orig ='data'

# Create timestamped results directory to avoid overwriting
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f"classification_results_{timestamp}"

for file in os.listdir(root_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(root_dir, file)
        imputation = file.split('_')[0]
        missing_mech = file.split('_')[-1].replace('.csv', '')  # Extract missing mechanism from filename
        print(f"Imputation method: {imputation}")
        df = pd.read_csv(file_path)
        # print(f"Processing {file} with shape {df.shape}")
        # Call the classification_data function
        results = classification_data(f'{imputation}_{missing_mech}', df, results_dir)
        # print(acc)
        for metric, score in results.items():
            print(f"{metric}: {score}")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            with open(os.path.join(results_dir, f"{imputation}_{missing_mech}_results.txt"), 'a') as f:
                f.write(f"{metric}: {score}\n")
            sys.stdout.flush()  # Ensure output is printed immediately
#%%
# original data
df = pd.read_csv(os.path.join(root_dir_orig, 'eeg_eye_state_full.csv'))

# acc= classification_data('original', df, root_dir_orig)
results = classification_data('original', df, root_dir_orig)
# print(acc)
for metric, score in results.items():
    print(f"{metric}: {score}")

# Create a results table for wandb
results_table = wandb.Table(columns=["Imputation", "Missing_Mechanism", "Accuracy", "Precision", "Recall", "F1"])

# Add original data results
results_table.add_data("original", "none", 
                      results['Accuracy'], results['Precision'], 
                      results['Recall'], results['F1'])
# X = df.drop(columns=['target'])
# y = df['target']
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
#                            n_redundant=5, n_classes=2, random_state=42)
# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
       
# %% analysis importing classification results as one df

df = pd.DataFrame()
results_dir_to_read = results_dir if os.path.exists(results_dir) else "classification_results"

for file in os.listdir(results_dir_to_read):
    if file.endswith('.txt'):
        file_path = os.path.join(results_dir_to_read, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Extract the imputation method and missing mechanism from the filename
            imputation_method = file.split('_')[0]
            missing_mechanism = file.split('.')[0].split('_')[1] 
            # Create a dictionary to hold the results
            result_dict = {'Imputation': imputation_method, 'Missing Mechanism': missing_mechanism.upper()}
            for line in lines:
                metric, score = line.strip().split(': ')
                result_dict[metric] = score
            # Convert the dictionary to a DataFrame and concatenate
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
            
            # Add to wandb table
            if len(result_dict) > 2:  # Ensure we have metrics beyond just method names
                results_table.add_data(imputation_method, missing_mechanism.upper(),
                                     result_dict.get('Accuracy', 'N/A'),
                                     result_dict.get('Precision', 'N/A'),
                                     result_dict.get('Recall', 'N/A'),
                                     result_dict.get('F1', 'N/A'))

#%%
# group by misisng mechanism and imputation method

# ...existing code...

# Group and sort without calculating mean
df_grouped = (df.groupby(['Missing Mechanism', 'Imputation'])
              .agg(lambda x: x.iloc[0])  # Keep the first occurrence of each group
              .reset_index()
              .sort_values(by=['Missing Mechanism', 'Imputation']))

# Now df_grouped will contain one row per combination of Missing Mechanism and Imputation
# with their original metric values (including the ± notation)
# df_grouped
#export df_grouped to csv with timestamp
summary_file = f'{results_dir}/summary_results.csv'
df_grouped.to_csv(summary_file, index=False)

# Log the results table to wandb
wandb.log({"results_summary": results_table})

# Log configuration
wandb.config.update({
    "cv_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "model": "XGBClassifier",
    "missing_rate": 0.3,
    "results_file": summary_file
})

if __name__ == "__main__":
    print(f"Classification results saved to '{summary_file}'")
    print("Experiment logged to Weights & Biases")
    wandb.finish()
