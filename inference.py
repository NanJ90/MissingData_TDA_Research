#%%
import pandas as pd
import numpy as np
import xgboost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report
# %%
# Generate some example binary classification data

def classification_data_nocv(imp_name,df, root_path):
    """Generate synthetic binary classification data."""

    X = df.drop(columns=['target'])
    y = df['target']
    

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the XGBoost classifier
    model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
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
    
    return results
#%%
#import datasets
import os
import sys
root_dir = 'imp_data'
root_dir_orig ='data'

for file in os.listdir(root_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(root_dir, file)
        imputation = file.split('_')[0]
        missing_mech = file.split('_')[-1]  # Extract missing mechanism from filename
        print(f"Imputation method: {imputation}")
        df = pd.read_csv(file_path)
        # print(f"Processing {file} with shape {df.shape}")
        # Call the classification_data function
        results = classification_data('original', df, 'classification_results')
# print(acc)
        for metric, score in results.items():
            print(f"{metric}: {score}")
            if not os.path.exists("classification_results"):
                os.makedirs("classification_results")
            with open(os.path.join("classification_results", f"{imputation}_{missing_mech}_results.txt"), 'a') as f:
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
# X = df.drop(columns=['target'])
# y = df['target']
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
#                            n_redundant=5, n_classes=2, random_state=42)
# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
       
# %% analysis importing classification results as one df

df = pd.DataFrame()
for file in os.listdir("classification_results"):
    if file.endswith('.txt'):
        file_path = os.path.join("classification_results", file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Extract the imputation method and missing mechanism from the filename
            imputation_method = file.split('_')[0]
            missing_mechanism = file.split('.')[0].split('_')[-1] 
            # Create a dictionary to hold the results
            result_dict = {'Imputation': imputation_method, 'Missing Mechanism': missing_mechanism.upper()}
            for line in lines:
                metric, score = line.strip().split(': ')
                result_dict[metric] = score
            # Convert the dictionary to a DataFrame and concatenate
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)

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
#export df_grouped to csv
df_grouped.to_csv('classification_results/summary_results.csv', index=False)

if __name__ == "__main__":
    print("Classification results saved to 'classification_results/summary_results.csv'")
