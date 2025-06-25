# %%
from ucimlrepo import fetch_ucirepo

def fetch_eeg_eye_state_data():
    """
    Fetches the EEG Eye State dataset from the UCI repository and returns its features, targets, metadata, and variables.

    Returns:
        tuple: A tuple containing:
            - X (pandas.DataFrame): Features of the dataset.
            - y (pandas.DataFrame): Targets of the dataset.
            - metadata (dict): Metadata of the dataset.
            - variables (dict): Variable information of the dataset.
    """
    # fetch dataset
    eeg_eye_state = fetch_ucirepo(id=264)

    # data (as pandas dataframes)
    X = eeg_eye_state.data.features
    y = eeg_eye_state.data.targets

    # metadata
    metadata = eeg_eye_state.metadata

    # variable information
    variables = eeg_eye_state.variables

    return X, y, metadata, variables

# %%
def export_data_to_csv(X, y, output_dir, data_name='eeg_eye_state'):
    """
    Exports the dataset features and targets to CSV files.

    Args:
        X (pandas.DataFrame): Features of the dataset.
        y (pandas.DataFrame): Targets of the dataset.
        metadata (dict): Metadata of the dataset.
        variables (dict): Variable information of the dataset.
        output_dir (str): Directory where the CSV files will be saved.
    """
    full_data = X.copy()
    full_data['target'] = y
    full_data.to_csv(f"{output_dir}/{data_name}_full.csv", index=False)


# %%
if __name__ == "__main__":
    X, y, _,_ = fetch_eeg_eye_state_data()
    export_data_to_csv(X, y,output_dir='data') # TODO: only need once when there is new data
