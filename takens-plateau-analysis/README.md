# Takens Plateau Analysis

This project implements an analysis framework for exploring the geometry of attractors derived from time series data using Takens' embedding. The primary focus is on determining the common dimensionality at which the mean pairwise distances plateau for various threshold values (R_th), ensuring robustness in the analysis.

## Overview

The project consists of several components:

- **Data Loading and Preprocessing**: Functions to load and normalize the EEG Eye State dataset.
- **Takens' Embedding**: Implementation of Takens' embedding to convert time series data into point clouds.
- **Plateau Analysis**: Functions to compute mean pairwise distances for different values of R_th and identify the dimensionality at which these distances stabilize.
- **Visualizations**: Tools for visualizing the results, including plots of mean pairwise distances against embedding dimensions.

## Project Structure

```
takens-plateau-analysis
├── src
│   ├── plateau_analysis.py      # Functions for mean pairwise distance analysis
│   ├── takens_embedding.py       # Takens' embedding implementation
│   ├── utils.py                  # Utility functions for data handling
│   └── visualizations.py          # Visualization functions
├── notebooks
│   └── eeg_takens_embedding.ipynb # Jupyter notebook for interactive analysis
├── tests
│   └── test_plateau_analysis.py   # Unit tests for plateau analysis functions
├── data
│   └── eeg_eye_state_full.csv      # EEG Eye State dataset
├── requirements.txt                # Project dependencies
├── .gitignore                      # Files to ignore in Git
└── README.md                       # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd takens-plateau-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the dataset (`eeg_eye_state_full.csv`) is located in the `data` directory.

## Usage

- Use the Jupyter notebook `eeg_takens_embedding.ipynb` for an interactive exploration of the EEG Eye State dataset.
- The functions in `plateau_analysis.py` can be imported to compute mean pairwise distances and analyze the plateau behavior for different embedding dimensions and threshold values.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.