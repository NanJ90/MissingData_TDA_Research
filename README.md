# aaai-topology

This repository provides a reproducible pipeline for generating missing data
patterns, imputing them with multiple methods, and evaluating regression and
classification performance.

## Datasets
1. EEG Eye State via UCI [link](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
<!-- 2. PEMS dataset [link](https://www.kaggle.com/datasets/elmahy/pems-dataset/data) -->
2. Jena Climate

Mac:
```
curl -O https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
unzip jena_climate_2009_2016.csv.zip
```

Linux:
```
wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
unzip jena_climate_2009_2016.csv.zip
```
Data files are large and are not uploaded to GitHub; you can download them using the steps above or contact me (njia@gradcenter.cuny.edu) for the dataset list.
## Pipeline Process
### Stage 1: Data preparation
1. Generate missing data:
   `python missing_data_generation.py`
2. Run imputation baselines:
   `python imputation.py`
3. Evaluate regression metrics for imputed outputs:
   `python simple_pipeline.py`
4. Evaluate classification metrics:
   `python inference.py`
5. TDQA analysis notebooks:
   `analysis/TDA_TSA.ipynb`

## Directory Layout

- `regression_results/`: per-method RMSE/MSE outputs (CSV)
- `classification_results/`: per-method classification metrics (TXT/CSV)
- `analysis/`: notebooks for exploratory and forecasting analysis

## Script Reference
- `missing_data_generation.py`: interactive generation of MCAR/MAR/MNAR masks
- `imputation.py`: KNN, interpolation, LOCF, and GAIN imputations
- `simple_pipeline.py`: regression evaluation + aggregated metrics table
- `inference.py`: classification evaluation + summary CSV

There are also images, results, and EDA notebooks for reference.
