"""Run regression metrics for imputed datasets and aggregate the results."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the regression evaluation pipeline."""

    orig_data_path: Path
    imputed_data_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for regression metrics."""

    rmse: float
    mse: float


def calculate_regression_metrics(
    orig_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
) -> RegressionMetrics:
    """Calculate RMSE and MSE for aligned numeric data frames."""
    if orig_data.shape != imputed_data.shape:
        raise ValueError(
            f"Shape mismatch: orig={orig_data.shape}, imputed={imputed_data.shape}"
        )
    mse = ((orig_data - imputed_data) ** 2).mean().mean()
    rmse = mse**0.5
    return RegressionMetrics(rmse=rmse, mse=mse)


def parse_imputation_labels(file_path: Path) -> tuple[str, str]:
    """Parse imputation method and missing mechanism from filename."""
    tokens = file_path.stem.split("_")
    imputation_method = tokens[0]
    missing_mechanism = tokens[-1] if len(tokens) > 1 else "unknown"
    return imputation_method, missing_mechanism


def list_imputed_files(imputed_dir: Path) -> list[Path]:
    """Return all imputed CSV files under the provided directory."""
    if not imputed_dir.exists():
        raise FileNotFoundError(f"Imputed data directory not found: {imputed_dir}")
    files = sorted(imputed_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No imputed CSV files found in {imputed_dir}")
    return files


def evaluate_imputations(
    config: PipelineConfig,
) -> Iterable[Path]:
    """Compute metrics for each imputed file and write per-file results."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    orig_data = pd.read_csv(config.orig_data_path)

    for imputed_file in list_imputed_files(config.imputed_data_dir):
        imputation_method, missing_mechanism = parse_imputation_labels(imputed_file)
        LOGGER.info("Processing imputed file: %s", imputed_file.name)
        imputed_data = pd.read_csv(imputed_file)
        metrics = calculate_regression_metrics(orig_data, imputed_data)
        result_df = pd.DataFrame(
            {
                "Imputation Method": [imputation_method],
                "Missing Mechanism": [missing_mechanism],
                "RMSE": [metrics.rmse],
                "MSE": [metrics.mse],
            }
        )
        result_file = (
            config.results_dir
            / f"{imputation_method}_{missing_mechanism}_results.csv"
        )
        result_df.to_csv(result_file, index=False)
        yield result_file


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate regression results from per-file CSV outputs."""
    result_files = sorted(results_dir.glob("*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No regression results found in {results_dir}")
    frames = [pd.read_csv(result_file) for result_file in result_files]
    return pd.concat(frames, ignore_index=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute RMSE/MSE for imputed datasets and aggregate results."
    )
    parser.add_argument(
        "--orig-data",
        type=Path,
        default=Path("data/eeg_eye_state_full.csv"),
        help="Path to the original dataset CSV.",
    )
    parser.add_argument(
        "--imputed-dir",
        type=Path,
        default=Path("imp_data"),
        help="Directory containing imputed CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("regression_results"),
        help="Directory for per-file regression results.",
    )
    return parser


def main() -> None:
    """Entry point for CLI usage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()
    config = PipelineConfig(
        orig_data_path=args.orig_data,
        imputed_data_dir=args.imputed_dir,
        results_dir=args.results_dir,
    )

    imputed_files = list_imputed_files(config.imputed_data_dir)
    LOGGER.info("Imputed files found: %s", [f.name for f in imputed_files])
    list(evaluate_imputations(config))

    combined_results_df = aggregate_results(config.results_dir)
    print(combined_results_df)


if __name__ == "__main__":
    main()
