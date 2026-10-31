"""Command-line entry point for the CSE382 project."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import joblib
import pandas as pd

from src.classification.train import train_baseline_classification
from src.tuning.classification import tune_classification
from src.data.cleaning import clean_data
from src.data.eda import run_eda
from src.data.loader import load_config, load_h2
from src.features.feature_sets import get_features, print_feature_report
from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.splits import split_classification, split_regression
from src.classification.select_and_test import evaluate_final_classification

def analyze_data() -> None:
    """Load the hotel booking data and print a basic dataset summary."""
    data = load_h2()
    print(f"Shape: {data.shape}")
    print("\nDtypes:")
    print(data.dtypes)
    print("\nMissing values per column:")
    print(data.isna().sum())


def clean_dataset() -> None:
    """Print ADR distribution details and save the rule-cleaned dataset."""
    config = load_config()
    data = load_h2()
    print("ADR quantiles:")
    print(data["ADR"].quantile([0.50, 0.95, 0.99, 0.999]))
    print("\nTop 10 largest ADR values:")
    print(data["ADR"].nlargest(10).to_string(index=False))
    print(f"\nConfigured adr_max for later regression filtering: {config['adr_max']}")

    cleaned = clean_data(data)
    output_path = config["cleaned_data_path"]
    output_directory = Path(output_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


def split_dataset(task: str) -> None:
    """Load cleaned data and create reproducible splits for a task."""
    config = load_config()
    data = pd.read_csv(config["cleaned_data_path"])
    split_function = (
        split_classification if task == "classification" else split_regression
    )
    split_function(data, config)


def preprocess_dataset(task: str) -> None:
    """Fit preprocessing on training data and report transformed shapes."""
    config = load_config()
    data = pd.read_csv(config["cleaned_data_path"])

    if task == "classification":
        X_train, X_val, X_test, _, _, _ = split_classification(data, config)
    elif task == "regression":
        X_train, X_val, X_test, _, _, _ = split_regression(data, config)
    else:
        X_train = data.loc[:, get_features(task, config)].copy()
        X_val = X_train.copy()
        X_test = X_train.copy()

    preprocessor = build_preprocessor(task, config)
    train_transformed = preprocessor.fit_transform(X_train)
    val_transformed = preprocessor.transform(X_val)
    test_transformed = preprocessor.transform(X_test)

    print(f"Transformed train shape: {train_transformed.shape}")
    print(f"Transformed validation shape: {val_transformed.shape}")
    print(f"Transformed test shape: {test_transformed.shape}")
    print(f"Number of features: {train_transformed.shape[1]}")
    no_nan = not (
        pd.isna(train_transformed).any()
        or pd.isna(val_transformed).any()
        or pd.isna(test_transformed).any()
    )
    print(
        f"No NaN in transformed data: {no_nan}"
    )

    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = artifacts_dir / f"preprocessor_{task}.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Fitted preprocessor saved to: {preprocessor_path}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser; future project commands can be added here."""
    parser = argparse.ArgumentParser(description="CSE382 City Hotel ML project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Print a basic summary of the H2 dataset"
    )
    analyze_parser.set_defaults(handler=analyze_data)

    eda_parser = subparsers.add_parser(
        "eda", help="Run exploratory data analysis and save figures"
    )
    eda_parser.set_defaults(handler=run_eda)

    clean_parser = subparsers.add_parser(
        "clean", help="Apply deterministic cleaning rules and save the result"
    )
    clean_parser.set_defaults(handler=clean_dataset)

    features_parser = subparsers.add_parser(
        "features", help="Print task feature sets and leakage documentation"
    )
    features_parser.set_defaults(handler=print_feature_report)

    split_parser = subparsers.add_parser(
        "split", help="Create reproducible train/validation/test splits"
    )
    split_parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        required=True,
        help="Prediction task to split",
    )
    split_parser.set_defaults(handler=split_dataset)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Fit task preprocessing on training data"
    )
    preprocess_parser.add_argument(
        "--task",
        choices=("classification", "regression", "clustering"),
        required=True,
        help="Task-specific preprocessor to fit",
    )
    preprocess_parser.set_defaults(handler=preprocess_dataset)
    
    train_parser = subparsers.add_parser(
        "train", help="Train classification or regression models"
    )
    train_parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        required=True,
        help="Task to train",
    )
    train_parser.set_defaults(handler=train_baseline_classification)
    
    tune_parser = subparsers.add_parser(
        "tune", help="Tune classification or regression models"
    )
    tune_parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        required=True,
        help="Task to tune",
    )
    tune_parser.set_defaults(handler=tune_classification)

    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate final models and report test metrics"
    )
    eval_parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        required=True,
        help="Task to evaluate",
    )
    eval_parser.set_defaults(handler=evaluate_final_classification)
    
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse command-line arguments and run the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"split", "preprocess", "train", "tune"}:
        args.handler(args.task)
    elif args.command == "evaluate":
        args.handler()
    else:
        args.handler()

if __name__ == "__main__":
    main()
