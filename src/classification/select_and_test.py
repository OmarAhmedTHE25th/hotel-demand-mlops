"""Evaluate candidate models and test set performance for Task B."""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import pandas as pd

from src.classification.evaluate import (
    compute_classification_metrics,
    format_metrics_table,
    plot_confusion_matrix,
    plot_roc_pr_curves,
)
from src.data.loader import load_config
from src.preprocessing.splits import split_classification


def evaluate_final_classification(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Compare baseline vs tuned, select the best model, and run test evaluation."""
    if config is None:
        config = load_config()

    outputs_dir = Path(config.get("outputs_dir", "outputs"))
    artifacts_dir = Path(config.get("artifacts_dir", "artifacts"))

    baseline_csv = outputs_dir / "classification_baseline_metrics.csv"
    tuned_csv = outputs_dir / "classification_tuned_metrics.csv"

    if not baseline_csv.is_file() or not tuned_csv.is_file():
        raise FileNotFoundError(
            "Run 'main.py train --task classification' and 'main.py tune --task classification' first.")

    df_base = pd.read_csv(baseline_csv, index_col=0)
    df_tuned = pd.read_csv(tuned_csv, index_col=0)
    comparison = pd.concat([df_base, df_tuned]).sort_values(by="roc_auc", ascending=False)

    print("\n=== All Classification Models (Validation Performance) ===")
    print(comparison.to_string())
    comparison.to_csv(outputs_dir / "classification_all_models_comparison.csv")

    # Select best model based on validation ROC-AUC
    best_model_name = comparison.index[0]
    print(f"\nSelected best model: {best_model_name}")

    # Load test data and fitted preprocessor
    cleaned_data_path = config["cleaned_data_path"]
    data = pd.read_csv(cleaned_data_path)
    _, _, X_test, _, _, y_test = split_classification(data, config)

    preprocessor = joblib.load(artifacts_dir / "preprocessor_classification.joblib")
    best_model = joblib.load(artifacts_dir / f"model_classification_{best_model_name}.joblib")

    X_test_trans = preprocessor.transform(X_test)
    y_pred = best_model.predict(X_test_trans)
    y_prob = best_model.predict_proba(X_test_trans)[:, 1] if hasattr(best_model, "predict_proba") else None

    test_metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    test_df = format_metrics_table({f"{best_model_name}_final_test": test_metrics})

    print("\n=== Final Test Set Evaluation ===")
    print(test_df.to_string())
    test_df.to_csv(outputs_dir / "classification_final_test_metrics.csv")

    # Plot test confusion matrix and curves
    plot_confusion_matrix(
        y_test,
        y_pred,
        model_name=f"{best_model_name}_test",
        output_path=outputs_dir / f"cm_{best_model_name}_test.png",
    )
    if y_prob is not None:
        plot_roc_pr_curves(
            y_test,
            y_prob,
            model_name=f"{best_model_name}_test",
            output_dir=outputs_dir,
        )

    # Save as canonical inference model
    joblib.dump(best_model, artifacts_dir / "model_classification_best.joblib")
    print(f"Saved canonical deployment artifact: {artifacts_dir / 'model_classification_best.joblib'}")

    return test_df