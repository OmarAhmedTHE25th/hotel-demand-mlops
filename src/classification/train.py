"""Training workflows for baseline classification models."""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import pandas as pd

from src.classification.evaluate import (
    compute_classification_metrics,
    format_metrics_table,
    plot_confusion_matrix,
)
from src.classification.models import get_baseline_models
from src.data.loader import load_config
from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.splits import split_classification


def train_baseline_classification(
    task: str = "classification",
    config: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """Train all 3 baseline classifiers on train split and evaluate on val split."""
    if task != "classification":
        raise ValueError(f"Baseline classification train called for task: {task}")
    if config is None:
        config = load_config()

    random_seed = int(config.get("random_seed", 42))
    cleaned_data_path = config["cleaned_data_path"]
    data = pd.read_csv(cleaned_data_path)

    X_train, X_val, _, y_train, y_val, _ = split_classification(data, config)

    preprocessor = build_preprocessor("classification", config)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    models = get_baseline_models(random_seed=random_seed)
    results = {}

    artifacts_dir = Path(config.get("artifacts_dir", "artifacts"))
    outputs_dir = Path(config.get("outputs_dir", "outputs"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if save_artifacts:
        joblib.dump(preprocessor, artifacts_dir / "preprocessor_classification.joblib")

    for name, model in models.items():
        print(f"Training baseline: {name}...")
        model.fit(X_train_trans, y_train)

        y_pred = model.predict(X_val_trans)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val_trans)[:, 1]

        metrics = compute_classification_metrics(y_val, y_pred, y_prob)
        results[f"{name}_baseline"] = metrics

        plot_confusion_matrix(
            y_val,
            y_pred,
            model_name=f"{name}_baseline",
            output_path=outputs_dir / f"cm_{name}_baseline.png",
        )

        if save_artifacts:
            joblib.dump(model, artifacts_dir / f"model_classification_{name}_baseline.joblib")

    report_df = format_metrics_table(results)
    print("\n--- Baseline Validation Performance ---")
    print(report_df.to_string())

    report_df.to_csv(outputs_dir / "classification_baseline_metrics.csv")
    return report_df