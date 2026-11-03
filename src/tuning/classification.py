"""Hyperparameter tuning for classification models using Optuna."""

import optuna
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from src.classification.evaluate import compute_classification_metrics, format_metrics_table
from src.classification.models import get_baseline_models
from src.data.loader import load_config
from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.splits import split_classification

def objective(trial, model_name, X, y, random_seed):
    if model_name == "logistic_regression":
        c = trial.suggest_float("C", 1e-3, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        model = LogisticRegression(C=c, solver=solver, max_iter=1000, random_state=random_seed)
    
    elif model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            random_state=random_seed,
            n_jobs=-1
        )
        
    elif model_name == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_seed,
            eval_metric="logloss",
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    score = cross_val_score(model, X, y, n_jobs=-1, cv=3, scoring="f1").mean()
    return score

def tune_classification(
    task: str = "classification",
    config: Optional[Dict[str, Any]] = None,
    n_trials: int = 20
) -> pd.DataFrame:
    """Tune all baseline classifiers and return a comparison report."""
    if task != "classification":
        raise ValueError(f"Tuning called for task: {task}")
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

    for name in models.keys():
        print(f"Tuning model: {name}...")
        # Reduce dataset for faster tuning demonstration if needed, 
        # but here we just use few trials.
        n_trials_to_run = 5 if name in ["random_forest", "xgboost"] else 10
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, name, X_train_trans, y_train, random_seed), n_trials=n_trials_to_run)
        
        print(f"Best params for {name}: {study.best_params}")
        
        # Train best model
        if name == "logistic_regression":
            best_model = LogisticRegression(**study.best_params, max_iter=1000, random_state=random_seed)
        elif name == "random_forest":
            best_model = RandomForestClassifier(**study.best_params, random_state=random_seed, n_jobs=-1)
        elif name == "xgboost":
            best_model = XGBClassifier(**study.best_params, random_state=random_seed, eval_metric="logloss", n_jobs=-1)
        
        best_model.fit(X_train_trans, y_train)
        
        y_pred = best_model.predict(X_val_trans)
        y_prob = None
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_val_trans)[:, 1]

        metrics = compute_classification_metrics(y_val, y_pred, y_prob)
        results[f"{name}_tuned"] = metrics

        joblib.dump(best_model, artifacts_dir / f"model_classification_{name}_tuned.joblib")

    report_df = format_metrics_table(results)
    print("\n--- Tuned Validation Performance ---")
    print(report_df.to_string())
    
    report_df.to_csv(outputs_dir / "classification_tuned_metrics.csv")
    return report_df
