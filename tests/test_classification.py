import numpy as np
import pandas as pd
import pytest
from src.classification.evaluate import compute_classification_metrics
from src.classification.models import get_baseline_models
from src.preprocessing.pipeline import build_preprocessor

def test_compute_classification_metrics_standard():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

def test_compute_classification_metrics_zero_division():
    # Case where precision/recall might have zero division
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    
    y_true_zero = np.array([0, 0, 0, 0])
    y_pred_one = np.array([1, 1, 1, 1])
    metrics_zero = compute_classification_metrics(y_true_zero, y_pred_one)
    assert metrics_zero["precision"] == 0.0
    assert metrics_zero["recall"] == 0.0

def test_model_pipeline_output_dimensions():
    # Create dummy data
    data = pd.DataFrame({
        "LeadTime": [10, 20, 30, 40],
        "ArrivalDateMonth": ["July", "July", "August", "August"],
        "ArrivalDateYear": [2015, 2015, 2015, 2015],
        "ArrivalDateWeekNumber": [27, 27, 27, 27],
        "ArrivalDateDayOfMonth": [1, 1, 1, 1],
        "StaysInWeekendNights": [0, 0, 0, 0],
        "StaysInWeekNights": [2, 2, 2, 2],
        "Adults": [2, 2, 2, 2],
        "Children": [0, 0, 0, 0],
        "Babies": [0, 0, 0, 0],
        "Meal": ["BB", "BB", "BB", "BB"],
        "Country": ["PRT", "PRT", "PRT", "PRT"],
        "MarketSegment": ["Direct", "Direct", "Direct", "Direct"],
        "DistributionChannel": ["Direct", "Direct", "Direct", "Direct"],
        "IsRepeatedGuest": [0, 0, 0, 0],
        "PreviousCancellations": [0, 0, 0, 0],
        "PreviousBookingsNotCanceled": [0, 0, 0, 0],
        "ReservedRoomType": ["C", "C", "C", "C"],
        "AssignedRoomType": ["C", "C", "C", "C"],
        "BookingChanges": [3, 4, 3, 4],
        "DepositType": ["No Deposit", "No Deposit", "No Deposit", "No Deposit"],
        "Agent": ["NULL", "NULL", "NULL", "NULL"],
        "Company": ["NULL", "NULL", "NULL", "NULL"],
        "DaysInWaitingList": [0, 0, 0, 0],
        "CustomerType": ["Transient", "Transient", "Transient", "Transient"],
        "RequiredCarParkingSpaces": [0, 0, 0, 0],
        "TotalOfSpecialRequests": [0, 0, 0, 0],
        "ReservationStatus": ["Check-Out", "Check-Out", "Check-Out", "Check-Out"],
        "ReservationStatusDate": ["2015-07-03", "2015-07-03", "2015-07-03", "2015-07-03"],
        "IsCanceled": [0, 0, 1, 1]
    })
    
    config = {"classification_use_adr": False}
    from src.features.feature_sets import get_features
    X = data[get_features("classification", config)]
    y = data["IsCanceled"]
    
    preprocessor = build_preprocessor("classification", config)
    X_trans = preprocessor.fit_transform(X)
    
    models = get_baseline_models(random_seed=42)
    
    for name, model in models.items():
        model.fit(X_trans, y)
        y_pred = model.predict(X_trans)
        assert y_pred.shape == (4,)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_trans)
            assert y_prob.shape == (4, 2)
