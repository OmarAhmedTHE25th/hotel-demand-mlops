"""Task-specific feature lists and leakage documentation.

Classification and regression features assume prediction at booking time,
when a booking enters the PMS. The target column is returned separately.
"""

from typing import Any


_COMMON_TASK_EXCLUSIONS = ["classification", "regression", "clustering"]

EXCLUDED = {
    "ReservationStatus": {
        "tasks": _COMMON_TASK_EXCLUSIONS,
        "type": "direct",
        "reason": "Encodes the final reservation outcome.",
    },
    "ReservationStatusDate": {
        "tasks": _COMMON_TASK_EXCLUSIONS,
        "type": "direct",
        "reason": "The date is associated with the final reservation outcome.",
    },
    "BookingChanges": {
        "tasks": ["classification", "regression"],
        "type": "indirect",
        "reason": "May change after booking and is unavailable at the assumed prediction point.",
    },
    "AssignedRoomType": {
        "tasks": ["classification", "regression"],
        "type": "indirect",
        "reason": "May be assigned or changed after booking.",
    },
    "DaysInWaitingList": {
        "tasks": ["classification", "regression"],
        "type": "indirect",
        "reason": "Can reflect post-booking waiting-list events.",
    },
    "IsCanceled": {
        "tasks": ["regression", "clustering"],
        "type": "direct",
        "reason": "Classification target; excluded as an input for other tasks.",
    },
    "ADR": {
        "tasks": ["regression"],
        "type": "direct",
        "reason": "Regression target; returned separately from its input features.",
    },
}

BORDERLINE = {
    "ADR": {
        "tasks": ["classification"],
        "type": "borderline",
        "reason": (
            "May be informative at booking time, but is kept out of classification "
            "by default; enable classification_use_adr to include it."
        ),
    },
    "RequiredCarParkingSpaces": {
        "tasks": ["classification", "regression"],
        "type": "borderline",
        "reason": "May be updated after booking.",
    },
    "TotalOfSpecialRequests": {
        "tasks": ["classification", "regression", "clustering"],
        "type": "borderline",
        "reason": "May be updated after booking.",
    },
}

CLASSIFICATION_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

REGRESSION_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

CLUSTERING_FEATURES = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "MarketSegment",
    "DistributionChannel",
    "DepositType",
    "CustomerType",
    "ADR",
    "IsRepeatedGuest",
    "TotalOfSpecialRequests",
]

TASK_FEATURES = {
    "classification": CLASSIFICATION_FEATURES,
    "regression": REGRESSION_FEATURES,
    "clustering": CLUSTERING_FEATURES,
}

TASK_TARGETS = {
    "classification": "IsCanceled",
    "regression": "ADR",
    "clustering": None,
}


def get_features(task: str, config: dict[str, Any]) -> list[str]:
    """Return a copy of the input feature names for a task and config."""
    if task not in TASK_FEATURES:
        raise ValueError(
            f"Unsupported task {task!r}; expected one of {', '.join(TASK_FEATURES)}."
        )

    features = list(TASK_FEATURES[task])
    if task == "classification" and config.get("classification_use_adr", False):
        if "ADR" not in features:
            features.append("ADR")
    return features


def print_feature_report() -> None:
    """Print kept features and documented exclusions for each task."""
    for task in TASK_FEATURES:
        print(f"{task.title()} kept features:")
        print(", ".join(get_features(task, {"classification_use_adr": False})))
        target = TASK_TARGETS[task] or "None (unsupervised)"
        print(f"Target: {target}")

        print("Excluded columns:")
        excluded = {
            column: details
            for column, details in EXCLUDED.items()
            if task in details["tasks"]
        }
        for column, details in excluded.items():
            print(f"  {column} [{details['type']}]: {details['reason']}")

        print("Borderline columns:")
        for column, details in BORDERLINE.items():
            if task in details["tasks"]:
                print(f"  {column} [{details['type']}]: {details['reason']}")
        print()
