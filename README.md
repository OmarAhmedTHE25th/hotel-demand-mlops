# CSE382 University ML Project — City Hotel (H2)

## Team members and GitHub accounts

- Omar - GitHub: `OmarAhmedTHE25th` - Task A (Regression)
- Team member: `<Name>` — GitHub: `<username>`
- Team member: `<Name>` — GitHub: `<username>`

## Repository

- GitHub repository URL: `<repository URL>`
- Commit SHA: `<commit SHA>`

## Environment setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data download

Place the City Hotel booking dataset at `data/H2.csv`. It can be downloaded from:

<https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv>

The dataset is not tracked by Git.

## Run commands

Analyze the dataset:

```bash
python main.py analyze
```

Run the tests:

```bash
pytest
```

## Milestone 1 - Task A: ADR regression (Omar)

### Prediction point and feature choices

Estimate a booking's eventual Average Daily Rate (`ADR`) when it first enters
the PMS. This is a retrospective approximation of booking-time prediction:
H2 does not provide versioned snapshots of every field at booking creation.
The published ADR is the outcome, not a quoted rate supplied as an input.

Candidate predictors reuse the team's source-label mapping and feature builder:

| Inputs | Why they may explain ADR / timing assumption |
|---|---|
| `LeadTime` and arrival year/month/week/day | Advance booking and seasonality; assume the original planned arrival date. |
| Weekday/weekend nights, `Adults`, `Children`, `Babies` | Stay and party composition; assume booking-time quantities rather than final amendments. |
| `Meal`, `ReservedRoomType` | The requested product/package; assume the original reservation. |
| `Country`, `MarketSegment`, `DistributionChannel`, `CustomerType` | Booking context and distribution; assume information available on creation. |
| `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled` | Prior history only; it must exclude events after the current booking's creation. |
| `DepositType`, `Agent`, `Company` | Booking terms and intermediaries; assume these are established on creation. |

The shared preprocessor derives total nights, total guests, `HasAgent`, and
`HasCompany`, uses `log1p(LeadTime)`, and replaces the sparse Company identifier
with its presence indicator. These transformations and the team's category
grouping are reused rather than implemented again.

### Preliminary leakage audit

- `ADR` is removed from inputs; no ADR-derived feature or target encoding is used.
- `IsCanceled`, `ReservationStatus`, and `ReservationStatusDate` reveal later
  outcomes and are excluded.
- `AssignedRoomType`, `BookingChanges`, and `DaysInWaitingList` can reflect
  later events and are excluded by the team's feature definition.
- `RequiredCarParkingSpaces` and `TotalOfSpecialRequests` are excluded for this
  regression run through `regression_exclude_features` in `config.yaml`, because
  the shared audit flags possible updates after booking. Other tasks keep their
  existing feature choices.
- Median/mode imputation, category grouping, one-hot encoding, and scaling are
  fitted only on training rows inside each complete model pipeline. Validation
  uses `predict`, never `fit` or `fit_transform`.
- Shared exact-duplicate removal occurs before splitting. Identical predictor
  rows with different outcomes and related bookings can still cross partitions;
  there are no reliable booking/customer group IDs in the selected inputs.
- The random split estimates performance on a mixed historical population, not
  a guaranteed forecast of later bookings. Temporal evaluation remains future
  work. Amendable fields and historical counters still need provenance checks;
  excluding obvious outcome fields does not establish full temporal safety.

### Reproduction and evaluation protocol

From the repository root in Windows PowerShell, activate the project's virtual
environment and install `requirements.txt` as described above. Place the original
H2 CSV at `data/H2.csv`, then run:

```powershell
python main.py train --task regression
python -m pytest -q
```

The training command calls the team's loader and deterministic `clean_data`,
then applies the shared `regression_filter`: **0 < ADR <= 500**. It reads the
committed `data/splits/regression_indices.json` rather than generating new
splits. Positions refer to the cleaned, ADR-filtered frame in its existing row
order. The manifest must be a disjoint, complete partition of that population.
Do not regenerate or overwrite the shared splits while comparing models.

Training uses the shared 70% training subset; the 15% validation subset is used
for initial comparison. No final test rows are selected for preprocessing,
prediction, or scoring by this training command. The existing general-purpose
`preprocess` and `split` commands are not needed for this experiment.

Both models are untuned: `DummyRegressor(strategy="mean")` predicts the training
mean, and `LinearRegression()` estimates a linear relationship after shared
preprocessing. The dummy is a reference baseline, not a second regression model
family toward the final three-model requirement. Neither model clips predictions.

Generated files:

- `outputs/regression/validation_metrics.csv`: MAE, MSE, RMSE, R-squared, and fit time.
- `outputs/regression/run_metadata.json`: feature list, configuration, versions,
  train/validation counts, and raw-data/split SHA-256 hashes.
- `artifacts/regression_dummy_mean.joblib` and
  `artifacts/regression_linear_regression.joblib`: fitted preprocessing-plus-model
  pipelines for loading with `joblib.load`. These files are ignored by Git and
  must be excluded from the submission ZIP.

The existing split manifest has no original dataset fingerprint. Coverage checks
detect row-count/partition problems but cannot prove the identity or ordering of
a different same-sized dataset. Use the specified original H2 file, shared
cleaning, and recorded hashes consistently.

### Initial results

Measured on 31 October 2026 using 36,623 training rows and 7,848 validation
rows from the committed shared split. Cleaning leaves 53,043 rows; the shared
ADR filter removes another 724, leaving 52,319 rows across all partitions.

| Untuned model | Validation MAE | Validation MSE | Validation RMSE | Validation R-squared |
|---|---:|---:|---:|---:|
| Training-mean dummy | 30.6516 | 1586.2637 | 39.8279 | -0.000027 |
| Linear Regression | 15.0267 | 447.8113 | 21.1616 | 0.717687 |

Linear Regression reduces validation MAE by approximately 51% relative to the
training-mean baseline. Its mean absolute error is about 15 ADR units per
booking; the larger RMSE indicates that some errors are substantially larger.
R-squared compares squared error against a validation-mean reference; it is not
an accuracy percentage. The dummy's slightly negative R-squared is possible
because its training mean differs from the validation mean. No final test result
is reported for Milestone 1.

The recorded environment was Python 3.12.14, scikit-learn 1.9.1, pandas 3.0.1,
and NumPy 2.5.4. Versions and file hashes are stored in `run_metadata.json`.
The regression, loader, cleaning, preprocessing, and split tests passed (17 tests).
The existing full-data EDA test was not rerun as part of this regression check.

An earlier independent prototype reported MAE 15.24, RMSE 22.19, and R-squared
0.720. Those are prototype-only validation scores and are not evidence of this
integrated pipeline's performance.

### Limitations, contributions, and next steps

The ADR filter excludes zero/negative rates and rates above 500; performance is
conditional on that restricted population and should not be generalized to all
hotel bookings. The threshold comes from shared configuration rather than
validation optimization. Prior team EDA inspected the full data distribution,
so the test set was not fully blind for that exploratory stage; this regression
experiment performs no test evaluation. Linear Regression cannot directly model
all nonlinearities or interactions, and redundant engineered features mean
individual coefficient interpretations need care. Country and intermediary
features can encode population differences and require generalization checks.

**Omar's contribution:** regression training/evaluation module, CLI integration,
read-only reuse and validation of the shared split manifest, conservative
regression feature configuration, holdout-isolation tests, and this report section.
The data loader, cleaning, preprocessing builder, original feature definitions,
and original split assignments are the preprocessing teammate's contribution.

Next: review booking-time availability with the team/TA; compare additional
untuned models such as Random Forest and SVR; tune at least three selected
regression families using training-only CV with the complete preprocessing
pipeline inside each fold; coordinate PCA comparisons; select by a declared
validation/CV metric (proposed MAE); evaluate the frozen selected model on the
final test set only at Milestone 2; integrate saved-pipeline inference with the
application teammate. The final three-model requirement includes baseline and
tuned configurations for every selected family.
