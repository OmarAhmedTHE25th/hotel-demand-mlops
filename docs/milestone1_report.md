# CSE382 Milestone 1: Data Preparation and Leakage Controls

## 1. Scope and repository

This section documents the data-side work in the CSE382 City Hotel (H2) project: dataset loading and verification, exploratory data analysis (EDA), rule-based cleaning, task-specific feature selection, splitting, and preprocessing. Model training and baseline comparisons are outside this section and are not implemented in the inspected source tree.

- **GitHub repository:** <https://github.com/OmarAhmedTHE25th/hotel-demand-mlops>
- **Commit SHA:** `<commit SHA>` (the workspace has no Git metadata, so a local commit could not be verified)
- **Author:** `<name and GitHub handle>`
- **Dataset:** City Hotel H2; `load_h2()` asserts a shape of `(79330, 31)` and reports the dataset download URL on missing-file errors.

The report was reproduced with these commands, using the workspace Python 3.13 environment:

```bash
python main.py analyze
python main.py eda
python main.py clean
python main.py features
python main.py split --task classification
python main.py split --task regression
python main.py preprocess --task classification
python main.py preprocess --task regression
python main.py preprocess --task clustering
pytest
```

All commands exited successfully. `pytest` reported **14 passed** and one third-party seaborn pending-deprecation warning.

## 2. Dataset overview

The loader verified the expected **79,330 rows and 31 columns**. `analyze` reported 16 `int64`, 13 `str`, and 2 `float64` columns. The two floating-point columns are `Children` and `ADR`; the string columns are `ArrivalDateMonth`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `AssignedRoomType`, `DepositType`, `Agent`, `Company`, `CustomerType`, `ReservationStatus`, and `ReservationStatusDate`. All remaining columns are `int64`.

Raw-data statistics, calculated directly from `data/H2.csv`:

| Statistic | `LeadTime` | `ADR` |
|---|---:|---:|
| Count | 79,330 | 79,330 |
| Mean | 109.7357 | 105.3045 |
| Standard deviation | 110.9485 | 43.6030 |
| Minimum | 0 | 0 |
| 25th percentile | 23 | 79.2 |
| Median | 74 | 99.9 |
| 75th percentile | 163 | 126 |
| 95th percentile | 334.55 | 175.5 |
| 99th percentile | 461 | 228 |
| 99.9th percentile | 615 | 289.59048 |
| Maximum | 629 | 5,400 |

The raw `IsCanceled` proportions are 0.58273 for class `0` and 0.41727 for class `1`.

## 3. Data quality findings

The EDA reports missing values by treating both NaN and case-insensitive, whitespace-trimmed `"NULL"` strings as missing. In the raw `analyze` output, `Agent` and `Company` each have zero NaN values because their missing entries are stored as strings.

| Column | Missing entries (NaN or `"NULL"`) | Percentage |
|---|---:|---:|
| `Agent` | 8,131 | 10.25% |
| `Company` | 75,641 | 95.35% |
| `Country` | 24 | 0.03% |
| `Children` | 4 | 0.01% |

There are **25,902 exact duplicate rows** in the raw data. The EDA’s raw-data `IsCanceled` proportions are reported above. The raw EDA also found:

| Check | Count/value |
|---|---:|
| `Adults == 0` | 390 |
| `ADR <= 0` | 1,208 |
| Maximum `ADR` | 5,400 |
| Zero total nights | 331 |
| `Adults + Children + Babies == 0` | 167 |
| `Children` NaN | 4 |

The EDA categorical cardinalities are: `ArrivalDateMonth` 12; `Meal` 4; `Country` 166; `MarketSegment` 8; `DistributionChannel` 5; `ReservedRoomType` 8; `AssignedRoomType` 9; `DepositType` 3; `Agent` 224; `Company` 208; `CustomerType` 4; and `ReservationStatus` 3. Since `"NULL"` is a string in the raw file, it is included among the distinct values counted for `Agent` and `Company`.

## 4. Exploratory analysis

`python main.py eda` converts `ReservationStatusDate` to datetime for the EDA run, prints the preceding quality summary and numeric description, and writes nine figures to `outputs/`.

| Figure | Visual interpretation |
|---|---|
| [`adr_histogram.png`](../outputs/adr_histogram.png) | The visible mass is concentrated at the low end of the ADR axis, with a long, sparse right tail. |
| [`adr_boxplot.png`](../outputs/adr_boxplot.png) | Many high-side outliers are visible, including an isolated point around 5,400. |
| [`lead_time_histogram.png`](../outputs/lead_time_histogram.png) | Counts are greatest near zero and generally decline toward the upper end of the plotted range, forming a right-skewed distribution. |
| [`lead_time_boxplot.png`](../outputs/lead_time_boxplot.png) | The boxplot shows a wide upper tail and many high-side outlier points. |
| [`cancellation_by_arrival_month.png`](../outputs/cancellation_by_arrival_month.png) | Monthly cancellation rates appear broadly similar; April is visually among the higher bars and March among the lower bars. |
| [`cancellation_by_deposit_type.png`](../outputs/cancellation_by_deposit_type.png) | `Non Refund` has the highest visible cancellation rate, `Refundable` is lower, and `No Deposit` is lowest. |
| [`cancellation_by_market_segment.png`](../outputs/cancellation_by_market_segment.png) | The displayed rates vary by segment; `Undefined` is highest, while `Complementary` and `Aviation` are among the lowest visible bars. |
| [`numeric_correlation_heatmap.png`](../outputs/numeric_correlation_heatmap.png) | Most off-diagonal cells appear near the neutral color; the `ArrivalDateYear`–`ArrivalDateWeekNumber` cell is visibly negative, while some booking-history feature pairs are positive. |
| [`top_15_countries.png`](../outputs/top_15_countries.png) | `PRT` has a much longer bar than the other displayed countries; `FRA` and `DEU` follow at a substantial distance. |

The figures are descriptive outputs only. The cancellation-rate plots do not display group sample sizes, so their bar heights should not be read as evidence of equal precision across groups.

## 5. Cleaning rules and results

`clean_data()` in [`src/data/cleaning.py`](../src/data/cleaning.py) applies rules in this order:

1. Convert trimmed, case-insensitive `"NULL"` strings in `Agent` and `Company` to missing values; convert `ReservationStatusDate` to datetime with invalid parses coerced to `NaT`. This step removes **0 rows**.
2. Drop exact duplicate rows with `drop_duplicates()`: **25,902 rows removed**.
3. Drop rows with no guests, treating missing `Children` as zero only for this check: **154 rows removed**.
4. Drop rows with zero total nights: **231 rows removed**.
5. Reset the index.

The cleaned result has **53,043 rows and 31 columns**. The cleaning command reported the `IsCanceled` distribution changing from raw proportions of `0: 0.58273`, `1: 0.41727` to cleaned proportions of `0: 0.697943`, `1: 0.302057`.

Duplicate removal is a heuristic, not proof that identical records are accidental copies: distinct bookings can have identical values in every retained column. The code nevertheless removes exact full-row matches before splitting. This choice should be considered when interpreting the resulting population.

The configured regression threshold is `adr_max: 500`. `clean` prints raw `ADR` quantiles (50%: 99.9; 95%: 175.5; 99%: 228; 99.9%: 289.59048) and the top ten raw values (5,400; 510; 451.50; 375.50; 372.33; 365; 352.50; 349.63; 345; 338). The general cleaning function does **not** apply this threshold. `regression_filter()` keeps `0 < ADR <= 500` only for regression; on the cleaned dataset it removes **724 rows**. The command output does not separate how many of those removed rows were nonpositive versus above 500.

## 6. Prediction point and feature selection

The feature definitions assume classification and regression predictions are made at booking time, when the booking enters the PMS. Original dataset column names are retained. The target is returned separately from input features.

**Classification kept features by default** (24):

`LeadTime`, `ArrivalDateYear`, `ArrivalDateMonth`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `ReservedRoomType`, `DepositType`, `Agent`, `Company`, `CustomerType`, `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`.

`IsCanceled` is the classification target. `ADR` is excluded by default; `classification_use_adr: true` would add it to the classification inputs.

**Regression kept features** (24):

`LeadTime`, `ArrivalDateYear`, `ArrivalDateMonth`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `ReservedRoomType`, `DepositType`, `Agent`, `Company`, `CustomerType`, `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`.

`ADR` is the regression target.

**Clustering kept features** (13):

`LeadTime`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `MarketSegment`, `DistributionChannel`, `DepositType`, `CustomerType`, `ADR`, `IsRepeatedGuest`, `TotalOfSpecialRequests`.

The `EXCLUDED` and `BORDERLINE` documentation in [`src/features/feature_sets.py`](../src/features/feature_sets.py) specifies:

| Column | Tasks | Type | Reason recorded in code |
|---|---|---|---|
| `ReservationStatus` | All | Direct | Encodes the final reservation outcome. |
| `ReservationStatusDate` | All | Direct | Date associated with the final reservation outcome. |
| `BookingChanges` | Classification, regression | Indirect | May change after booking and is unavailable at the assumed prediction point. |
| `AssignedRoomType` | Classification, regression | Indirect | May be assigned or changed after booking. |
| `DaysInWaitingList` | Classification, regression | Indirect | Can reflect post-booking waiting-list events. |
| `IsCanceled` | Regression, clustering | Direct | Classification target; excluded as an input for other tasks. It is the classification target. |
| `ADR` | Regression | Direct | Regression target, returned separately from regression inputs. |
| `ADR` | Classification | Borderline | May be informative at booking time; excluded by default but configurable. |
| `RequiredCarParkingSpaces` | Classification, regression | Borderline | May be updated after booking; retained in both feature lists. |
| `TotalOfSpecialRequests` | All | Borderline | May be updated after booking; retained in all feature lists. |

## 7. Preliminary leakage audit

| Stage | What the code does / what it is fitted on | Leakage note |
|---|---|---|
| Cleaning | Deterministic conversions and row filters run on the full raw dataset before splitting; no learned parameters are fitted. | Exact duplicates are detected across the entire dataset before split assignment. |
| Feature selection | Uses static task lists and the `classification_use_adr` config flag. | No data-dependent feature selection is implemented. |
| Split | `train_test_split` assigns row positions; classification uses `IsCanceled` for stratification. Regression first applies the configured ADR target filter, then splits without stratification. | The ADR filter is a target-based rule applied before regression splitting, not a learned model step. |
| Classification/regression preprocessing | Fits the preprocessor on `X_train` only, then transforms validation and test inputs. | No metrics or model evaluation are run by `preprocess`. |
| Clustering preprocessing | The CLI assigns all cleaned rows to `X_train`, `X_val`, and `X_test` (copies of the same feature frame), then fits on the full frame. | This is not a genuine split; clustering preprocessing is fitted on all cleaned records. |
| Test set | Supervised preprocessing transforms test inputs after fitting on train. Split-function docstrings say not to use test data for tuning or model selection. | No tuning or model-selection implementation is present to enforce this rule programmatically. |
| Dimensionality reduction and tuning | No implementation was found in the corresponding source packages; those package directories contain only initializers. | Future work must fit any learned transformation/tuning exclusively within training data or training folds. |

## 8. Train/validation/test strategy

The configuration sets `random_seed: 42`, `val_size: 0.15`, and `test_size: 0.15`, giving an intended 70/15/15 allocation. The split functions save zero-based row positions to `data/splits/classification_indices.json` and `data/splits/regression_indices.json`.

| Task | Filter / stratification | Train rows | Validation rows | Test rows | Target summary |
|---|---|---:|---:|---:|---|
| Classification | No ADR filter; stratified on `IsCanceled` | 37,130 | 7,956 | 7,957 | Train: `0` 0.697953, `1` 0.302047; validation: `0` 0.697964, `1` 0.302036; test: `0` 0.697876, `1` 0.302124 |
| Regression | `0 < ADR <= 500`; not stratified; 724 rows filtered before splitting | 36,623 | 7,848 | 7,848 | Train ADR mean/std: 113.0530 / 39.4341; validation: 113.2592 / 39.8299; test: 113.7608 / 39.4394 |

These are row counts from the executed split commands. Clustering does not use this splitting strategy in the current `preprocess` command; see the leakage audit and discrepancy list.

## 9. Preprocessing pipeline

[`src/preprocessing/pipeline.py`](../src/preprocessing/pipeline.py) builds a task-specific pipeline from `get_features()`. Its docstring identifies imputers, encoder categories/frequencies, and scaler statistics as data-learned steps and instructs callers to fit on training data only.

- **Stateless feature engineering:** if source columns are present in a task’s feature set, add `total_nights` from `StaysInWeekendNights` + `StaysInWeekNights`, and `total_guests` from `Adults` + `Children` + `Babies` with missing `Children` treated as zero for this calculation. `HasCompany` indicates whether `Company` is nonmissing, after which raw `Company` is dropped. `HasAgent` indicates whether `Agent` is nonmissing while raw `Agent` remains a categorical feature. The clustering feature set has neither `Company` nor `Agent`, so the corresponding flags are skipped.
- **LeadTime:** when enabled by the default `log_transform_leadtime: true`, apply stateless `np.log1p` before numeric imputation/scaling.
- **Numeric columns:** median `SimpleImputer`, followed by `StandardScaler` when `scale=True` (the CLI uses the default).
- **Low-cardinality categoricals:** `ArrivalDateMonth`, `Meal`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `DepositType`, and `CustomerType` use most-frequent imputation followed by dense `OneHotEncoder(handle_unknown="ignore")`.
- **High-cardinality categoricals:** `Country` and `Agent` use constant imputation with `"Unknown"` followed by dense `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)`. Categorical values are converted to strings while retaining missing values before imputation, which allows mixed numeric-looking IDs and strings to be encoded.
- The pipeline implements `get_feature_names_out()`. No input dataset columns are renamed; transformed outputs include engineered and one-hot-expanded features.

Executed output feature counts and artifacts:

| Task | Transformed train/validation/test shapes | Feature count | NaN check | Saved preprocessor |
|---|---|---:|---|---|
| Classification | 37,130 × 88 / 7,956 × 88 / 7,957 × 88 | 88 | No NaN | `artifacts/preprocessor_classification.joblib` |
| Regression | 36,623 × 87 / 7,848 × 87 / 7,848 × 87 | 87 | No NaN | `artifacts/preprocessor_regression.joblib` |
| Clustering | 53,043 × 31 / 53,043 × 31 / 53,043 × 31 | 31 | No NaN | `artifacts/preprocessor_clustering.joblib` |

For clustering, the three shapes describe repeated transforms of the same full cleaned dataset, not separate partitions.

## 10. Verification

The executed `pytest` run collected **14 tests** and passed all of them:

| Test file | Tests | Coverage read from test definitions |
|---|---:|---|
| `tests/test_cleaning.py` | 5 | Duplicate removal, zero-guest removal, zero-night removal, `"NULL"` conversion while retaining missing `Children`, and regression ADR filtering. |
| `tests/test_eda.py` | 1 | Checks that EDA creates its summary and expected PNG files; skips when `data/H2.csv` is absent. |
| `tests/test_loader.py` | 1 | Checks the loader’s expected shape; skips when `data/H2.csv` is absent. |
| `tests/test_pipeline.py` | 3 | Checks unseen validation categories and NaN-free equal-width outputs, train-only numeric median imputation, and engineered totals/flags. |
| `tests/test_splits.py` | 4 | Checks split disjointness/sizes, class proportions, task leakage exclusions, regression ADR filtering, repeatability with a fixed seed, and configurable classification ADR inclusion. |
| **Total** | **14** | Full suite passed; one seaborn pending-deprecation warning. |

## 11. Known limitations and plan for Milestone 2

No model training, baseline evaluation, clustering algorithm, inference implementation, dimensionality reduction, or hyperparameter tuning is present in the inspected project source. The cleaning rules are deterministic but rely on exact-row equality and the selected zero-guest/zero-night rules. The regression-only ADR filter changes the population used for regression and must be stated when comparing tasks. `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`, and optionally `ADR` for classification are retained despite being documented as borderline. Clustering currently includes `ADR`, does not apply the regression ADR filter, and fits its preprocessing on all cleaned rows. The next milestone should define and implement task models and evaluation while preserving the booking-time feature boundary, using train-only fitting, and reserving the test set for final evaluation.

### Discrepancy list

- **`LeadTime` log transform:** implemented as `np.log1p`, enabled by default in config. The test suite does not assert the transformed `LeadTime` values or the disabled-config behavior.
- **`HasCompany` / `HasAgent`:** both are implemented when their source columns are in the task features; raw `Company` is dropped and raw `Agent` is retained. The pipeline test checks both flags.
- **Rare-category grouping at 1%:** implemented for high-cardinality `Country` and `Agent` only. Low-cardinality categoricals do not use `min_frequency`; tests exercise an unseen category but do not verify the 1% grouping threshold.
- **Seed 42:** present in config and used by the supervised split functions. Tests use seed 42 and check repeatability, but do not verify alternate configured seeds. Clustering does not perform a split or use the seed.
- **70/15/15 split:** implemented for classification and regression using configured validation/test sizes. Regression first filters by target `ADR`. Clustering preprocessing instead copies the full cleaned data into train, validation, and test roles and fits on the full data.
- **Regression ADR threshold:** the configured value is 500 and regression filtering removed 724 cleaned rows. The general cleaning command does not apply the ADR filter.
- **Test-set isolation:** supervised preprocessing fits on train and transforms validation/test, and split docstrings prohibit test-set tuning. No model selection/tuning code currently enforces that policy; clustering preprocessing fits on all cleaned rows.
- **Duplicate handling:** exact duplicates are removed before splitting. The code and tests do not establish that every exact duplicate is an accidental duplicate booking rather than a distinct booking with identical recorded values.
- **Future procedures:** dimensionality reduction and tuning are not implemented or tested in this milestone.
