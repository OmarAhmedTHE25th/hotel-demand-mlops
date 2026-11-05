# CSE382 University ML Project — City Hotel (H2)

## 1. Project Overview & Current Scope
This project develops a machine learning operations pipeline to analyze and predict booking demand, cancellation, and customer segments for the City Hotel (H2) dataset.

- **Current State (Milestone 1 Implemented):**
  - **Data Cleaning & EDA:** Automated duplicate removal, guest/night validation, and comprehensive visual analysis.
  - **Task A (Regression):** Linear Regression baseline for ADR prediction.
  - **Task B (Classification):** Logistic Regression, Random Forest, and XGBoost baselines for cancellation prediction.
  - **Task C (Clustering):** K-Means baseline (k=5) for unsupervised customer segmentation.
- **Upcoming Work (Milestone 2 Planned):**
  - **Hyperparameter Tuning:** Automated search via Optuna for all supervised models.
  - **Feature Engineering:** PCA/LDA feature comparisons and advanced transformation experiments.
  - **Final Evaluation:** Comprehensive scoring on the held-out test set.
  - **Deployment:** Modular deployment interface and final application integration.

> *For full methodology, feature rationales, in-depth leakage audit, and detailed metric discussions, see [Milestone 1 Report](docs/milestone1_report.md).*

## 2. Team Members & Roles
| Member Name                | Student ID | GitHub Account              | Assigned Task / Component |
|----------------------------|------------|-----------------------------|---------------------------|
| Omar Ahmed Abdel Hameed    | 24P0201    | OmarAhmedTHE25th            | Task B (Classification)   |
| Omar Mohamed Salaheldin    | 24P0466    | subzeroomar                 | Task A (Regression)       |
| Mohamed Amr Nady           | 24P0477    | mohamedamrnady              | Task C (Clustering)       |
| Ahmed Mohamed Abd-Elmageed | 2400136    | ahmedmohamedioskareem-cloud | EDA & Preprocessing       |
| Shahd Ali Babiker Hassan   | 23P0436    | bvsil17                     | Inference & CLI Engine    |

## 3. Repository & Submission
- **GitHub repository URL:** https://github.com/OmarAhmedTHE25th/hotel-demand-mlops
- **Submission commit SHA:** `426dd830ee8c28893937c901bb1bc04b8492f61e`
- **Verification command:** `git rev-parse HEAD`

## 4. Environment Setup & Data
### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Placement
1. Download the dataset: [H2.csv](https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv)
2. Place it at: `data/H2.csv`

## 5. Execution & CLI Commands
### Data Prep & EDA
```powershell
python main.py analyze
python main.py eda
python main.py clean
```

### Task A (Regression)
```powershell
python main.py train --task regression
```
*Outputs to `outputs/regression/`*

### Task B (Classification)
```powershell
python main.py split --task classification
python main.py preprocess --task classification
python main.py train --task classification
```
*Outputs to `outputs/classification/`*

### Task C (Clustering)
```powershell
python main.py cluster --algorithm kmeans --clusters 5
```
*Outputs to `outputs/clustering/`*

### Testing
```powershell
pytest -q
```

## 6. Baseline Summary Results
### Task A (Regression): ADR Prediction
| Model             | Val MAE | Val MSE | Val RMSE | Val R² |
|-------------------|---------|---------|----------|--------|
| Dummy (Mean)      | 30.65   | 1586.26 | 39.83    | -0.00  |
| Linear Regression | 15.03   | 447.81  | 21.16    | 0.72   |

### Task B (Classification): Cancellation Prediction
| Model               | Accuracy | Precision | Recall | F1     | ROC AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.7824   | 0.6741    | 0.5414 | 0.6005 | 0.8415  |
| Random Forest       | 0.8159   | 0.7288    | 0.6217 | 0.6710 | 0.8854  |
| XGBoost             | 0.8213   | 0.7285    | 0.6509 | 0.6875 | 0.8892  |

### Task C (Clustering): Customer Segmentation (k=5)
| Metric               | Value   |
|----------------------|---------|
| Silhouette Score     | 0.2157  |
| Davies-Bouldin Index | 1.3600  |
| Calinski-Harabasz    | 8765.79 |

| Cluster | Share | Top Segment | Interpretation                      |
|---------|-------|-------------|-------------------------------------|
| 0       | 14.3% | Online TA   | Longer-stay couples/small groups    |
| 1       | 58.4% | Online TA   | Dominant short-stay couple segment  |
| 2       | 8.5%  | Online TA   | Families with children; highest ADR |
| 3       | 15.9% | Online TA   | Solo short-stay travellers          |
| 4       | 2.9%  | Corporate   | Repeat Corporate/Direct bookers     |

## 7. Individual Contributions
- **Omar Mohamed Salaheldin:** Regression training module, CLI integration, split manifest validation, and leakage audit for Task A.
- **Omar Ahmed Abdel Hameed:** Classification models (LR, RF, XGB) implementation, evaluation suite, and pipeline integration for Task B.
- **Mohamed Amr Nady:** Clustering feature selection, K-Means baseline, profile analysis, and visualization for Task C.
- **Ahmed Mohamed Abd-Elmageed:** Data loader, deterministic cleaning rules, shared preprocessing pipeline, and exploratory data analysis.
