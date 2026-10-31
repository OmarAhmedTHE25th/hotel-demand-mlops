# CSE382 University ML Project — City Hotel (H2)

## Team members and GitHub accounts

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

The project uses a unified CLI in `main.py`.

### Data and EDA

Analyze the raw dataset summary:
```bash
python main.py analyze
```

Run exploratory data analysis (saves plots to `outputs/`):
```bash
python main.py eda
```

Apply deterministic cleaning rules (saves `data/H2_clean.csv`):
```bash
python main.py clean
```

### Classification Workflow

Create reproducible train/validation/test splits:
```bash
python main.py split --task classification
```

Fit and save the preprocessing pipeline:
```bash
python main.py preprocess --task classification
```

Train baseline models (Logistic Regression, Random Forest, XGBoost):
```bash
python main.py train --task classification
```

Tune models using Optuna:
```bash
python main.py tune --task classification
```

Evaluate final models on the test set:
```bash
python main.py evaluate --task classification
```

### Testing

Run all project tests:
```bash
pytest
```
