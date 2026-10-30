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

Analyze the dataset:

```bash
python main.py analyze
```

Run the tests:

```bash
pytest
```
