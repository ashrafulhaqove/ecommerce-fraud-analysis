# Ecommerce Fraud Analysis

Detects suspicious ecommerce transactions using a fully local, reproducible data pipeline built with Python, dbt Core, and DuckDB. Results are published automatically to GitHub Pages on every push.

## Architecture

```mermaid
flowchart LR
    A[Python\nData Generator] -->|transactions.csv| B[dbt Staging\nstg_transactions]
    B --> C[dbt Intermediate\norder velocity]
    B --> D[dbt Intermediate\ncustomer profile]
    C --> E[dbt Mart\nfct_fraud_signals]
    D --> E
    E -->|DuckDB| ML[XGBoost\nML Pipeline]
    ML -->|ml_predictions| F[Python\nReport Generator]
    E -->|DuckDB| F
    F -->|index.html| G[GitHub Pages]

    style A fill:#4299e1,color:#fff,stroke:none
    style B fill:#68d391,color:#fff,stroke:none
    style C fill:#68d391,color:#fff,stroke:none
    style D fill:#68d391,color:#fff,stroke:none
    style E fill:#f6ad55,color:#fff,stroke:none
    style ML fill:#9f7aea,color:#fff,stroke:none
    style F fill:#4299e1,color:#fff,stroke:none
    style G fill:#fc8181,color:#fff,stroke:none
```

## Fraud Signals

Each order is scored across 8 rules. Orders scoring 50+ are flagged.

| Rule | Points |
|---|---|
| IP country ≠ billing country | +25 |
| Billing country ≠ shipping country | +25 |
| New account (age < 30 days) | +30 |
| Order amount ≥ $150 | +20 |
| New customer + order ≥ $100 | +20 |
| High velocity (3+ orders/day) | +20 |
| Travel category purchase | +15 |
| Late-night order (1am–5am) | +10 |

**Risk tiers:** high (≥80) · medium (50–79) · low (25–49) · none (<25)

## Stack

| Tool | Role |
|---|---|
| Python + pandas | Data preparation, report rendering |
| dbt Core | Data transformation, testing, lineage |
| DuckDB | Local analytical database (no server needed) |
| XGBoost + scikit-learn | ML fraud detection model |
| Jinja2 | HTML report templating |
| GitHub Actions | CI — runs full pipeline on every push |
| GitHub Pages | Hosts the fraud report |

## Project Layout

```
├── scripts/
│   ├── generate_transactions.py   # synthetic data generator
│   ├── train_model.py             # XGBoost training (run locally on real data)
│   ├── predict.py                 # loads committed model, writes ml_predictions to DuckDB
│   └── generate_report.py         # HTML report from DuckDB
├── models/
│   ├── staging/                   # type casting, raw flag derivation
│   ├── intermediate/              # velocity + customer risk profile
│   └── marts/                     # final scored fraud signals table
├── data/                          # raw CSV and DuckDB file (git-ignored)
├── reports/                       # generated HTML (git-ignored, deployed by CI)
├── profiles.yml                   # dbt DuckDB connection
└── .github/workflows/pipeline.yml # CI/CD
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_transactions.py
dbt build --profiles-dir .
python scripts/predict.py        # loads committed model, writes predictions
python scripts/generate_report.py
# open reports/index.html in a browser
```

To retrain from scratch on new data, run `python scripts/train_model.py` instead of `predict.py`.

> **Note on model storage:** The trained model (`models/xgboost_fraud.pkl`) is committed directly — it is 832 KB, well within Git's limits. For larger models (>100 MB) the standard approach is [DVC](https://dvc.org/) with a remote backend (S3, Google Drive, etc.), which stores only a small pointer file in Git and the actual artifact in object storage.