import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd

DB_PATH      = Path('data/ecommerce_fraud.duckdb')
MODEL_PATH   = Path('models/xgboost_fraud.pkl')
METRICS_PATH = Path('models/metrics.json')

FEATURES  = [
    'order_amount', 'order_hour', 'account_age_days', 'orders_on_day',
    'customer_avg_order_amount', 'customer_ip_mismatch_count',
    'is_new_customer', 'country_mismatch', 'ip_mismatch', 'high_velocity',
    'payment_method', 'product_category',
]
BOOL_COLS = ['is_new_customer', 'country_mismatch', 'ip_mismatch', 'high_velocity']
CAT_COLS  = ['payment_method', 'product_category']


def main():
    for path in (MODEL_PATH, METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run scripts/train_model.py first")

    print(f"Loading model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    print("Loading features from DuckDB ...")
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute('SELECT * FROM fct_fraud_signals').df()
    X   = pd.get_dummies(df[FEATURES], columns=CAT_COLS, drop_first=True)
    for c in BOOL_COLS:
        X[c] = X[c].astype(int)
    print(f"  {len(X):,} rows")

    print("Running predictions ...")
    probs = model.predict_proba(X.values)[:, 1]
    flags = (probs >= 0.5).astype(int)

    preds = pd.DataFrame({
        'order_id':      df['order_id'].values,
        'ml_fraud_prob': probs.round(4),
        'ml_fraud_flag': flags,
    })
    con.execute('CREATE OR REPLACE TABLE ml_predictions AS SELECT * FROM preds')
    print(f"  ml_predictions written: {len(preds):,} rows ({flags.sum():,} flagged)")

    # use committed metrics from training on real data, not synthetic re-evaluation
    m = json.loads(METRICS_PATH.read_text())
    print(f"  Metrics (from training): ROC-AUC {m['roc_auc']}  AP {m['avg_precision']}  F1 {m['fraud_f1']}")

    metrics_df = pd.DataFrame([
        {'metric': 'roc_auc',       'value': m['roc_auc']},
        {'metric': 'avg_precision', 'value': m['avg_precision']},
        {'metric': 'fraud_f1',      'value': m['fraud_f1']},
        {'metric': 'baseline_auc',  'value': m['baseline_auc']},
        {'metric': 'baseline_ap',   'value': m['baseline_ap']},
        {'metric': 'baseline_f1',   'value': m['baseline_f1']},
    ])
    con.execute('CREATE OR REPLACE TABLE ml_metrics AS SELECT * FROM metrics_df')
    print("  ml_metrics written")

    con.close()
    print("\nDone.")


if __name__ == '__main__':
    main()