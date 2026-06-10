from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

DB_PATH    = Path('data/ecommerce_fraud.duckdb')
MODEL_PATH = Path('models/xgboost_fraud.pkl')

FEATURES  = [
    'order_amount', 'order_hour', 'account_age_days', 'orders_on_day',
    'customer_avg_order_amount', 'customer_ip_mismatch_count',
    'is_new_customer', 'country_mismatch', 'ip_mismatch', 'high_velocity',
    'payment_method', 'product_category',
]
BOOL_COLS = ['is_new_customer', 'country_mismatch', 'ip_mismatch', 'high_velocity']
CAT_COLS  = ['payment_method', 'product_category']

BASELINE = {'roc_auc': 0.631, 'avg_precision': 0.052, 'fraud_f1': 0.093}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found — run scripts/train_model.py first"
        )

    print(f"Loading model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    print("Loading features from DuckDB ...")
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute('SELECT * FROM fct_fraud_signals').df()
    X   = pd.get_dummies(df[FEATURES], columns=CAT_COLS, drop_first=True)
    for c in BOOL_COLS:
        X[c] = X[c].astype(int)
    y = df['is_fraud']
    print(f"  {len(X):,} rows, fraud rate {y.mean():.2%}")

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

    auc = roc_auc_score(y, probs)
    ap  = average_precision_score(y, probs)
    f1  = f1_score(y, flags)
    print(f"  ROC-AUC {auc:.3f}  AP {ap:.3f}  F1 {f1:.3f}")

    metrics = pd.DataFrame([
        {'metric': 'roc_auc',       'value': round(auc, 4)},
        {'metric': 'avg_precision', 'value': round(ap,  4)},
        {'metric': 'fraud_f1',      'value': round(f1,  4)},
        {'metric': 'baseline_auc',  'value': BASELINE['roc_auc']},
        {'metric': 'baseline_ap',   'value': BASELINE['avg_precision']},
        {'metric': 'baseline_f1',   'value': BASELINE['fraud_f1']},
    ])
    con.execute('CREATE OR REPLACE TABLE ml_metrics AS SELECT * FROM metrics')
    print("  ml_metrics written")

    con.close()
    print("\nDone.")


if __name__ == '__main__':
    main()