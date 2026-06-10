from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DB_PATH    = Path('data/ecommerce_fraud.duckdb')
MODEL_PATH = Path('models/xgboost_fraud.pkl')

FEATURES     = [
    'order_amount', 'order_hour', 'account_age_days', 'orders_on_day',
    'customer_avg_order_amount', 'customer_ip_mismatch_count',
    'is_new_customer', 'country_mismatch', 'ip_mismatch', 'high_velocity',
    'payment_method', 'product_category',
]
BOOL_COLS    = ['is_new_customer', 'country_mismatch', 'ip_mismatch', 'high_velocity']
CAT_COLS     = ['payment_method', 'product_category']


def load_features(con):
    df = con.execute('SELECT * FROM fct_fraud_signals').df()
    X  = pd.get_dummies(df[FEATURES], columns=CAT_COLS, drop_first=True)
    for c in BOOL_COLS:
        X[c] = X[c].astype(int)
    y = df['is_fraud']
    return df['order_id'], X, y


def train(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train.values, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    print(f"  ROC-AUC          : {auc:.3f}")
    print(f"  Avg precision    : {ap:.3f}")
    print(f"  Fraud F1         : {f1:.3f}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=['Legit', 'Fraud'], digits=3))
    return auc, ap, f1


def write_predictions(con, order_ids, X, model):
    probs = model.predict_proba(X.values)[:, 1]
    flags = (probs >= 0.5).astype(int)
    preds = pd.DataFrame({
        'order_id':      order_ids.values,
        'ml_fraud_prob': probs.round(4),
        'ml_fraud_flag': flags,
    })
    con.execute('CREATE OR REPLACE TABLE ml_predictions AS SELECT * FROM preds')
    print(f"  ml_predictions written: {len(preds):,} rows  "
          f"({flags.sum():,} flagged, {flags.mean():.2%} rate)")


def write_metrics(con, auc, ap, f1):
    metrics = pd.DataFrame([
        {'metric': 'roc_auc',       'value': round(auc, 4)},
        {'metric': 'avg_precision', 'value': round(ap,  4)},
        {'metric': 'fraud_f1',      'value': round(f1,  4)},
        {'metric': 'baseline_auc',  'value': 0.631},
        {'metric': 'baseline_ap',   'value': 0.052},
        {'metric': 'baseline_f1',   'value': 0.093},
    ])
    con.execute('CREATE OR REPLACE TABLE ml_metrics AS SELECT * FROM metrics')
    print("  ml_metrics written")


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)

    print("Loading data ...")
    con = duckdb.connect(str(DB_PATH))
    order_ids, X, y = load_features(con)
    print(f"  {len(X):,} rows, {X.shape[1]} features, fraud rate {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    print("\nTraining XGBoost ...")
    model = train(X_train, y_train)

    print("\nEvaluating on test set ...")
    evaluate_results = evaluate(model, X_test, y_test)

    print("Saving model ...")
    joblib.dump(model, MODEL_PATH)
    print(f"  Saved → {MODEL_PATH}")

    print("\nWriting predictions to DuckDB ...")
    write_predictions(con, order_ids, X, model)
    write_metrics(con, *evaluate_results)

    con.close()
    print("\nDone.")


if __name__ == '__main__':
    main()