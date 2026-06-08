"""
Stress test for the fraud pipeline.

Runs three checks independently of dbt (pure DuckDB + Python):
  1. Scale      — pipeline performance at 1k / 10k / 50k rows
  2. Coverage   — every fraud rule fires at least once
  3. Boundaries — scoring thresholds behave correctly at edge values
"""

import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data generation (mirrors generate_transactions.py but parameterised)
# ---------------------------------------------------------------------------

def make_transactions(row_count: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    order_dates = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2024-01-01").value,
            pd.Timestamp("2024-12-31").value,
            size=row_count,
        )
    ).normalize()

    hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01,
        0.02, 0.03, 0.05, 0.07, 0.07,
        0.07, 0.07, 0.06, 0.06, 0.06,
        0.06, 0.06, 0.06, 0.05, 0.04,
        0.04, 0.03, 0.02, 0.01,
    ], dtype=float)
    hour_weights /= hour_weights.sum()

    billing = rng.choice(["US", "CA", "GB", "AU", "DE"], size=row_count, p=[0.60, 0.15, 0.10, 0.08, 0.07])
    shipping = rng.choice(["US", "CA", "GB", "AU", "DE"], size=row_count, p=[0.65, 0.13, 0.09, 0.07, 0.06])
    ip = rng.choice(["US", "CA", "GB", "AU", "DE", "NG", "RO", "BR"], size=row_count,
                    p=[0.55, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.05])
    is_new = rng.random(size=row_count) < 0.20

    return pd.DataFrame({
        "order_id":        np.arange(1, row_count + 1),
        "customer_id":     rng.integers(1000, max(1002, row_count // 10 + 1000), size=row_count),
        "order_date":      order_dates.strftime("%Y-%m-%d"),
        "order_hour":      rng.choice(range(24), size=row_count, p=hour_weights).astype(int),
        "order_amount":    np.round(rng.gamma(shape=2.0, scale=55.0, size=row_count), 2),
        "payment_method":  rng.choice(["credit_card", "debit_card", "paypal", "crypto"], size=row_count, p=[0.50, 0.25, 0.20, 0.05]),
        "product_category":rng.choice(["electronics", "clothing", "gift_cards", "luxury", "home"], size=row_count, p=[0.20, 0.35, 0.15, 0.10, 0.20]),
        "billing_country": billing,
        "shipping_country":shipping,
        "ip_country":      ip,
        "is_new_customer": is_new,
        "account_age_days":np.where(is_new, rng.integers(1, 30, size=row_count), rng.integers(30, 1825, size=row_count)),
    })


# ---------------------------------------------------------------------------
# Pipeline SQL (mirrors dbt models as a single CTE chain)
# ---------------------------------------------------------------------------

PIPELINE_SQL = """
with stg as (
    select
        order_id, customer_id,
        cast(order_date as date)      as order_date,
        cast(order_hour as integer)   as order_hour,
        cast(order_amount as double)  as order_amount,
        payment_method, product_category,
        billing_country, shipping_country, ip_country,
        cast(is_new_customer as boolean)  as is_new_customer,
        cast(account_age_days as integer) as account_age_days,
        billing_country != shipping_country as country_mismatch,
        ip_country != billing_country       as ip_mismatch
    from transactions
),

daily_counts as (
    select customer_id, order_date, count(*) as orders_on_day
    from stg
    group by customer_id, order_date
),

velocity as (
    select s.order_id, dc.orders_on_day, dc.orders_on_day >= 3 as high_velocity
    from stg s
    join daily_counts dc on s.customer_id = dc.customer_id and s.order_date = dc.order_date
),

customer_profile as (
    select
        customer_id,
        count(*)                    as total_orders,
        avg(order_amount)           as avg_order_amount,
        min(account_age_days)       as account_age_days,
        bool_or(is_new_customer)    as is_new_customer,
        sum(case when ip_mismatch then 1 else 0 end) as ip_mismatch_count
    from stg
    group by customer_id
),

scored as (
    select
        s.order_id, s.customer_id, s.order_amount, s.payment_method,
        s.product_category, s.is_new_customer, s.order_hour,
        s.country_mismatch, s.ip_mismatch, v.high_velocity,
        (
            case when s.order_amount >= 150                                then 20 else 0 end
          + case when s.country_mismatch                                   then 25 else 0 end
          + case when v.high_velocity                                      then 20 else 0 end
          + case when s.product_category = 'gift_cards'                   then 15 else 0 end
          + case when s.is_new_customer and s.order_amount >= 100         then 20 else 0 end
          + case when s.order_hour between 1 and 5                        then 10 else 0 end
          + case when s.ip_mismatch                                        then 25 else 0 end
          + case when s.is_new_customer and s.payment_method = 'crypto'   then 30 else 0 end
        ) as risk_score
    from stg s
    join velocity        v  on s.order_id    = v.order_id
    join customer_profile cp on s.customer_id = cp.customer_id
)

select *, case when risk_score >= 50 then 1 else 0 end as fraud_flag
from scored
"""


def run_pipeline(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    con.register("transactions", df)
    return con.execute(PIPELINE_SQL).df()


# ---------------------------------------------------------------------------
# 1. Scale test
# ---------------------------------------------------------------------------

def test_scale() -> None:
    print("\n── Scale test ─────────────────────────────────────")
    print(f"{'Rows':>8}  {'Build time':>12}  {'Flagged':>8}  {'Fraud %':>8}")
    print("-" * 46)

    for n in [1_000, 10_000, 50_000]:
        df = make_transactions(n)
        con = duckdb.connect(":memory:")

        t0 = time.perf_counter()
        result = run_pipeline(con, df)
        elapsed = time.perf_counter() - t0

        flagged = result["fraud_flag"].sum()
        rate = flagged / n * 100
        print(f"{n:>8,}  {elapsed:>10.3f}s  {flagged:>8,}  {rate:>7.1f}%")
        con.close()


# ---------------------------------------------------------------------------
# 2. Rule coverage test
# ---------------------------------------------------------------------------

RULES: list[dict[str, Any]] = [
    {"name": "High order amount (>=150)",        "check": lambda r: (r["order_amount"] >= 150).any()},
    {"name": "Country mismatch",                 "check": lambda r: r["country_mismatch"].any()},
    {"name": "IP mismatch",                      "check": lambda r: r["ip_mismatch"].any()},
    {"name": "High velocity (3+ orders/day)",    "check": lambda r: r["high_velocity"].any()},
    {"name": "Gift card purchase",               "check": lambda r: (r["product_category"] == "gift_cards").any()},
    {"name": "New customer + high amount",       "check": lambda r: (r["is_new_customer"] & (r["order_amount"] >= 100)).any()},
    {"name": "New customer + crypto",            "check": lambda r: (r["is_new_customer"] & (r["payment_method"] == "crypto")).any()},
    {"name": "Late-night order (1am-5am)",       "check": lambda r: r["order_hour"].between(1, 5).any()},
]


def test_coverage() -> None:
    print("\n── Rule coverage (10k rows) ───────────────────────")
    df = make_transactions(10_000)
    con = duckdb.connect(":memory:")
    result = run_pipeline(con, df)
    con.close()

    all_pass = True
    for rule in RULES:
        fires = rule["check"](result)
        status = "PASS" if fires else "FAIL"
        if not fires:
            all_pass = False
        print(f"  [{status}]  {rule['name']}")

    if all_pass:
        print("\n  All 8 rules fire correctly.")
    else:
        print("\n  WARNING: Some rules never fired — check data generator probabilities.")


# ---------------------------------------------------------------------------
# 3. Boundary test
# ---------------------------------------------------------------------------

def boundary_row(order_id: int, **overrides: Any) -> dict:
    base = {
        "order_id": order_id, "customer_id": 9000 + order_id,  # unique per row — avoids cross-row velocity
        "order_date": "2024-06-01", "order_hour": 12,
        "order_amount": 50.0, "payment_method": "credit_card",
        "product_category": "clothing", "billing_country": "US",
        "shipping_country": "US", "ip_country": "US",
        "is_new_customer": False, "account_age_days": 365,
    }
    base.update(overrides)
    return base


def test_boundaries() -> None:
    print("\n── Boundary test ──────────────────────────────────")

    cases = [
        # (description, row overrides, expected_score, expected_flag)
        ("Clean order — no signals",              {},                                          0,  0),
        ("Amount $149 — just under threshold",    {"order_amount": 149.99},                   0,  0),
        ("Amount $150 — hits threshold",          {"order_amount": 150.00},                  20,  0),
        ("Country mismatch only",                 {"shipping_country": "CA"},                 25,  0),
        ("IP mismatch only",                      {"ip_country": "NG"},                       25,  0),
        ("Country + IP mismatch = flagged",       {"shipping_country": "CA", "ip_country": "NG"}, 50, 1),
        ("New customer + crypto = high risk",     {"is_new_customer": True, "payment_method": "crypto"}, 30, 0),
        ("New + crypto + gift card = flagged",    {"is_new_customer": True, "payment_method": "crypto", "product_category": "gift_cards"}, 45, 0),
        ("New + crypto + gift + high amt",        {"is_new_customer": True, "payment_method": "crypto", "product_category": "gift_cards", "order_amount": 150.0}, 85, 1),
        ("Late night + country mismatch",         {"order_hour": 3, "shipping_country": "GB"}, 35, 0),
    ]

    rows = [boundary_row(i + 1, **overrides) for i, (_, overrides, _, _) in enumerate(cases)]
    df = pd.DataFrame(rows)

    con = duckdb.connect(":memory:")
    result = run_pipeline(con, df)
    con.close()

    print(f"  {'Case':<42}  {'Expected':>10}  {'Got':>6}  {'Status':>6}")
    print("  " + "-" * 70)

    all_pass = True
    for i, (desc, _, exp_score, exp_flag) in enumerate(cases):
        row = result[result["order_id"] == i + 1].iloc[0]
        score_ok = int(row["risk_score"]) == exp_score
        flag_ok  = int(row["fraud_flag"]) == exp_flag
        ok = score_ok and flag_ok
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        got = f"score={int(row['risk_score'])} flag={int(row['fraud_flag'])}"
        exp = f"score={exp_score} flag={exp_flag}"
        print(f"  {desc:<42}  {exp:>10}  {got:>18}  {status:>6}")

    if all_pass:
        print("\n  All boundary cases pass.")
    else:
        print("\n  WARNING: Some boundary cases failed — review scoring logic.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Fraud Pipeline — Stress Test")
    print("=" * 50)
    test_scale()
    test_coverage()
    test_boundaries()
    print("\nDone.\n")