from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    row_count = 500

    customer_ids = rng.integers(1000, 1040, size=row_count)
    order_amounts = np.round(rng.gamma(shape=2.0, scale=55.0, size=row_count), 2)
    order_dates = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2024-01-01").value,
            pd.Timestamp("2024-12-31").value,
            size=row_count,
        )
    ).normalize()
    hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01,  # 0–4  (late night, low volume)
        0.02, 0.03, 0.05, 0.07, 0.07,  # 5–9
        0.07, 0.07, 0.06, 0.06, 0.06,  # 10–14
        0.06, 0.06, 0.06, 0.05, 0.04,  # 15–19
        0.04, 0.03, 0.02, 0.01,         # 20–23
    ], dtype=float)
    hour_weights /= hour_weights.sum()
    order_hours = rng.choice(range(24), size=row_count, p=hour_weights)

    billing_countries = rng.choice(
        ["US", "CA", "GB", "AU", "DE"],
        size=row_count,
        p=[0.60, 0.15, 0.10, 0.08, 0.07],
    )
    shipping_countries = rng.choice(
        ["US", "CA", "GB", "AU", "DE"],
        size=row_count,
        p=[0.65, 0.13, 0.09, 0.07, 0.06],
    )
    ip_countries = rng.choice(
        ["US", "CA", "GB", "AU", "DE", "NG", "RO", "BR"],
        size=row_count,
        p=[0.55, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.05],
    )

    payment_methods = rng.choice(
        ["visa", "mastercard", "discover", "american express"],
        size=row_count,
        p=[0.45, 0.35, 0.12, 0.08],
    )
    product_categories = rng.choice(
        ["Electronics", "Clothing", "Travel", "Home & Garden", "Services"],
        size=row_count,
        p=[0.20, 0.35, 0.15, 0.20, 0.10],
    )

    is_new_customer = rng.random(size=row_count) < 0.20
    account_age_days = np.where(
        is_new_customer,
        rng.integers(1, 30, size=row_count),
        rng.integers(30, 1825, size=row_count),
    )

    # synthetic fraud label (~5% rate) for dbt pipeline compatibility
    is_fraud = (rng.random(size=row_count) < 0.05).astype(int)

    data = pd.DataFrame(
        {
            "order_id": np.arange(1, row_count + 1),
            "customer_id": customer_ids,
            "order_date": order_dates.strftime("%Y-%m-%d"),
            "order_hour": order_hours,
            "order_amount": order_amounts,
            "payment_method": payment_methods,
            "product_category": product_categories,
            "billing_country": billing_countries,
            "shipping_country": shipping_countries,
            "ip_country": ip_countries,
            "is_new_customer": is_new_customer,
            "account_age_days": account_age_days,
            "is_fraud": is_fraud,
        }
    )

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "transactions.csv", index=False)
    print(f"Generated {row_count} transactions → data/raw/transactions.csv")


if __name__ == "__main__":
    main()