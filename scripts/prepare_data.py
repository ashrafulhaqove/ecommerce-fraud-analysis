import pandas as pd
from datetime import datetime, timedelta

# IEEE-CIS transactions start from this reference point
REFERENCE_DATE = datetime(2017, 11, 30)

PRODUCT_MAP = {
    'W': 'Electronics',
    'C': 'Clothing',
    'R': 'Travel',
    'H': 'Home & Garden',
    'S': 'Services',
}

# addr2 is a numeric country code; 87 = US dominates (~88% of data)
ADDR2_COUNTRY_MAP = {
    87.0: 'US', 60.0: 'MX', 96.0: 'CA', 32.0: 'GB',
    65.0: 'AU', 16.0: 'DE', 31.0: 'FR', 19.0: 'JP',
    26.0: 'BR', 27.0: 'IN',
}

US_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'comcast.net',
    'icloud.com', 'outlook.com', 'msn.com', 'att.net', 'live.com',
    'sbcglobal.net', 'verizon.net', 'ymail.com', 'bellsouth.net',
    'cox.net', 'charter.net', 'earthlink.net', 'anonymous.com',
}

TLD_MAP = {
    '.co.uk': 'GB', '.uk': 'GB', '.de': 'DE', '.fr': 'FR',
    '.ru': 'RU', '.cn': 'CN', '.jp': 'JP', '.au': 'AU',
    '.ca': 'CA', '.br': 'BR', '.mx': 'MX', '.in': 'IN',
}


def email_to_country(domain):
    if pd.isna(domain) or domain in US_DOMAINS:
        return 'US'
    for tld, country in TLD_MAP.items():
        if domain.endswith(tld):
            return country
    return 'US'


def main():
    print("Loading train_transaction.csv ...")
    txn = pd.read_csv('data/raw/train_transaction.csv')
    print(f"  {len(txn):,} rows loaded")

    df = pd.DataFrame()

    df['order_id']         = txn['TransactionID'].astype(str)
    df['customer_id']      = 'C' + txn['card1'].fillna(0).astype(int).astype(str)
    df['order_amount']     = txn['TransactionAmt'].round(2)
    df['payment_method']   = txn['card4'].fillna('visa')
    df['product_category'] = txn['ProductCD'].map(PRODUCT_MAP).fillna('Electronics')

    df['order_date'] = txn['TransactionDT'].apply(
        lambda x: (REFERENCE_DATE + timedelta(seconds=int(x))).strftime('%Y-%m-%d')
    )
    df['order_hour'] = txn['TransactionDT'].apply(
        lambda x: (int(x) % 86400) // 3600
    )

    # billing country from purchaser address; shipping inferred from recipient email
    df['billing_country']  = txn['addr2'].map(ADDR2_COUNTRY_MAP).fillna('US')
    df['shipping_country'] = txn['R_emaildomain'].apply(email_to_country)
    df['ip_country']       = txn['P_emaildomain'].apply(email_to_country)

    # account_age_days: D1 = days since last transaction (good proxy)
    median_d1 = txn['D1'].median()
    df['account_age_days'] = txn['D1'].fillna(median_d1).round(0).astype(int)

    # is_new_customer: card seen only once in full dataset
    card_counts = txn['card1'].value_counts()
    df['is_new_customer'] = txn['card1'].map(card_counts).apply(lambda x: 1 if x == 1 else 0)

    # keep fraud label — dbt ignores it, ML scripts use it
    df['is_fraud'] = txn['isFraud']

    df.to_csv('data/raw/transactions.csv', index=False)

    print(f"\nSaved data/raw/transactions.csv")
    print(f"  Rows           : {len(df):,}")
    print(f"  Fraud rate     : {df['is_fraud'].mean():.2%}")
    print(f"  Billing country: {df['billing_country'].value_counts().head(4).to_dict()}")
    print(f"  Products       : {df['product_category'].value_counts().to_dict()}")
    print(f"  New customers  : {df['is_new_customer'].mean():.2%}")


if __name__ == '__main__':
    main()