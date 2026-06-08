with source_data as (
    select *
    from read_csv_auto('data/raw/transactions.csv')
)

select
    order_id,
    customer_id,
    cast(order_date as date)     as order_date,
    cast(order_hour as integer)  as order_hour,
    cast(order_amount as double) as order_amount,
    payment_method,
    product_category,
    billing_country,
    shipping_country,
    ip_country,
    cast(is_new_customer as boolean) as is_new_customer,
    cast(account_age_days as integer) as account_age_days,
    billing_country != shipping_country          as country_mismatch,
    ip_country != billing_country                as ip_mismatch
from source_data