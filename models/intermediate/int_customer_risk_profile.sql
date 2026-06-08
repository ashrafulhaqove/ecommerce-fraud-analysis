with transactions as (
    select *
    from {{ ref('stg_transactions') }}
)

select
    customer_id,
    count(*)                            as total_orders,
    round(avg(order_amount), 2)         as avg_order_amount,
    round(max(order_amount), 2)         as max_order_amount,
    min(account_age_days)               as account_age_days,
    bool_or(is_new_customer)            as is_new_customer,
    count(distinct product_category)    as distinct_categories,
    count(distinct payment_method)      as distinct_payment_methods,
    sum(case when ip_mismatch then 1 else 0 end) as ip_mismatch_count
from transactions
group by customer_id