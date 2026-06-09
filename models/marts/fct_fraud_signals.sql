with transactions as (
    select * from {{ ref('stg_transactions') }}
),

velocity as (
    select * from {{ ref('int_customer_order_velocity') }}
),

customer_profile as (
    select * from {{ ref('int_customer_risk_profile') }}
),

scored as (
    select
        t.order_id,
        t.customer_id,
        t.order_date,
        t.order_hour,
        t.order_amount,
        t.payment_method,
        t.product_category,
        t.billing_country,
        t.shipping_country,
        t.ip_country,
        t.is_new_customer,
        t.account_age_days,
        t.country_mismatch,
        t.ip_mismatch,
        v.orders_on_day,
        v.high_velocity,
        cp.avg_order_amount        as customer_avg_order_amount,
        cp.ip_mismatch_count       as customer_ip_mismatch_count,

        -- risk scoring: each rule adds weighted points
        (
            case when t.order_amount >= 150                                     then 20 else 0 end
          + case when t.country_mismatch                                        then 25 else 0 end
          + case when v.high_velocity                                           then 20 else 0 end
          + case when t.product_category = 'Travel'                            then 15 else 0 end
          + case when t.is_new_customer and t.order_amount >= 100              then 20 else 0 end
          + case when t.order_hour between 1 and 5                             then 10 else 0 end
          + case when t.ip_mismatch                                             then 25 else 0 end
          + case when t.account_age_days < 30                                  then 30 else 0 end
        ) as risk_score

    from transactions t
    join velocity        v  on t.order_id    = v.order_id
    join customer_profile cp on t.customer_id = cp.customer_id
)

select
    *,
    case when risk_score >= 50 then 1 else 0 end as fraud_flag,
    case
        when risk_score >= 80 then 'high'
        when risk_score >= 50 then 'medium'
        when risk_score >= 25 then 'low'
        else 'none'
    end as risk_tier
from scored