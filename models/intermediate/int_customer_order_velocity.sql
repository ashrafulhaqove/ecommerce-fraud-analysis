with transactions as (
    select *
    from {{ ref('stg_transactions') }}
),

daily_counts as (
    select
        customer_id,
        order_date,
        count(*) as orders_on_day
    from transactions
    group by customer_id, order_date
)

select
    t.order_id,
    t.customer_id,
    t.order_date,
    dc.orders_on_day,
    dc.orders_on_day >= 3 as high_velocity
from transactions t
join daily_counts dc
    on  t.customer_id = dc.customer_id
    and t.order_date  = dc.order_date