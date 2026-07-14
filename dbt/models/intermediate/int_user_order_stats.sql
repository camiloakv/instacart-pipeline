-- One row per user, summarizing their overall ordering behavior.
-- These become user-level features for the ML model.

select
    user_id,
    count(distinct order_id) as user_total_orders,
    avg(days_since_prior_order) as user_avg_days_between_orders,
    avg(order_dow) as user_avg_order_dow,
    avg(order_hour_of_day) as user_avg_order_hour,
    max(order_number) as user_max_order_number
from {{ ref('stg_orders') }}
group by user_id
