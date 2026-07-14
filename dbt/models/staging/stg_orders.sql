select
    order_id,
    user_id,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order,
    eval_set
from {{ source('raw', 'orders') }}
