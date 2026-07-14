-- Combines the two order_products splits into one table, and attaches
-- each order's metadata (user_id, order_number, eval_set) so downstream
-- models don't need to keep joining back to stg_orders.

with order_products as (

    select * from {{ ref('stg_order_products_prior') }}
    union all
    select * from {{ ref('stg_order_products_train') }}

),

orders as (

    select * from {{ ref('stg_orders') }}

)

select
    op.order_id,
    op.product_id,
    op.add_to_cart_order,
    op.reordered,
    o.user_id,
    o.order_number,
    o.order_dow,
    o.order_hour_of_day,
    o.days_since_prior_order,
    o.eval_set
from order_products op
inner join orders o on op.order_id = o.order_id
