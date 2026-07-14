-- One row per product, summarizing how often it gets reordered across
-- all users. A high-level "popularity/stickiness" signal for the model.

select
    product_id,
    count(*) as product_total_orders,
    avg(reordered) as product_reorder_rate
from {{ ref('int_order_products_unioned') }}
where eval_set = 'prior'
group by product_id
