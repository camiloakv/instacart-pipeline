-- One row per (user, product) pair that appears in a user's PRIOR order
-- history. Deliberately excludes the 'train' split, since that's reserved
-- as the prediction target in the marts layer -- using it here would leak
-- the answer into the features.

select
    user_id,
    product_id,
    count(*) as up_times_ordered,
    sum(reordered) as up_times_reordered,
    avg(add_to_cart_order) as up_avg_add_to_cart_order,
    min(order_number) as up_first_order_number,
    max(order_number) as up_last_order_number
from {{ ref('int_order_products_unioned') }}
where eval_set = 'prior'
group by user_id, product_id
