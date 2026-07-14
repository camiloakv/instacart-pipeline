-- One row per (user, product) pair the user has purchased before.
-- Target: did they reorder it in their most recent ('train') order?
-- This is the table XGBoost will train on directly.

with candidate_pairs as (

    -- Every (user, product) pair a user has EVER purchased in their prior
    -- history -- these are the pairs we're predicting reorder for.
    select user_id, product_id
    from {{ ref('int_user_product_stats') }}

),

target as (

    -- The actual reorder outcome from each user's most recent (train) order.
    -- Not every user has a train-split order in this dataset (only a subset
    -- of users were held out for the Kaggle competition's train split), so
    -- this is a left join -- rows without a match get target = 0 (product
    -- not present in that user's most recent order).
    select
        user_id,
        product_id,
        1 as target_reordered
    from {{ ref('int_order_products_unioned') }}
    where eval_set = 'train'

)

select
    cp.user_id,
    cp.product_id,

    -- user-level features
    uos.user_total_orders,
    uos.user_avg_days_between_orders,
    uos.user_avg_order_dow,
    uos.user_avg_order_hour,

    -- product-level features
    ps.product_total_orders,
    ps.product_reorder_rate,

    -- user-product interaction features
    ups.up_times_ordered,
    ups.up_times_reordered,
    ups.up_avg_add_to_cart_order,
    ups.up_first_order_number,
    ups.up_last_order_number,
    (uos.user_max_order_number - ups.up_last_order_number) as up_orders_since_last_purchase,

    -- product dimension attributes (useful for categorical features)
    p.product_name,
    p.aisle_id,
    p.department_id,

    -- prediction target
    coalesce(t.target_reordered, 0) as target_reordered

from candidate_pairs cp
inner join {{ ref('int_user_product_stats') }} ups
    on cp.user_id = ups.user_id and cp.product_id = ups.product_id
inner join {{ ref('int_user_order_stats') }} uos
    on cp.user_id = uos.user_id
inner join {{ ref('int_product_stats') }} ps
    on cp.product_id = ps.product_id
inner join {{ ref('stg_products') }} p
    on cp.product_id = p.product_id
left join target t
    on cp.user_id = t.user_id and cp.product_id = t.product_id
