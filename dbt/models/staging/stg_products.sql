select
    product_id,
    product_name,
    aisle_id,
    department_id
from {{ source('raw', 'products') }}
