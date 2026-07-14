select
    aisle_id,
    aisle
from {{ source('raw', 'aisles') }}
