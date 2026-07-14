select
    department_id,
    department
from {{ source('raw', 'departments') }}
