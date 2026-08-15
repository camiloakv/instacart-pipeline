from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="Must be a positive integer")
    product_id: int = Field(..., gt=0, description="Must be a positive integer")


class PredictResponse(BaseModel):
    user_id: int
    product_id: int
    reorder_probability: float


class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list]
    row_count: int
