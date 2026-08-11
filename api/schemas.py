from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="Must be a positive integer")
    product_id: int = Field(..., gt=0, description="Must be a positive integer")


class PredictResponse(BaseModel):
    user_id: int
    product_id: int
    reorder_probability: float
