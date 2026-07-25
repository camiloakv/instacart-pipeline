from pydantic import BaseModel


class PredictRequest(BaseModel):
    user_id: int
    product_id: int


class PredictResponse(BaseModel):
    user_id: int
    product_id: int
    reorder_probability: float
