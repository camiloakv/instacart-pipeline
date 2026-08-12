"""
FastAPI service exposing the trained XGBoost reorder-prediction model.

POST /predict {"user_id": ..., "product_id": ...}
    -> looks up precomputed features for that (user, product) pair from
       the dbt marts table, scores them with the trained model, and
       returns a reorder probability.
"""

import os

import psycopg2
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from schemas import PredictRequest, PredictResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.json")
API_KEY = os.environ["API_KEY"]

api_key_header = APIKeyHeader(name="X-API-Key")
limiter = Limiter(key_func=get_remote_address)

# Must match the feature order/set used in ml/train.py
FEATURE_COLUMNS = [
    "user_total_orders",
    "user_avg_days_between_orders",
    "user_avg_order_dow",
    "user_avg_order_hour",
    "product_total_orders",
    "product_reorder_rate",
    "up_times_ordered",
    "up_times_reordered",
    "up_reorder_ratio",
    "up_avg_add_to_cart_order",
    "up_orders_since_last_purchase",
    "aisle_id",
    "department_id",
]

app = FastAPI(title="Instacart Reorder Prediction API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


def verify_api_key(x_api_key: str = Depends(api_key_header)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def fetch_features(user_id: int, product_id: int) -> dict:
    query = """
        select
            user_total_orders, user_avg_days_between_orders,
            user_avg_order_dow, user_avg_order_hour,
            product_total_orders, product_reorder_rate,
            up_times_ordered, up_times_reordered, up_avg_add_to_cart_order,
            up_orders_since_last_purchase,
            aisle_id, department_id
        from dbt_dev.fct_user_product_features
        where user_id = %s and product_id = %s
        limit 1
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, product_id))
            row = cur.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
    finally:
        conn.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def frontend():
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(static_path)


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def predict(request: Request, body: PredictRequest):
    features = fetch_features(body.user_id, body.product_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No feature data found for user_id={body.user_id}, "
                   f"product_id={body.product_id}. This pair may not "
                   f"exist in the user's purchase history.",
        )

    # Same derived feature as in ml/train.py
    features["up_reorder_ratio"] = (
        float(features["up_times_reordered"]) / float(features["up_times_ordered"])
        if features["up_times_ordered"] else 0.0
    )

    # psycopg2 returns Decimal for numeric aggregates -- cast explicitly
    row = [[float(features[col]) for col in FEATURE_COLUMNS]]

    probability = float(model.predict_proba(row)[0][1])

    return PredictResponse(
        user_id=body.user_id,
        product_id=body.product_id,
        reorder_probability=round(probability, 4),
    )
