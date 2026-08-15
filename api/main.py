"""
FastAPI service exposing the trained XGBoost reorder-prediction model.

POST /predict {"user_id": ..., "product_id": ...}
    -> looks up precomputed features for that (user, product) pair from
       the dbt marts table, scores them with the trained model, and
       returns a reorder probability.
"""

import os
import re

import psycopg2
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from schemas import PredictRequest, PredictResponse, QueryRequest, QueryResponse

# Layer 2 (application-level) of the query console's defense-in-depth
# design: only these leading keywords are allowed through at all, checked
# BEFORE the query ever reaches Postgres. This is intentionally independent
# of Layer 1 (the db_viewer role's grants) and Layer 3 (the read-only
# transaction wrapper we'll add next) -- each layer assumes the others
# might fail.
ALLOWED_STATEMENT_PREFIXES = ("select", "explain", "show")

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


def get_db_viewer_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user="db_viewer",
        password=os.environ["DB_VIEWER_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def validate_readonly_statement(sql: str) -> None:
    # Strip SQL comments and surrounding whitespace before inspecting the
    # statement -- otherwise "-- comment\nDELETE ..." would slip past a
    # naive prefix check.
    stripped = re.sub(r"--[^\n]*", "", sql)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = stripped.strip()

    if not stripped:
        raise HTTPException(status_code=400, detail="Query is empty.")

    # Reject stacked statements (e.g. "SELECT 1; DELETE FROM orders;") --
    # only a single statement is allowed through. A trailing semicolon on
    # an otherwise single statement is fine.
    without_trailing_semicolon = stripped.rstrip(";").strip()
    if ";" in without_trailing_semicolon:
        raise HTTPException(
            status_code=400,
            detail="Only a single SQL statement is allowed per request.",
        )

    first_word = re.match(r"^([a-zA-Z]+)", without_trailing_semicolon)
    keyword = first_word.group(1).lower() if first_word else ""

    if keyword not in ALLOWED_STATEMENT_PREFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_STATEMENT_PREFIXES)} statements "
                   f"are allowed. This is a read-only query console.",
        )


@app.post("/admin/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def run_query(request: Request, body: QueryRequest):
    validate_readonly_statement(body.sql)

    conn = get_db_viewer_connection()
    try:
        # Layer 3 of the defense-in-depth design: force this transaction
        # into Postgres's own READ ONLY mode. Even if Layer 1 (role grants)
        # or Layer 2 (keyword allowlist) were ever wrong or bypassed, the
        # database engine itself now refuses any write/DDL inside this
        # transaction -- this check doesn't trust our own application code.
        conn.set_session(readonly=True)

        with conn.cursor() as cur:
            cur.execute(body.sql)
            if cur.description is None:
                # Statement executed but returned no result set (e.g. SHOW
                # with no output) -- shouldn't normally happen given our
                # allowlist, but handled defensively.
                return QueryResponse(columns=[], rows=[], row_count=0)

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            # Convert each row's values to JSON-safe types (Decimal, date,
            # etc. aren't natively JSON-serializable).
            safe_rows = [[str(v) if v is not None else None for v in row] for row in rows]

            return QueryResponse(columns=columns, rows=safe_rows, row_count=len(safe_rows))
    except psycopg2.Error as e:
        raise HTTPException(status_code=400, detail=str(e).strip())
    finally:
        conn.close()


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
