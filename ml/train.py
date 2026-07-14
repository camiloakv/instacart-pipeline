"""
Trains an XGBoost binary classifier to predict whether a user will reorder
a given product, using the dbt marts table `fct_user_product_features`.

Usage:
    python train.py

Reads Postgres connection details from environment variables (same ones
used across the project): POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB,
POSTGRES_HOST, POSTGRES_PORT.
"""

import os

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# Loads variables from a .env file at the project root into os.environ,
# so this script works the same whether run via Docker or directly (uv run).
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models", "model.json")

# Features fed to the model. Excludes identifiers (user_id, product_id),
# high-cardinality text (product_name), and the target itself.
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

TARGET_COLUMN = "target_reordered"


def load_features_from_postgres() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST_LOCAL", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    print("CONN CREATED")
    query = """
        select
            user_id, product_id,
            user_total_orders, user_avg_days_between_orders,
            user_avg_order_dow, user_avg_order_hour,
            product_total_orders, product_reorder_rate,
            up_times_ordered, up_times_reordered, up_avg_add_to_cart_order,
            up_first_order_number, up_last_order_number,
            up_orders_since_last_purchase,
            aisle_id, department_id,
            target_reordered
        from dbt_dev.fct_user_product_features
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    print("FETCHED")
    conn.close()
    print("CONN CLOSED")
    return pd.DataFrame(rows, columns=columns)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalized reorder ratio -- often more informative than raw counts alone
    df["up_reorder_ratio"] = df["up_times_reordered"] / df["up_times_ordered"]
    return df


def main():
    print("Loading features from Postgres...")
    df = load_features_from_postgres()
    print(f"Loaded {len(df):,} rows")

    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df["user_id"]  # split by user, not by row, to avoid leakage

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Train class balance -- pos: {pos_count:,}, neg: {neg_count:,}, "
          f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\n--- Evaluation on held-out users ---")
    print(f"AUC:      {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"LogLoss:  {log_loss(y_test, y_pred_proba):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Feature importance
    print("\n--- Feature importance ---")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).to_string())

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save_model(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
