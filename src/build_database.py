
import sqlite3
from pathlib import Path
import pandas as pd
import math

CLEAN = Path("data/clean")
DB_PATH = Path("data/instamart.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load cleaned CSV files.
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Ensure timestamps are stored as text in SQLite.
orders["order_ts"] = orders["order_ts"].astype(str)

# Create and populate the database.
with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

# Read database tables for validation.
with sqlite3.connect(DB_PATH) as conn:
    db_customers = pd.read_sql_query("SELECT * FROM customers", conn)
    db_products = pd.read_sql_query("SELECT * FROM products", conn)
    db_orders = pd.read_sql_query("SELECT * FROM orders", conn)

# Check at least eight assignment reference checkpoints.
assert len(db_customers) == 1200
assert len(db_products) == 40
assert len(db_orders) == 12309

assert db_customers["city"].nunique() == 6
assert db_customers["gender"].value_counts().to_dict() == {
    "female": 610, "male": 560, "unknown": 30
}

assert db_customers["age"].isna().sum() == 0
assert db_customers["age"].min() == 18
assert db_customers["age"].max() == 60

status_counts = db_orders["status"].value_counts().to_dict()
assert status_counts.get("completed", 0) == 11110
assert status_counts.get("cancelled", 0) == 1199
assert "pending" not in status_counts

assert db_orders["order_amount"].isna().sum() == 0

completed = db_orders["status"] == "completed"
cancelled = db_orders["status"] == "cancelled"

assert db_orders.loc[completed, "delivery_minutes"].isna().sum() == 0
assert db_orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199

revenue = db_orders.loc[completed, "order_amount"].sum()
assert math.isclose(revenue, 2998668.80, abs_tol=0.01)

# Confirm the three database tables exist.
with sqlite3.connect(DB_PATH) as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )["name"].tolist()

assert {"customers", "products", "orders"}.issubset(set(tables))

print("ALL CHECKPOINTS PASSED")
