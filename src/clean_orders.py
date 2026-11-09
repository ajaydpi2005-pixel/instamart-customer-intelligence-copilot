
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw orders and products.
orders = pd.read_csv(RAW / "orders.csv")
products = pd.read_csv(RAW / "products.csv")

# 1. Convert order timestamps to datetime.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# 2. Remove exact duplicate rows.
before_duplicates = len(orders)
orders = orders.drop_duplicates().copy()
exact_duplicates_removed = before_duplicates - len(orders)

# 3. Keep only the latest record for each order_id.
before_latest = len(orders)
orders = orders.sort_values("order_ts", kind="mergesort")
orders = orders.drop_duplicates(
    subset=["order_id"], keep="last"
).copy()
older_records_removed = before_latest - len(orders)

# Add product category and price temporarily.
orders = orders.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

assert orders["category"].notna().all(), "Product category missing"

# 4. Convert order amounts to numeric.
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Fill missing amounts using the product price formula.
missing_amounts = orders["order_amount"].isna().sum()

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# 5. Convert delivery time to numeric.
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Negative delivery times are invalid.
negative_delivery = orders["delivery_minutes"] < 0
negative_delivery_count = int(negative_delivery.sum())
orders.loc[negative_delivery, "delivery_minutes"] = float("nan")

# Cancelled orders must have missing delivery time.
cancelled = orders["status"].eq("cancelled")
orders.loc[cancelled, "delivery_minutes"] = float("nan")

# Fill missing delivery times for completed orders using
# the median of completed orders in the same product category.
completed = orders["status"].eq("completed")

category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

completed_delivery = orders.loc[completed, "delivery_minutes"]

orders.loc[completed, "delivery_minutes"] = (
    completed_delivery.fillna(category_medians)
)

# Remove temporary product columns.
orders = orders.drop(columns=["category", "price_inr"])

# Sort by order timestamp for reproducible output.
orders = orders.sort_values(
    ["order_ts", "order_id"]
).reset_index(drop=True)

# Save clean orders.
orders.to_csv(CLEAN / "orders.csv", index=False)

# Display cleaning results.
print("\nORDER CLEANING RESULTS")
print("=" * 45)

print("Raw rows:", 13084)
print("Exact duplicate rows removed:", exact_duplicates_removed)
print("Older order records removed:", older_records_removed)
print("Final clean rows:", len(orders))

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing amount values:", orders["order_amount"].isna().sum())
print("Missing completed delivery times:",
      orders.loc[completed, "delivery_minutes"].isna().sum())
print("Missing cancelled delivery times:",
      orders.loc[cancelled, "delivery_minutes"].isna().sum())

print("\nNegative delivery times corrected:", negative_delivery_count)
print("Missing amounts filled:", missing_amounts)

revenue = orders.loc[
    orders["status"].eq("completed"), "order_amount"
].sum()

print(f"\nCompleted order revenue: INR {revenue:.2f}")

print("\nSaved: data/clean/orders.csv")
