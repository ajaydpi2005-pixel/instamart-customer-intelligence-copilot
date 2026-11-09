
import pandas as pd
import numpy as np

# Load cleaned datasets.
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# Use completed orders only for revenue analysis.
completed = orders.loc[
    orders["status"] == "completed"
].copy()
# Join customer city and product category into completed orders.
completed = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)

completed = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

assert completed["city"].notna().all()
assert completed["category"].notna().all()

for col in ["order_amount", "discount_pct", "quantity", "delivery_minutes"]:
    completed[col] = pd.to_numeric(completed[col], errors="coerce")

# --------------------------------------------------
# 1. PIVOT TABLE: CITY AND CATEGORY REVENUE
# --------------------------------------------------
print("\n1. REVENUE PIVOT TABLE")
print("=" * 65)

pivot = pd.pivot_table(
    completed,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2))

pivot_total = pivot.loc["Grand Total", "Grand Total"]
print("\nPandas grand total: INR", round(pivot_total, 2))

assert abs(pivot_total - 2998668.80) < 0.01

# --------------------------------------------------
# 2. CITY-WISE GROUPBY ANALYSIS
# --------------------------------------------------
print("\n2. CITY-WISE REVENUE SUMMARY")
print("=" * 65)

city_summary = completed.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    mean_order_amount=("order_amount", "mean"),
    completed_order_count=("order_amount", "count")
).round(2)

print(city_summary)

# Reference checks for Bengaluru and Mumbai.
assert abs(city_summary.loc["Bengaluru", "total_revenue"] - 523643.40) < 0.01
assert abs(city_summary.loc["Bengaluru", "mean_order_amount"] - 276.47) < 0.01
assert city_summary.loc["Bengaluru", "completed_order_count"] == 1894

assert abs(city_summary.loc["Mumbai", "total_revenue"] - 479058.50) < 0.01
assert abs(city_summary.loc["Mumbai", "mean_order_amount"] - 297.00) < 0.01
assert city_summary.loc["Mumbai", "completed_order_count"] == 1613

# --------------------------------------------------
# 3. ORDER AMOUNT DISTRIBUTION AND OUTLIERS
# --------------------------------------------------
print("\n3. ORDER AMOUNT DISTRIBUTION")
print("=" * 65)

amount = completed["order_amount"].dropna()

mean_value = amount.mean()
median_value = amount.median()
skew_value = amount.skew()

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

lower_outliers = int((amount < lower_fence).sum())
upper_outliers = int((amount > upper_fence).sum())

print(f"Mean: INR {mean_value:.2f}")
print(f"Median: INR {median_value:.2f}")
print(f"Skewness: {skew_value:.2f}")
print(f"Q1: INR {q1:.2f}")
print(f"Q3: INR {q3:.2f}")
print(f"IQR: INR {iqr:.2f}")
print(f"Lower fence: INR {lower_fence:.2f}")
print(f"Upper fence: INR {upper_fence:.2f}")
print("Outliers below lower fence:", lower_outliers)
print("Outliers above upper fence:", upper_outliers)

assert abs(mean_value - 269.91) < 0.02
assert abs(median_value - 135.00) < 0.01
assert abs(skew_value - 4.61) < 0.05
assert abs(q1 - 64.80) < 0.02
assert abs(q3 - 297.00) < 0.02
assert abs(upper_fence - 645.30) < 0.05
assert lower_outliers == 0
assert upper_outliers == 897

# --------------------------------------------------
# 4. CORRELATION MATRIX
# --------------------------------------------------
print("\n4. CORRELATION MATRIX")
print("=" * 65)

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

corr = completed[correlation_columns].corr()
print(corr.round(4))

# Find the strongest absolute off-diagonal correlation.
upper_triangle = corr.where(
    np.triu(np.ones(corr.shape, dtype=bool), k=1)
)

pairs = upper_triangle.stack()
strongest_pair = pairs.abs().idxmax()
strongest_value = pairs.loc[strongest_pair]

print("\nStrongest correlation pair:", strongest_pair)
print(f"Correlation value: {strongest_value:.4f}")

# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------
print("\nALL PANDAS CHECKPOINTS PASSED")
