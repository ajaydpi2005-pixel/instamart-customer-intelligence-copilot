
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

# Load the three raw datasets without changing the original files.
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

datasets = {
    "CUSTOMERS": customers,
    "PRODUCTS": products,
    "ORDERS": orders
}

# Report shape, data types and missing values for every column.
for name, df in datasets.items():
    print(f"\n{'=' * 60}")
    print(f"{name} DATA AUDIT")
    print("=" * 60)

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2)
    })
    print(missing)

# Specific raw orders audit figures.
amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

# Specific raw customers audit figures.
age_numeric = pd.to_numeric(customers["age"], errors="coerce")

audit_results = {
    "Raw orders rows": len(orders),
    "Exact duplicate order rows": int(orders.duplicated().sum()),
    "Pending order rows": int((orders["status"].str.lower() == "pending").sum()),
    "Distinct order IDs": int(orders["order_id"].nunique()),
    "Non-numeric order amounts": int(amount_numeric.isna().sum()),
    "Negative delivery minutes": int((delivery_numeric < 0).sum()),
    "Blank delivery minutes": int(orders["delivery_minutes"].isna().sum()),
    "Blank customer ages": int(customers["age"].isna().sum()),
    "Customer ages outside 10-100": int(
        ((age_numeric < 10) | (age_numeric > 100)).sum()
    ),
    "Distinct raw city spellings": int(customers["city"].nunique()),
    "Distinct non-blank gender spellings": int(
        customers.loc[customers["gender"].notna() &
                      (customers["gender"].astype(str).str.strip() != ""),
                      "gender"].nunique()
    ),
    "Blank customer genders": int(customers["gender"].isna().sum())
}

print("\n" + "=" * 60)
print("SPECIFIC AUDIT CHECKPOINTS")
print("=" * 60)

for item, value in audit_results.items():
    print(f"{item}: {value}")