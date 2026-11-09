
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw customer data.
customers = pd.read_csv(RAW / "customers.csv")

# 1. Standardise city names.
customers["city"] = customers["city"].str.strip().str.title()

# 2. Standardise gender values.
gender_values = customers["gender"].fillna("").str.strip().str.lower()

gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female"
}

customers["gender"] = gender_values.map(gender_map).fillna("unknown")

# 3. Convert age to numeric.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

# Treat ages outside 10–100 as missing.
valid_age = customers["age"].between(10, 100)
customers.loc[~valid_age, "age"] = float("nan")

# Fill missing ages using the median age of each city.
city_median = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median)

# 4. Convert signup_date into datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

# Save cleaned customers.
customers.to_csv(CLEAN / "customers.csv", index=False)

# Products have no cleaning transformations specified.
# Save the original product records into the clean folder.
products = pd.read_csv(RAW / "products.csv")
products.to_csv(CLEAN / "products.csv", index=False)

# Display validation results.
print("\nCUSTOMER CLEANING RESULTS")
print("-" * 40)

print("\nCity counts:")
print(customers["city"].value_counts())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("\nCustomer rows:", len(customers))
print("Product rows:", len(products))

print("\nSaved:")
print("data/clean/customers.csv")
print("data/clean/products.csv")
