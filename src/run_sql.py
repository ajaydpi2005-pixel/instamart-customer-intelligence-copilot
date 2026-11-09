
import sqlite3
from pathlib import Path
import pandas as pd
import math

DB = "data/instamart.db"
SQL_FILE = Path("sql/queries.sql")

sql_text = SQL_FILE.read_text(encoding="utf-8")
queries = [
    q.strip() for q in sql_text.split(";")
    if q.strip()
]

assert len(queries) == 7, f"Expected 7 queries, found {len(queries)}"

with sqlite3.connect(DB) as conn:
    results = []

    for i, query in enumerate(queries, start=1):
        df = pd.read_sql_query(query, conn)
        results.append(df)

        print("\n" + "=" * 65)
        print(f"QUERY Q{i}")
        print("=" * 65)
        print("Rows returned:", len(df))
        print(df.head(10).to_string(index=False))

    # Q1
    assert len(results[0]) == 30

    # Q2
    q2 = results[1]
    assert len(q2) == 30
    assert math.isclose(
        q2["revenue_inr"].sum(),
        2998668.80,
        abs_tol=0.01
    )

    largest = q2.loc[q2["revenue_inr"].idxmax()]
    assert largest["city"] == "Bengaluru"
    assert largest["category"] == "Electronics"
    assert math.isclose(
        largest["revenue_inr"], 233909.45, abs_tol=0.01
    )

    # Q3
    assert len(results[2]) == 11

    # Q4
    assert len(results[3]) == 25

    # Q5
    assert len(results[4]) == 550
    average_spend = results[4]["average_spend_inr"].iloc[0]
    assert math.isclose(average_spend, 2552.06, abs_tol=0.01)

    # Q6
    q6 = results[5]
    rank_difference_count = int(
        (q6["rank_position"] != q6["row_number_position"]).sum()
    )
    assert rank_difference_count == 8

    # Q7
    q7 = results[6]
    assert len(q7) == 183
    assert math.isclose(
        q7["cumulative_revenue_inr"].iloc[-1],
        2998668.80,
        abs_tol=0.01
    )

    print("\n" + "=" * 65)
    print("SQL VALIDATION SUMMARY")
    print("=" * 65)
    print("Q1 customers:", len(results[0]))
    print("Q2 city/category rows:", len(q2))
    print("Q2 total revenue: INR", round(q2["revenue_inr"].sum(), 2))
    print("Q2 largest revenue cell:", largest["city"],
          "/", largest["category"],
          "INR", round(largest["revenue_inr"], 2))
    print("Q3 products:", len(results[2]))
    print("Q4 customers without completed orders:", len(results[3]))
    print("Q5 above-average customers:", len(results[4]))
    print("Q5 average spend: INR", average_spend)
    print("Q6 RANK vs ROW_NUMBER differences:", rank_difference_count)
    print("Q7 daily rows:", len(q7))
    print("Q7 final cumulative revenue: INR",
          q7["cumulative_revenue_inr"].iloc[-1])

    print("\nALL 7 SQL QUERIES PASSED")
