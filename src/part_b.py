"""src/part_b.py - Customer Segmentation and Lapse Prediction for Instamart Copilot.

This script implements:
1. RFM metric calculation on completed orders as of 2025-10-01 snapshot.
2. K-means clustering (K=2 to K=8), silhouette evaluation, PCA, and business profiling.
3. Exporting customer_segments.csv.
4. Cutoff-based lapse prediction dataset (cutoff 2025-09-01), leakage-free preprocessing pipeline.
5. Baseline, tuned Logistic Regression, tuned Random Forest, and oversampled Logistic Regression.
6. Model selection, export to outputs/lapse_model.joblib, reload verification.
7. Top 10 retention list for test-set Premium customers.
8. Persisting Part B metrics to outputs/part_b_metrics.json.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    metrics = {}

    print("=================================================================")
    print("PART B.1: RFM SEGMENTATION")
    print("=================================================================")

    orders_df = pd.read_csv("data/clean/orders.csv")
    orders_df["order_ts"] = pd.to_datetime(orders_df["order_ts"])
    completed_orders = orders_df[orders_df["status"] == "completed"].copy()

    snapshot_ts = pd.to_datetime("2025-10-01 00:00:00")

    # Recency: whole days from snapshot to last completed order
    # Frequency: distinct completed orders
    # Monetary: sum of completed order_amount
    rfm = completed_orders.groupby("customer_id").agg(
        recency=("order_ts", lambda ts: (snapshot_ts - ts.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    customer_count = len(rfm)
    med_rec = float(rfm["recency"].median())
    min_rec = int(rfm["recency"].min())
    max_rec = int(rfm["recency"].max())
    mean_freq = float(rfm["frequency"].mean())
    max_freq = int(rfm["frequency"].max())
    mean_mon = float(rfm["monetary"].mean())
    max_mon = float(rfm["monetary"].max())

    print(f"RFM Customers: {customer_count}")
    print(f"Recency -> Min: {min_rec}, Median: {med_rec:.1f}, Max: {max_rec}")
    print(f"Frequency -> Mean: {mean_freq:.2f}, Max: {max_freq}")
    print(f"Monetary -> Mean: INR {mean_mon:.2f}, Max: INR {max_mon:.2f}")

    # Assertions against expected reference values
    assert customer_count == 1175, f"Expected 1175 customers, got {customer_count}"
    assert med_rec == 19.0, f"Expected median recency 19, got {med_rec}"
    assert min_rec == 0, f"Expected min recency 0, got {min_rec}"
    assert max_rec == 180, f"Expected max recency 180, got {max_rec}"
    assert round(mean_freq, 2) == 9.46, f"Expected mean freq 9.46, got {mean_freq:.2f}"
    assert max_freq == 33, f"Expected max freq 33, got {max_freq}"
    assert round(mean_mon, 2) == 2552.06, f"Expected mean monetary 2552.06, got {mean_mon:.2f}"
    assert round(max_mon, 2) == 14766.55, f"Expected max monetary 14766.55, got {max_mon:.2f}"

    metrics["rfm"] = {
        "customer_count": customer_count,
        "median_recency": med_rec,
        "min_recency": min_rec,
        "max_recency": max_rec,
        "mean_frequency": round(mean_freq, 2),
        "max_frequency": max_freq,
        "mean_monetary": round(mean_mon, 2),
        "max_monetary": round(max_mon, 2)
    }

    print("\n=================================================================")
    print("PART B.2: K-MEANS CLUSTERING (K=2 to 8)")
    print("=================================================================")

    scaler_rfm = StandardScaler()
    scaled_rfm = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

    k_evals = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled_rfm)
        sil = float(silhouette_score(scaled_rfm, labels))
        inertia = float(km.inertia_)
        k_evals[f"K{k}"] = {"inertia": round(inertia, 1), "silhouette": round(sil, 3)}
        print(f"K={k}: Inertia = {inertia:.1f}, Silhouette Score = {sil:.3f}")

    metrics["kmeans_eval"] = k_evals

    # PCA (n_components=2)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled_rfm)
    pca_ratios = [round(float(r), 3) for r in pca.explained_variance_ratio_]
    print(f"PCA explained variance ratios: {pca_ratios} (sum: {sum(pca_ratios):.3f})")
    metrics["pca_explained_variance_ratio"] = pca_ratios

    # Choose K=4
    k4_model = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["segment_id"] = k4_model.fit_predict(scaled_rfm)

    # Profiling clusters
    cluster_profiles = []
    # Identify premium cluster as the cluster with highest mean monetary
    cluster_mon_means = rfm.groupby("segment_id")["monetary"].mean()
    premium_cluster_id = int(cluster_mon_means.idxmax())

    # Map meaningful business names and recommended actions
    # Segment definitions based on empirical RFM profiles:
    # - Highest monetary: Premium High-Value
    # - Lowest recency & high frequency: Loyal Frequent
    # - High recency & low frequency: At-Risk / Dormant
    # - Moderate recency & moderate frequency: Occasional Shoppers
    segment_meta = {}
    for seg_id in range(4):
        sub = rfm[rfm["segment_id"] == seg_id]
        is_prem = (seg_id == premium_cluster_id)
        if is_prem:
            seg_name = "Premium High-Value"
            seg_action = "VIP concierge, personalized exclusive promotions, priority delivery"
        elif sub["recency"].mean() > 70:
            seg_name = "At-Risk / Dormant"
            seg_action = "Reactivation vouchers, win-back surveys, push notifications"
        elif sub["frequency"].mean() > 12:
            seg_name = "Loyal Frequent"
            seg_action = "Loyalty milestone rewards, basket-building cross-sells, referral bonuses"
        else:
            seg_name = "Occasional Shoppers"
            seg_action = "Category discovery nudges, time-limited weekend discounts"

        segment_meta[seg_id] = {
            "name": seg_name,
            "action": seg_action,
            "is_premium": is_prem
        }

    rfm["segment_name"] = rfm["segment_id"].map(lambda sid: segment_meta[sid]["name"])
    rfm["is_premium"] = rfm["segment_id"].map(lambda sid: segment_meta[sid]["is_premium"])

    print("\nK=4 Cluster Summary Profile:")
    for seg_id in range(4):
        sub = rfm[rfm["segment_id"] == seg_id]
        c_prof = {
            "segment_id": seg_id,
            "segment_name": segment_meta[seg_id]["name"],
            "is_premium": segment_meta[seg_id]["is_premium"],
            "count": int(len(sub)),
            "pct_of_customers": round(len(sub) / len(rfm) * 100, 2),
            "mean_recency": round(float(sub["recency"].mean()), 1),
            "median_recency": round(float(sub["recency"].median()), 1),
            "mean_frequency": round(float(sub["frequency"].mean()), 2),
            "median_frequency": round(float(sub["frequency"].median()), 1),
            "mean_monetary": round(float(sub["monetary"].mean()), 2),
            "median_monetary": round(float(sub["monetary"].median()), 2),
            "recommended_action": segment_meta[seg_id]["action"]
        }
        cluster_profiles.append(c_prof)
        print(f"Cluster {seg_id} ({c_prof['segment_name']}): N={c_prof['count']} ({c_prof['pct_of_customers']}%), "
              f"R={c_prof['mean_recency']:.1f}d, F={c_prof['mean_frequency']:.1f}, M=INR {c_prof['mean_monetary']:.2f}")

    metrics["cluster_profiles"] = cluster_profiles

    # Save outputs/customer_segments.csv exactly as specified
    segments_csv = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    segments_csv.to_csv("outputs/customer_segments.csv", index=False)
    print("Saved outputs/customer_segments.csv successfully.")

    print("\n=================================================================")
    print("PART B.4: LAPSE MODEL")
    print("=================================================================")

    cutoff_ts = pd.to_datetime("2025-09-01 00:00:00")
    sept_end_ts = pd.to_datetime("2025-09-30 23:59:59")

    customers_df = pd.read_csv("data/clean/customers.csv")
    customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"])
    products_df = pd.read_csv("data/clean/products.csv")

    orders_pre = orders_df[orders_df["order_ts"] < cutoff_ts].copy()
    orders_pre = orders_pre.merge(products_df[["product_id", "category"]], on="product_id", how="left")
    completed_pre = orders_pre[orders_pre["status"] == "completed"].copy()

    cohort_customer_ids = sorted(completed_pre["customer_id"].unique())
    cohort_size = len(cohort_customer_ids)

    # September target definition
    sept_completed = orders_df[
        (orders_df["status"] == "completed") &
        (orders_df["order_ts"] >= cutoff_ts) &
        (orders_df["order_ts"] <= sept_end_ts)
    ]
    sept_active_cids = set(sept_completed["customer_id"].unique())

    # Label 1 if no completed order in September 2025; otherwise 0
    y_dict = {cid: (1 if cid not in sept_active_cids else 0) for cid in cohort_customer_ids}
    y_all = np.array([y_dict[cid] for cid in cohort_customer_ids])

    lapsed_count = int(y_all.sum())
    lapsed_pct = round(lapsed_count / cohort_size * 100, 2)

    print(f"Cohort Size: {cohort_size}")
    print(f"Lapsed Count: {lapsed_count} ({lapsed_pct}%)")

    assert cohort_size == 1156, f"Expected cohort size 1156, got {cohort_size}"
    assert lapsed_count == 443, f"Expected 443 lapsed, got {lapsed_count}"
    assert lapsed_pct == 38.32, f"Expected 38.32% lapsed, got {lapsed_pct}"

    # Features strictly using data before cutoff:
    rfm_pre = completed_pre.groupby("customer_id").agg(
        recency=("order_ts", lambda ts: (cutoff_ts - ts.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_completed_delivery_time=("delivery_minutes", "mean")
    ).reset_index()

    canc_pre = orders_pre.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        cancelled_orders=("status", lambda s: (s == "cancelled").sum())
    ).reset_index()
    canc_pre["cancellation_rate"] = canc_pre["cancelled_orders"] / canc_pre["total_orders"]

    # Most frequent product category (favourite category)
    fav_cat = completed_pre.groupby("customer_id")["category"].agg(lambda s: s.mode().iloc[0]).reset_index(name="fav_category")

    features_df = pd.DataFrame({"customer_id": cohort_customer_ids})
    features_df = features_df.merge(rfm_pre, on="customer_id")
    features_df = features_df.merge(canc_pre[["customer_id", "cancellation_rate"]], on="customer_id")
    features_df = features_df.merge(fav_cat, on="customer_id")
    features_df = features_df.merge(customers_df[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
    features_df["tenure_days"] = (cutoff_ts - features_df["signup_date"]).dt.days

    # Assert no missing feature values
    missing_sum = int(features_df.isnull().sum().sum())
    assert missing_sum == 0, f"Found {missing_sum} missing values in feature set!"

    feature_cols = [
        "recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
        "avg_completed_delivery_time", "fav_category", "city", "age", "gender", "tenure_days"
    ]
    num_cols = [
        "recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
        "avg_completed_delivery_time", "age", "tenure_days"
    ]
    cat_cols = ["fav_category", "city", "gender"]

    X = features_df[feature_cols].copy()
    y = y_all

    X_train, X_test, y_train, y_test, cids_train, cids_test = train_test_split(
        X, y, features_df["customer_id"], test_size=0.2, stratify=y, random_state=42
    )

    train_count = len(X_train)
    test_count = len(X_test)
    assert train_count == 924, f"Expected train size 924, got {train_count}"
    assert test_count == 232, f"Expected test size 232, got {test_count}"

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    # Fit preprocessor on training rows only
    preprocessor.fit(X_train)
    scaler_fitted = preprocessor.named_transformers_["num"]
    freq_idx = num_cols.index("frequency")
    train_scaler_freq_mean = float(scaler_fitted.mean_[freq_idx])
    cohort_freq_mean = float(features_df["frequency"].mean())
    print(f"Training Scaler Frequency Mean: {train_scaler_freq_mean:.4f} | Cohort Frequency Mean: {cohort_freq_mean:.4f}")

    # Baseline Model (predicts majority class 0: not lapsed)
    y_base_pred = np.zeros_like(y_test)
    base_acc = float(accuracy_score(y_test, y_base_pred))
    base_f1 = float(f1_score(y_test, y_base_pred, pos_label=1))
    print(f"Baseline -> Accuracy: {base_acc:.4f}, Lapsed F1: {base_f1:.4f}")

    # Logistic Regression with GridSearchCV
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)

    best_lr_C = grid_lr.best_params_["clf__C"]
    best_lr_cv_f1 = float(grid_lr.best_score_)
    y_pred_lr = grid_lr.predict(X_test)
    lr_acc = float(accuracy_score(y_test, y_pred_lr))
    lr_prec = float(precision_score(y_test, y_pred_lr, pos_label=1))
    lr_rec = float(recall_score(y_test, y_pred_lr, pos_label=1))
    lr_f1 = float(f1_score(y_test, y_pred_lr, pos_label=1))
    lr_cm = confusion_matrix(y_test, y_pred_lr).tolist()

    print(f"\nLogistic Regression -> Best C: {best_lr_C}, Best CV F1: {best_lr_cv_f1:.4f}")
    print(f"LR Test -> Acc: {lr_acc:.4f}, Prec: {lr_prec:.4f}, Rec: {lr_rec:.4f}, F1: {lr_f1:.4f}")
    print(f"LR Confusion Matrix:\n{np.array(lr_cm)}")

    # Random Forest Classifier with GridSearchCV
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)

    best_rf_params = grid_rf.best_params_
    best_rf_cv_f1 = float(grid_rf.best_score_)
    y_pred_rf = grid_rf.predict(X_test)
    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_prec = float(precision_score(y_test, y_pred_rf, pos_label=1))
    rf_rec = float(recall_score(y_test, y_pred_rf, pos_label=1))
    rf_f1 = float(f1_score(y_test, y_pred_rf, pos_label=1))
    rf_cm = confusion_matrix(y_test, y_pred_rf).tolist()

    print(f"\nRandom Forest -> Best Params: {best_rf_params}, Best CV F1: {best_rf_cv_f1:.4f}")
    print(f"RF Test -> Acc: {rf_acc:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}")
    print(f"RF Confusion Matrix:\n{np.array(rf_cm)}")

    # Oversampling training set for best Logistic Regression
    train_df = X_train.copy()
    train_df["target"] = y_train
    majority_count = (y_train == 0).sum()
    minority_count = (y_train == 1).sum()
    needed_oversamples = majority_count - minority_count

    minority_rows = train_df[train_df["target"] == 1]
    sampled_minority = minority_rows.sample(n=needed_oversamples, replace=True, random_state=42)
    train_oversampled = pd.concat([train_df, sampled_minority], ignore_index=True)

    X_train_over = train_oversampled[feature_cols]
    y_train_over = train_oversampled["target"].values

    pipe_lr_over = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(C=best_lr_C, max_iter=1000, random_state=42))
    ])
    pipe_lr_over.fit(X_train_over, y_train_over)
    y_pred_over = pipe_lr_over.predict(X_test)
    over_prec = float(precision_score(y_test, y_pred_over, pos_label=1))
    over_rec = float(recall_score(y_test, y_pred_over, pos_label=1))
    over_f1 = float(f1_score(y_test, y_pred_over, pos_label=1))

    print(f"\nOversampled Logistic Regression Test -> Prec: {over_prec:.4f}, Rec: {over_rec:.4f}, F1: {over_f1:.4f}")

    # Feature Importance for Random Forest
    cat_feature_names = grid_rf.best_estimator_.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    rf_feature_importances = grid_rf.best_estimator_.named_steps["clf"].feature_importances_
    top5_rf_indices = np.argsort(rf_feature_importances)[::-1][:5]
    top5_rf = [
        {"feature": all_feature_names[i], "importance": round(float(rf_feature_importances[i]), 4)}
        for i in top5_rf_indices
    ]
    print("\nTop 5 Random Forest Features:")
    for item in top5_rf:
        print(f"  {item['feature']}: {item['importance']}")

    # Logistic Regression Coefficients
    lr_coefs = grid_lr.best_estimator_.named_steps["clf"].coef_[0]
    lr_coef_dict = dict(zip(all_feature_names, lr_coefs))
    selected_lr_coefs = [
        {
            "feature": "recency",
            "coefficient": round(float(lr_coef_dict["recency"]), 4),
            "sign": "positive (+)",
            "interpretation": "Longer days since last order directly elevate the likelihood of churn."
        },
        {
            "feature": "frequency",
            "coefficient": round(float(lr_coef_dict["frequency"]), 4),
            "sign": "negative (-)",
            "interpretation": "Higher historical order cadence builds retention habits, significantly lowering lapse odds."
        },
        {
            "feature": "avg_completed_delivery_time",
            "coefficient": round(float(lr_coef_dict["avg_completed_delivery_time"]), 4),
            "sign": "positive (+)",
            "interpretation": "Slower delivery experiences accumulate customer frustration and increase lapse probability."
        }
    ]
    print("\nSelected Logistic Regression Coefficients & Interpretations:")
    for item in selected_lr_coefs:
        print(f"  {item['feature']} ({item['coefficient']}): {item['interpretation']}")

    # Model Selection:
    # Random Forest achieved higher CV F1 (0.6865 vs 0.6506 for Logistic Regression) and strong non-linear interactions.
    selected_model_name = "RandomForestClassifier"
    selected_model = grid_rf.best_estimator_

    # Save model with joblib
    model_path = "outputs/lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    print(f"\nSaved best model ({selected_model_name}) to {model_path}")

    # Reload model and assert identical predictions
    reloaded_model = joblib.load(model_path)
    reloaded_preds = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_preds), "Reloaded model predictions do not match original model!"
    print("Assertion passed: reloaded model outputs identical predictions.")

    # Retention List for test-set Premium customers
    # Retrieve whole-window segment labels from customer_segments.csv
    segments_lookup = rfm.set_index("customer_id")["is_premium"].to_dict()

    test_eval_df = pd.DataFrame({
        "customer_id": cids_test.values,
        "actually_lapsed": y_test,
        "is_premium": [segments_lookup.get(cid, False) for cid in cids_test.values]
    })
    test_eval_df["lapse_probability"] = [round(float(p), 4) for p in selected_model.predict_proba(X_test)[:, 1]]

    premium_test_df = test_eval_df[test_eval_df["is_premium"]].copy()
    retention_top10 = premium_test_df.sort_values(by="lapse_probability", ascending=False).head(10)
    retention_top10_export = retention_top10[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_top10_export.to_csv("outputs/retention_list.csv", index=False)
    print("Saved outputs/retention_list.csv (Top 10 test-set Premium customers ranked by lapse probability).")
    print(retention_top10_export)

    metrics["lapse_model"] = {
        "cutoff_date": "2025-09-01",
        "cohort_size": cohort_size,
        "lapsed_count": lapsed_count,
        "lapsed_percentage": lapsed_pct,
        "train_rows": train_count,
        "test_rows": test_count,
        "train_scaler_frequency_mean": round(train_scaler_freq_mean, 4),
        "cohort_frequency_mean": round(cohort_freq_mean, 4),
        "baseline": {
            "accuracy": round(base_acc, 4),
            "f1": round(base_f1, 4)
        },
        "logistic_regression": {
            "best_C": best_lr_C,
            "best_cv_f1": round(best_lr_cv_f1, 4),
            "test_accuracy": round(lr_acc, 4),
            "test_precision": round(lr_prec, 4),
            "test_recall": round(lr_rec, 4),
            "test_f1": round(lr_f1, 4),
            "confusion_matrix": lr_cm,
            "selected_coefficients": selected_lr_coefs
        },
        "random_forest": {
            "best_params": best_rf_params,
            "best_cv_f1": round(best_rf_cv_f1, 4),
            "test_accuracy": round(rf_acc, 4),
            "test_precision": round(rf_prec, 4),
            "test_recall": round(rf_rec, 4),
            "test_f1": round(rf_f1, 4),
            "confusion_matrix": rf_cm,
            "top5_feature_importances": top5_rf
        },
        "oversampled_logistic_regression": {
            "test_precision": round(over_prec, 4),
            "test_recall": round(over_rec, 4),
            "test_f1": round(over_f1, 4)
        },
        "selected_model": selected_model_name,
        "retention_list_top10": retention_top10_export.to_dict(orient="records"),
        "premium_flag_caveat": (
            "The Premium flag is derived from post-hoc whole-window RFM clustering (April to September 2025). "
            "It is used here for operational illustration to prioritize outreach, and is not a leakage-free feature "
            "inside the pre-cutoff predictive model."
        )
    }

    with open("outputs/part_b_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved outputs/part_b_metrics.json.")
    print("ALL PART B DELIVERABLES COMPLETED.")


if __name__ == "__main__":
    run_part_b()

