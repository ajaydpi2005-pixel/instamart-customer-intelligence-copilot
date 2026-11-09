# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an end-to-end customer intelligence and support copilot for Instamart using synthetic retail data. The system spans data cleaning, relational SQLite modeling, SQL and pandas analytics, unsupervised RFM customer segmentation, supervised churn/lapse prediction, support ticket triage with privacy guardrails and token budget enforcement, and a resilient, checkpointed ReAct refund agent grounded in corporate policy documents.

All monetary values are in INR. All data is synthetic and fictional. All language model operations are performed deterministically offline using the provided `mock_llm.py`.

## Reproduce

Run these commands from the repository root:

1. Create a virtual environment:
   `python -m venv .venv`

2. Activate it:
   `source .venv/bin/activate`

3. Install dependencies:
   `pip install -r requirements.txt`

4. Generate raw data:
   `python generate_data.py`

5. Audit raw data:
   `python src/audit_raw.py`

6. Clean customers and products:
   `python src/clean_customers.py`

7. Clean orders:
   `python src/clean_orders.py`

8. Build and validate the database:
   `python src/build_database.py`

9. Run SQL analysis:
   `python src/run_sql.py`

10. Run pandas analysis:
    `python src/pandas_analysis.py`

11. Run customer segmentation and lapse prediction (Part B):
    `python src/part_b.py`

12. Run support ticket triage with guardrails (Part C):
    `python src/part_c.py`

13. Run policy RAG and refund agent (Part D):
    `python src/part_d.py`

## Part A – Data Foundation: Clean, Store, Query

### Raw data audit

The raw dataset was audited using `src/audit_raw.py` and output saved to `outputs/raw_audit.txt`.

#### Dataset Shapes and Data Types

| Dataset | Rows | Columns | Column Data Types |
|---|---:|---:|---|
| Customers | 1,200 | 5 | `customer_id` (object), `city` (object), `signup_date` (object), `age` (float64), `gender` (object) |
| Products | 40 | 4 | `product_id` (object), `product_name` (object), `category` (object), `price_inr` (int64) |
| Orders | 13,084 | 9 | `order_id` (object), `customer_id` (object), `product_id` (object), `quantity` (int64), `order_ts` (object), `status` (object), `discount_pct` (int64), `order_amount` (object), `delivery_minutes` (float64) |

#### Per-Column Missing Value Counts and Percentages

- **Customers**:
  - `customer_id`: 0 (0.00%)
  - `city`: 0 (0.00%)
  - `signup_date`: 0 (0.00%)
  - `age`: 61 missing (5.08%)
  - `gender`: 30 missing (2.50%)

- **Products**:
  - `product_id`: 0 (0.00%)
  - `product_name`: 0 (0.00%)
  - `category`: 0 (0.00%)
  - `price_inr`: 0 (0.00%)

- **Orders**:
  - `order_id`: 0 (0.00%)
  - `customer_id`: 0 (0.00%)
  - `product_id`: 0 (0.00%)
  - `quantity`: 0 (0.00%)
  - `order_ts`: 0 (0.00%)
  - `status`: 0 (0.00%)
  - `discount_pct`: 0 (0.00%)
  - `order_amount`: 53 missing (0.41%)
  - `delivery_minutes`: 2,541 missing (19.42%)

#### Specific Audit Findings

| Audit item | Result |
|---|---:|
| Total raw orders | 13,084 |
| Distinct order IDs | 12,309 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Non-numeric order amounts | 120 |
| Negative delivery minutes | 106 |
| Blank delivery minutes | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Non-blank gender spellings | 10 |
| Blank genders | 30 |

### Customer and product cleaning

- City spellings were standardised using trimming and title case.
- Gender spellings were mapped to male, female or unknown.
- Ages outside 10–100 were treated as missing.
- Missing ages were filled using the median age of each customer's city.
- Signup dates were converted to datetime.
- Clean customer and product files were saved under `data/clean/`.

Customer validation:

| City | Customers |
|---|---:|
| Bengaluru | 197 |
| Chennai | 191 |
| Delhi | 210 |
| Hyderabad | 191 |
| Mumbai | 184 |
| Pune | 227 |

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

Missing ages: 0. Minimum age: 18. Maximum age: 60.

### Order cleaning

The following rules were applied in order:

1. Converted order timestamps to datetime.
2. Removed 119 exact duplicate rows.
3. Kept the latest timestamp record for each order ID, removing 656 older records.
4. Converted order amounts to numeric and filled missing amounts using product price × quantity × (1 − discount percentage / 100).
5. Set negative delivery times to missing.
6. Filled missing completed-order delivery times using the median of the product category.
7. Kept cancelled-order delivery times missing.

Validation results:

| Check | Result |
|---|---:|
| Clean order rows | 12,309 |
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |
| Missing amounts | 0 |
| Missing completed delivery times | 0 |
| Missing cancelled delivery times | 1,199 |
| Completed revenue | INR 2,998,668.80 |

### SQLite database

Created `data/instamart.db` with three tables:
- `customers`
- `products`
- `orders`

Database validation output: `ALL CHECKPOINTS PASSED`.

### SQL analysis

| Query | Result |
|---|---:|
| Q1 – WHERE and HAVING | 30 customers |
| Q2 – City/category revenue | 30 rows |
| Q3 – Above-average products | 11 products |
| Q4 – No completed orders | 25 customers |
| Q5 – Above-average spend | 550 customers |
| Q6 – RANK vs ROW_NUMBER differences | 8 rows |
| Q7 – Daily revenue | 183 rows |

Total completed revenue: INR 2,998,668.80.
The largest city/category revenue was Bengaluru / Electronics at INR 233,909.45.
For Q6, ties receive the same RANK value, whereas ROW_NUMBER assigns a unique sequential number to every row.

### Pandas analysis

| Metric | Result |
|---|---:|
| Mean order amount | INR 269.91 |
| Median order amount | INR 135.00 |
| Skewness | 4.61 |
| Q1 | INR 64.80 |
| Q3 | INR 297.00 |
| Lower fence | INR -283.50 |
| Upper fence | INR 645.30 |
| Lower outliers | 0 |
| Upper outliers | 897 |

The order amount distribution is strongly right-skewed. The mean is higher than the median because a relatively small number of high-value purchases increase the average. The strongest absolute correlation is between order amount and discount percentage (-0.0951). This is a weak negative relationship and does not establish causation.
Pandas revenue total: INR 2,998,668.80.

## Part B – Customer Segments and Lapse Prediction

### RFM Metrics

RFM metrics were calculated on completed orders with snapshot `2025-10-01 00:00:00`:
- **Recency**: Whole days from snapshot to customer's most recent completed order.
- **Frequency**: Distinct completed orders per customer.
- **Monetary**: Sum of completed order amounts in INR.

| Metric | Value | Reference Check |
|---|---:|---|
| Qualifying Customers | 1,175 | Exactly matches expected (1,175) |
| Recency (Min / Median / Max) | 0 / 19.0 / 180 days | Min 0, Median 19, Max 180 |
| Frequency (Mean / Max) | 9.46 / 33 orders | Mean 9.46, Max 33 |
| Monetary (Mean / Max) | INR 2,552.06 / INR 14,766.55 | Mean INR 2,552.06, Max INR 14,766.55 |

### K-Means Clustering and PCA

StandardScaler was applied to RFM features. K-means clustering was evaluated from K=2 to K=8 using `n_init=10` and `random_state=42`:

| Clusters (K) | Inertia | Silhouette Score | Reference Benchmark |
|---|---:|---:|---|
| K=2 | 1820.8 | 0.465 | 1820.8 / 0.465 |
| K=3 | 1268.2 | 0.445 | 1268.2 / 0.445 |
| **K=4** | **743.1** | **0.526** | **743.1 / 0.526 (Optimal Peak)** |
| K=5 | 617.4 | 0.480 | 617.4 / 0.481 |
| K=6 | 503.2 | 0.482 | 503.2 / 0.482 |
| K=7 | 408.5 | 0.416 | 408.5 / 0.416 |
| K=8 | 366.5 | 0.412 | 366.5 / 0.412 |

PCA applied to scaled RFM with `n_components=2` yielded explained variance ratios of **0.634** (PC1) and **0.218** (PC2), capturing **85.2%** of total variance.

K=4 was selected based on the silhouette score maximum (0.526) and business interpretability. Cluster profiling results:

| Cluster ID | Segment Name | Customers (%) | Mean Recency | Mean Frequency | Mean Monetary | Recommended Business Action |
|---|---|---:|---:|---:|---:|---|
| 0 | Occasional Shoppers | 436 (37.11%) | 28.9 days | 3.8 orders | INR 1,057.62 | Category discovery nudges, weekend flash discounts |
| 1 | **Premium High-Value** | 138 (11.74%) | 24.1 days | 6.8 orders | **INR 7,129.70** | VIP concierge desk, exclusive products, waived delivery |
| 2 | Loyal Frequent | 419 (35.66%) | 8.8 days | 19.4 orders | INR 3,410.02 | Loyalty tier rewards, basket-building cross-sells |
| 3 | At-Risk / Dormant | 182 (15.49%) | 109.0 days | 2.2 orders | INR 685.99 | Win-back vouchers, re-engagement surveys |

Cluster 1 exhibited the highest mean monetary spend (INR 7,129.70) and was designated as **Premium** (`is_premium=True`). Results were saved to `outputs/customer_segments.csv`.

### Lapse Prediction Model

- **Cutoff Date**: `2025-09-01 00:00:00`.
- **Cohort**: Customers with $\ge 1$ completed order prior to cutoff ($N = 1,156$).
- **Target Label**: `1` (Lapsed) if no completed order between September 1 and September 30, 2025; otherwise `0`.
  - Lapsed count: **443** (38.32%).
- **Feature Store**: Strictly computed before cutoff (no lookahead leakage): recency, frequency, monetary, average discount percentage, cancellation rate, average completed delivery time, favourite product category, city, age, gender, and customer tenure in days.
- **Data Splitting**: Stratified 80/20 train/test split (`random_state=42`):
  - Training set: 924 customers (570 active, 354 lapsed).
  - Test set: 232 customers (143 active, 89 lapsed).
- **Leakage Prevention**: ColumnTransformer with StandardScaler fitted on training rows only.
  - Training scaler frequency mean: **8.1147** vs Cohort frequency mean: **8.0285**.

#### Model Comparison and Evaluation

All models were evaluated on the held-out test set ($N=232$):

| Model | Tuning / Hyperparameters | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 (Lapsed) |
|---|---|---:|---:|---:|---:|---:|
| Zero-Rule Baseline | Always predict Active (0) | N/A | 61.64% | 0.00% | 0.00% | 0.0000 |
| Logistic Regression | $C=100$, max_iter=1000 | 0.6506 | 72.84% | 64.44% | 65.17% | 0.6480 |
| **Random Forest** | **max_depth=8, min_samples_leaf=10** | **0.6865** | **72.41%** | **64.04%** | **64.04%** | **0.6404** |
| Oversampled Logistic Reg. | $C=100$, balanced train minority | N/A | 69.83% | 58.91% | **85.39%** | **0.6972** |

#### Confusion Matrices (Test Set, N=232)

- **Logistic Regression**:
  $$\begin{pmatrix} 111 & 32 \\ 31 & 58 \end{pmatrix}$$
- **Random Forest**:
  $$\begin{pmatrix} 111 & 32 \\ 32 & 57 \end{pmatrix}$$

#### Feature Importance and Coefficient Interpretation

- **Top 5 Random Forest Features**:
  1. `monetary`: 0.2778
  2. `frequency`: 0.2768
  3. `recency`: 0.1112
  4. `avg_completed_delivery_time`: 0.0616
  5. `avg_discount`: 0.0532

- **Selected Logistic Regression Coefficients**:
  1. `recency` (+0.0396): Positive sign indicates that longer periods of inactivity before the cutoff directly elevate the odds of lapsing.
  2. `frequency` (-1.6982): Strong negative sign shows that established buying cadence reinforces customer retention habits and substantially lowers lapse odds.
  3. `avg_completed_delivery_time` (+0.0785): Positive sign demonstrates that operational friction in delivery speed increases customer churn risk.

#### Model Selection and Retention Prioritization

Random Forest was selected as the final production model based on superior 5-fold cross-validation performance (CV F1 **0.6865** vs **0.6506** for Logistic Regression) and robust non-linear interaction modeling. The pipeline was serialized to `outputs/lapse_model.joblib`. Reloading the artifact confirmed bitwise identical predictions on the test set.

Top 10 test-set Premium customers ranked by predicted lapse probability were exported to `outputs/retention_list.csv`:

| Customer ID | Predicted Lapse Probability | Actually Lapsed |
|---|---:|---:|
| C0862 | 0.5304 | 0 |
| C0060 | 0.4765 | 0 |
| C1008 | 0.4756 | 0 |
| C0267 | 0.4753 | 0 |
| C0133 | 0.4712 | 0 |
| C0942 | 0.4688 | 0 |
| C1000 | 0.4654 | 0 |
| C0553 | 0.4645 | 0 |
| C1112 | 0.4620 | 0 |
| C0565 | 0.4603 | 1 |

> [!NOTE]
> The `is_premium` flag is derived from post-hoc whole-window RFM clustering (April through September 2025). The retention list serves as an operational illustration for high-value customer intervention and is not a leakage-free predictive evaluation.

## Part C – Support Ticket Triage with Guardrails

### Input Guardrails and PII Redaction

Raw tickets from `data/raw/tickets.csv` ($N = 164$) were processed with strict pre-LLM guardrails:
1. **Empty/Whitespace Tickets**: 4 tickets identified and flagged as `skipped_empty` without invoking the model.
2. **Length Guardrail**: 4 tickets exceeded the 400-token threshold (measured via `mock_llm.est_tokens`, each exactly 734 tokens) and were flagged as `flagged_long` without invoking the model.
3. **PII Masking**: Redacted in strict sequence:
   - Email regex (`[\w.+-]+@[\w-]+\.[\w.]+`): 55 matches $\to$ `[EMAIL]`
   - Phone regex (`\b[6-9]\d{9}\b`): 91 matches $\to$ `[PHONE]`
   - Order ID regex (`ORD-\d{6}`): 132 matches $\to$ `[ORDER_ID]`
   - **Total PII Redacted**: 278 instances.
   - Re-scanning confirmed zero residual matches. Only redacted text was transmitted to the model.

### Resilience and Output Parsing

A resilient client was implemented with exponential backoff and jitter (`0.05 * 2^n * uniform(0.9, 1.15)`) catching `mock_llm.TransientLLMError`. All 19 transient 503 errors per full run were automatically recovered across retries and logged to `outputs/triage_retry_log.txt`.

Model responses were validated using a two-stage parsing fallback:
1. Valid JSON matching `{"category": "...", "reason": "..."}` with category in `{billing, delivery, technical, fraud}` $\to$ `classified_json`.
2. Fallback lowercase word extraction (`re.findall(r"[a-z]+", ...)`) matching exactly one allowed category $\to$ `classified_keyword`.
3. Out-of-domain categories (e.g. "shipping") or malformed outputs $\to$ `flagged_invalid_output`.

### Benchmark Results: Zero-Shot vs Few-Shot

| Metric | Zero-Shot Mode | Few-Shot Mode |
|---|---:|---:|
| Eligible Tickets Processed | 156 | 156 |
| `classified_json` | 135 | 135 |
| `classified_keyword` | 12 | 12 |
| `flagged_invalid_output` | 9 | 9 |
| Total Classified Tickets | 147 | 147 |
| **Classification Accuracy** | **85.71%** | **100.00%** |
| Total Tokens Consumed | 7,060 tokens | 19,852 tokens |
| Tokens per Classified Ticket | 48.03 tokens | 135.05 tokens |

#### Zero-Shot Confusion Matrix (Labels: billing, delivery, fraud, technical)

| True \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| **Billing** | **38** | 0 | 0 | 0 |
| **Delivery** | 0 | **36** | 0 | 0 |
| **Fraud** | 11 | 0 | **23** | 0 |
| **Technical** | 10 | 0 | 0 | **29** |

*Analysis*: In zero-shot mode, tickets mentioning financial words (e.g., unauthorized card charges) were misclassified as billing rather than fraud (11 cases) or technical (10 cases). In few-shot mode, explicit prompt demonstrations resolved all ambiguities, achieving 100% accuracy.

### Token Budget Demonstration

A budget ceiling of 6,000 tokens was applied to zero-shot triage. Processing halted once cumulative token consumption reached 6,015 tokens, stopping 23 remaining tickets with status `budget_exhausted`. Results were saved to `outputs/triage_budget_demo.csv`.

### Production Few-Shot Triage Run

With `TOKEN_BUDGET=100,000`, all 164 tickets were triaged and routed into business queues using Part B customer segmentation data:

- Guardrail fallbacks (`skipped_empty`, `flagged_long`, `flagged_invalid_output`) $\to$ `human_review` (NORMAL)
- `fraud` $\to$ `security_team` (HIGH)
- Classified tickets from `Premium` customers $\to$ `premium_desk` (HIGH)
- All other tickets $\to$ `support_queue` (NORMAL)

| Routing Queue | Priority | Ticket Count | Description |
|---|---|---:|---|
| `support_queue` | NORMAL | 102 | Standard customer support handling |
| `security_team` | HIGH | 34 | Immediate fraud and unauthorized charge investigation |
| `human_review` | NORMAL | 17 | Edge cases: 4 empty + 4 long + 9 schema failures |
| `premium_desk` | HIGH | 11 | Expedited support for Premium segment customers |

The final triage dataset containing all 164 records was saved to `outputs/triage_results.csv`.

## Part D – Policy RAG Assistant and Refund Agent

### Policy Chunking and RAG Evaluation

Five corporate policy documents (`delivery_policy.txt`, `escalation_policy.txt`, `payment_policy.txt`, `privacy_policy.txt`, `refund_policy.txt`) were chunked with a 40-word window:
- **Overlap 10 (Step 30)**: Produced exactly **15 chunks** (refund: 4, delivery: 3, escalation: 3, payment: 3, privacy: 2).
- **Overlap 0 (Step 40)**: Produced 13 chunks.

Chunks were indexed with `TfidfVectorizer` and retrieved via cosine similarity ($k=3$). Grounded answers were generated using `mock_llm.chat` under the `POLICY ASSISTANT` system persona.

#### Benchmark Evaluation on `data/raw/policy_eval.csv`

| Evaluation Metric | Overlap 10 Words | Overlap 0 Words (Ablation) |
|---|---:|---:|
| Answerable Questions | 12 | 12 |
| Unanswerable Questions | 2 | 2 |
| Top-3 Document Retrieval Hits | 12 / 12 (100.0%) | 12 / 12 (100.0%) |
| Exact Fact Substring Match | **12 / 12 (100.0%)** | **10 / 12 (83.3%)** |
| Fallback Answer Hits (`I could not find...`) | 2 / 2 (100.0%) | 2 / 2 (100.0%) |

**Impact of Chunk Boundary Splitting (Overlap 0)**:
Without overlap, semantic context split across chunk boundaries, causing two answer failures:
1. *"What delivery time makes a completed order eligible for a full refund?"*: The qualifying phrase *"more than 45 minutes"* was separated from the refund condition, returning an incomplete answer.
2. *"How soon is money returned when a payment fails but is deducted?"*: The *"48 hours"* timeline was clipped at the chunk boundary.

### Agent Architecture and Tool Suite

The refund agent implements a ReAct loop with five callable tools:
1. `get_order(order_id)`: Queries `data/instamart.db` for status, amount, and delivery minutes.
2. `search_policy(query)`: Grounded TF-IDF retrieval across policy chunks.
3. `calculate_refund(order_amount, delivery_minutes)`: Calculates 100% refund for delivery delays $>45$ minutes; 0% otherwise.
4. `issue_refund(order_id, refund_inr)`: Writes approved refund to `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Escalates claims $> \text{INR } 1,000$ to human lead and writes to ledger.

### Scenario Evaluation Results

All seven scenarios from `data/raw/agent_scenarios.csv` were executed and saved to `outputs/agent_runs.jsonl`:

| Scenario | Customer | Order ID | Input Message | Status | LLM Calls | Tokens | Final Outcome |
|---|---|---|---|---|---:|---:|---|
| S1 | C0007 | ORD-100048 | Order reached me very late... | REFUND | 5 | 853 | INR 128.25 refunded automatically |
| S2 | C0008 | ORD-100067 | Order reached me very late... | REFUND | 5 | 853 | INR 135.00 refunded automatically |
| S3 | C0016 | ORD-100145 | Arrived far too late... | ESCALATE | 5 | 840 | Escalated to lead (> INR 1000) |
| S4 | C0001 | ORD-100001 | Did not like delivery... | DENIED | 4 | 571 | Delivery within 45 mins (30 mins) |
| S5 | C0003 | ORD-100025 | Want money back... | DENIED | 3 | 318 | Order was cancelled (never charged) |
| S6 | C0001 | ORD-999999 | Refund for order please... | NOT_FOUND | 2 | 69 | Order ID not found in database |
| S7 | C0007 | ORD-100048 | [LOOP] Refund order... | FALLBACK_MAX_ITERATIONS | 6 | 2,763 | Reached 6-iteration loop ceiling |

### Safe Stops Demonstration

Safety guardrails were verified on Scenario S1:
- **Token Limit Guardrail (`token_limit=200`)**: Halted immediately before executing the third tool call once cumulative token usage exceeded the ceiling (335 tokens used), returning status `FALLBACK_TOKEN_LIMIT`.
- **Timeout Guardrail (`time_limit_seconds=0`)**: Halted at the top of the initial iteration, returning status `FALLBACK_TIMEOUT`.

### Checkpointing and Crash Recovery

State was persisted after every tool observation to `state/<run_id>.json`, recording run ID, scenario ID, message trajectory, token usage, step count, and executed tool names.

- **Crash Simulation**: S1 was interrupted immediately after the second tool execution (`search_policy`).
  - Pre-crash saved tools: `['get_order', 'search_policy']`.
- **Resumption (`resume_agent`)**: Re-loaded the session state from disk and seamlessly executed the remaining sequence (`calculate_refund`, `issue_refund`).
- **Idempotency Guarantee**: Across crash and resumption, tools executed exactly in the sequence `['get_order', 'search_policy', 'calculate_refund', 'issue_refund']` with zero duplicate ledger writes, concluding with `REFUND: INR 128.25 issued.`

### ReAct vs Plan-and-Execute Architecture Comparison

The ReAct (Reason + Act) architecture dynamically interleaves thought, tool execution, and environmental observation. In customer support, this flexibility allows the agent to inspect intermediate order statuses and pivot immediately—for instance, terminating early when an order is cancelled or not found (Scenarios S5 and S6), without executing unnecessary subsequent tools. However, ReAct is inherently sequential: each tool call requires a separate model round-trip, increasing cumulative latency, token consumption, and exposure to infinite loops if queries fail to converge (demonstrated by Scenario S7).

In contrast, Plan-and-Execute architectures decompose user intent upfront into an explicit task graph (e.g., fetch order $\to$ verify policy $\to$ calculate refund $\to$ log transaction) before executing steps sequentially or in parallel. This drastically reduces model round-trips and eliminates runaway recursion. However, Plan-and-Execute systems are brittle to intermediate failures; if a tool returns an unexpected schema or missing entity, the pre-computed plan cannot dynamically adjust without an expensive re-planning cycle. For quick-commerce refund workflows, a hybrid pattern is optimal: deterministic rule-based pre-validation for database checks and policy calculation, paired with ReAct flexibility for conversational reasoning and human handoffs.

## Limitations and Next Steps

### Limitations

1. **Synthetic Data Boundaries**: The underlying transaction data follows simplified synthetic distributions. Real-world quick-commerce purchasing includes non-linear seasonality, promotional surges, and localized stockouts.
2. **Post-Hoc Segmentation Lookahead**: The K-means RFM segments were derived across the entire six-month window. While suitable for retrospective customer grouping and support routing demonstrations, using full-window segment labels as real-time predictive features would introduce lookahead bias.
3. **Lexical Retrieval Constraints**: TF-IDF retrieval depends strictly on term frequency and token matching. While 100% effective on keyword-aligned policy queries, it fails on semantic paraphrasing or synonyms (e.g., "reimbursement" vs "refund") without dense vector representations.
4. **Synchronous Single-Threaded Agent**: The ReAct agent executes tool calls sequentially, accumulating latency across multi-step resolutions.

### Next Steps

1. **Dense Vector RAG**: Migrate policy chunking to local dense bi-encoders (e.g. `sentence-transformers/all-MiniLM-L6-v2`) backed by a lightweight embedded vector index (such as FAISS or ChromaDB) for semantic retrieval.
2. **Dynamic Time-Window Features**: Implement rolling-window temporal features (e.g., 30-day and 60-day velocity metrics) computed strictly at point-in-time to support real-time churn alerts.
3. **Multi-Agent Orchestration**: Decompose the triage and refund workflow into specialized micro-agents (Triage Router, Policy Specialist, Refund Auditor) running with parallel tool calling and human-in-the-loop escalation dashboards.
4. **Automated Feedback Loop**: Log agent handovers (`human_review`, `security_team`) to continuously evaluate policy edge cases and refine few-shot demonstrations.
