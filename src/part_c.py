"""src/part_c.py - Support Ticket Triage with Guardrails for Instamart Copilot.

This script implements:
1. Ticket loading with keep_default_na=False.
2. Pre-call validation: empty/whitespace check (skipped_empty) and token length guardrail (flagged_long).
3. 3-stage PII redaction: Email -> Phone -> Order ID, with re-check assertion and PII leakage assertion.
4. Resilient LLM client with exponential backoff + jitter and retry logging.
5. Zero-shot and few-shot prompts and validation (classified_json, classified_keyword, flagged_invalid_output).
6. Independent zero-shot and few-shot runs with metrics and confusion matrix.
7. Token budget enforcement demonstration (outputs/triage_budget_demo.csv).
8. Production few-shot triage run with customer segment integration, business routing, priority tagging.
9. Exporting outputs/triage_results.csv (exactly 164 rows) and audit reports under outputs/.
"""

import os
import sys
import re
import json
import random
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import mock_llm

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
ALLOWED_SET = set(ALLOWED_CATEGORIES)

# Regex patterns in exact mandated order
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_PHONE = re.compile(r"\b[6-9]\d{9}\b")
RE_ORDER = re.compile(r"ORD-\d{6}")


def redact_text(text):
    """Redacts email, phone, and order ID in strict sequential order."""
    emails = len(RE_EMAIL.findall(text))
    t1 = RE_EMAIL.sub("[EMAIL]", text)

    phones = len(RE_PHONE.findall(t1))
    t2 = RE_PHONE.sub("[PHONE]", t1)

    orders = len(RE_ORDER.findall(t2))
    t3 = RE_ORDER.sub("[ORDER_ID]", t2)

    return t3, emails, phones, orders


def call_llm_with_retries(messages, ticket_id, retry_log_collector=None, max_retries=3):
    """Executes mock_llm.chat with exponential backoff and jitter on TransientLLMError."""
    last_err = None
    for n in range(max_retries):
        try:
            resp = mock_llm.chat(messages)
            return resp
        except mock_llm.TransientLLMError as err:
            last_err = err
            wait_time = 0.05 * (2 ** n) * random.uniform(0.9, 1.15)
            log_line = f"RETRY: ticket_id={ticket_id}, retry={n+1}/{max_retries}, wait={wait_time:.4f}s, reason={err}"
            print(log_line)
            if retry_log_collector is not None:
                retry_log_collector.append(log_line)
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {max_retries} retries for ticket {ticket_id}: {last_err}")


def validate_model_output(raw_content):
    """Validates model output adhering to JSON contract and fallback keyword rules."""
    # Attempt 1: Strict JSON parsing
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict):
            cat = data.get("category")
            if cat in ALLOWED_SET:
                return "classified_json", cat
    except Exception:
        pass

    # Attempt 2: Fallback keyword scan using lowercase letters
    words = re.findall(r"[a-z]+", (raw_content or "").lower())
    matched_cats = [w for w in words if w in ALLOWED_SET]
    unique_matched = set(matched_cats)

    if len(unique_matched) == 1:
        return "classified_keyword", matched_cats[0]

    return "flagged_invalid_output", None


def run_part_c():
    os.makedirs("outputs", exist_ok=True)
    metrics = {}
    retry_logs = []

    print("=================================================================")
    print("PART C.1 & C.2: LOAD TICKETS AND AUDIT")
    print("=================================================================")

    tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    total_tickets = len(tickets_df)
    print(f"Loaded {total_tickets} tickets from data/raw/tickets.csv")
    assert total_tickets == 164, f"Expected 164 tickets, found {total_tickets}"

    # Guardrail checks before any LLM call
    is_empty = tickets_df["text"].str.strip() == ""
    ticket_token_counts = tickets_df["text"].apply(mock_llm.est_tokens)
    is_long = (~is_empty) & (ticket_token_counts > 400)

    empty_count = int(is_empty.sum())
    long_count = int(is_long.sum())
    long_tokens = ticket_token_counts[is_long].tolist()

    print(f"Empty tickets: {empty_count}")
    print(f"Long tickets (>400 tokens): {long_count}, Estimated tokens: {long_tokens}")

    assert empty_count == 4, f"Expected 4 empty tickets, got {empty_count}"
    assert long_count == 4, f"Expected 4 long tickets, got {long_count}"
    assert all(t == 734 for t in long_tokens), f"Expected long ticket tokens to be 734, got {long_tokens}"

    print("\n=================================================================")
    print("PART C.4: PII REDACTION AND ASSERTIONS")
    print("=================================================================")

    total_emails = 0
    total_phones = 0
    total_orders = 0
    redacted_texts = []

    for text in tickets_df["text"]:
        red_text, n_e, n_p, n_o = redact_text(text)
        total_emails += n_e
        total_phones += n_p
        total_orders += n_o
        redacted_texts.append(red_text)

    tickets_df["redacted_text"] = redacted_texts

    print(f"Redaction counts -> Emails: {total_emails}, Phones: {total_phones}, Order IDs: {total_orders}")
    total_redactions = total_emails + total_phones + total_orders
    print(f"Total Redactions: {total_redactions}")

    assert total_emails == 55, f"Expected 55 emails, got {total_emails}"
    assert total_phones == 91, f"Expected 91 phones, got {total_phones}"
    assert total_orders == 132, f"Expected 132 order IDs, got {total_orders}"
    assert total_redactions == 278, f"Expected 278 total redactions, got {total_redactions}"

    # Re-apply patterns to ensure 0 remaining matches
    for red_t in redacted_texts:
        assert len(RE_EMAIL.findall(red_t)) == 0, f"Leaked email in: {red_t}"
        assert len(RE_PHONE.findall(red_t)) == 0, f"Leaked phone in: {red_t}"
        assert len(RE_ORDER.findall(red_t)) == 0, f"Leaked order id in: {red_t}"
    print("Assertion passed: 0 PII matches found in redacted texts.")

    metrics["redaction"] = {
        "emails": total_emails,
        "phones": total_phones,
        "order_ids": total_orders,
        "total": total_redactions
    }

    print("\n=================================================================")
    print("PART C.5 - C.8: ZERO-SHOT AND FEW-SHOT BENCHMARK")
    print("=================================================================")

    prompt_zero_shot = (
        "CLASSIFY support tickets into billing, delivery, technical, fraud. "
        'Respond with JSON only: {"category":"...","reason":"..."}'
    )

    prompt_few_shot = (
        "CLASSIFY support tickets into billing, delivery, technical, fraud. "
        'Respond with JSON only: {"category":"...","reason":"..."}\n'
        "EXAMPLES:\n"
        'Input: There is an unauthorized transaction on my card for 1200 rupees\n'
        'Output: {"category":"fraud","reason":"unauthorized charge reported"}\n'
        'Input: The delivery rider marked package as delivered but I received nothing\n'
        'Output: {"category":"delivery","reason":"undelivered order"}\n'
        'Input: The mobile app crashes every time I tap payment checkout\n'
        'Output: {"category":"technical","reason":"app crash on payment"}\n'
        'Input: I was double billed on my bank statement for one order\n'
        'Output: {"category":"billing","reason":"duplicate charge"}'
    )

    benchmark_runs = {}

    for mode_name, sys_prompt in [("zero_shot", prompt_zero_shot), ("few_shot", prompt_few_shot)]:
        mock_llm.reset()
        status_counts = {}
        classified_records = []
        total_tokens_mode = 0

        for idx, row in tickets_df.iterrows():
            tid = row["ticket_id"]
            raw_t = row["text"]
            red_t = row["redacted_text"]

            if not raw_t.strip():
                status = "skipped_empty"
                cat = None
                tokens = 0
            elif mock_llm.est_tokens(raw_t) > 400:
                status = "flagged_long"
                cat = None
                tokens = 0
            else:
                # LLM call with redacted text only
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": red_t}
                ]
                resp = call_llm_with_retries(messages, tid, retry_log_collector=retry_logs)
                tokens = resp.usage.total_tokens
                total_tokens_mode += tokens
                content = resp.choices[0].message.content
                status, cat = validate_model_output(content)

            status_counts[status] = status_counts.get(status, 0) + 1
            classified_records.append({
                "ticket_id": tid,
                "customer_id": row["customer_id"],
                "true_category": row["true_category"],
                "status": status,
                "category": cat,
                "tokens_used": tokens
            })

        bench_df = pd.DataFrame(classified_records)
        classified_sub = bench_df[bench_df["status"].isin(["classified_json", "classified_keyword"])]
        acc = float(accuracy_score(classified_sub["true_category"], classified_sub["category"]))
        classified_count = len(classified_sub)
        tokens_per_ticket = round(total_tokens_mode / classified_count, 2) if classified_count > 0 else 0

        print(f"\nMode: {mode_name}")
        print(f"Status distribution: {status_counts}")
        print(f"Classified tickets: {classified_count}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"Total tokens: {total_tokens_mode} | Tokens per classified ticket: {tokens_per_ticket}")

        run_info = {
            "status_counts": status_counts,
            "classified_count": classified_count,
            "accuracy": round(acc, 4),
            "total_tokens": total_tokens_mode,
            "tokens_per_classified_ticket": tokens_per_ticket
        }

        if mode_name == "zero_shot":
            # Assertions for zero-shot run
            assert status_counts.get("classified_json") == 135, f"Expected 135 classified_json, got {status_counts.get('classified_json')}"
            assert status_counts.get("classified_keyword") == 12, f"Expected 12 classified_keyword, got {status_counts.get('classified_keyword')}"
            assert status_counts.get("flagged_invalid_output") == 9, f"Expected 9 flagged_invalid_output, got {status_counts.get('flagged_invalid_output')}"
            assert classified_count == 147, f"Expected 147 classified, got {classified_count}"
            assert round(acc, 3) == 0.857, f"Expected zero-shot accuracy ~85.7%, got {acc:.3f}"

            labels_cm = ["billing", "delivery", "fraud", "technical"]
            cm_zs = confusion_matrix(classified_sub["true_category"], classified_sub["category"], labels=labels_cm).tolist()
            print("Zero-shot confusion matrix (labels: billing, delivery, fraud, technical):")
            print(np.array(cm_zs))
            run_info["confusion_matrix_labels"] = labels_cm
            run_info["confusion_matrix"] = cm_zs
        elif mode_name == "few_shot":
            assert round(acc, 2) == 1.00, f"Expected few-shot accuracy ~100%, got {acc:.2f}"

        benchmark_runs[mode_name] = run_info

    metrics["benchmarks"] = benchmark_runs

    print("\n=================================================================")
    print("PART C.9: TOKEN BUDGET DEMONSTRATION")
    print("=================================================================")

    # Demonstrate budget cutoff
    mock_llm.reset()
    budget_limit = 6000
    accumulated_tokens = 0
    budget_records = []

    for idx, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        raw_t = row["text"]
        red_t = row["redacted_text"]

        if not raw_t.strip():
            status = "skipped_empty"
            cat = None
            t_used = 0
        elif mock_llm.est_tokens(raw_t) > 400:
            status = "flagged_long"
            cat = None
            t_used = 0
        elif accumulated_tokens >= budget_limit:
            status = "budget_exhausted"
            cat = None
            t_used = 0
        else:
            messages = [
                {"role": "system", "content": prompt_zero_shot},
                {"role": "user", "content": red_t}
            ]
            resp = call_llm_with_retries(messages, tid)
            t_used = resp.usage.total_tokens
            accumulated_tokens += t_used
            content = resp.choices[0].message.content
            status, cat = validate_model_output(content)

        budget_records.append({
            "ticket_id": tid,
            "customer_id": row["customer_id"],
            "status": status,
            "category": cat,
            "tokens_used": t_used,
            "accumulated_tokens": accumulated_tokens
        })

    budget_df = pd.DataFrame(budget_records)
    budget_counts = budget_df["status"].value_counts().to_dict()
    print(f"Budget={budget_limit} results: {budget_counts}")
    exhausted_count = budget_counts.get("budget_exhausted", 0)
    print(f"Tickets stopped due to budget exhaustion: {exhausted_count}")

    budget_df.to_csv("outputs/triage_budget_demo.csv", index=False)
    print("Saved outputs/triage_budget_demo.csv successfully.")

    metrics["budget_demo"] = {
        "budget_limit": budget_limit,
        "accumulated_tokens_at_end": accumulated_tokens,
        "exhausted_ticket_count": exhausted_count,
        "status_distribution": budget_counts
    }

    print("\n=================================================================")
    print("PART C.10: FINAL PRODUCTION TRIAGE RUN (BUDGET=100,000)")
    print("=================================================================")

    # Load segments from Part B
    segments_df = pd.read_csv("outputs/customer_segments.csv")
    segments_map = segments_df.set_index("customer_id")[["segment_name", "is_premium"]].to_dict(orient="index")

    mock_llm.reset()
    final_budget = 100000
    accumulated_tokens_final = 0
    triage_final_rows = []

    for idx, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_t = row["text"]
        red_t = row["redacted_text"]

        # Check customer segment
        seg_info = segments_map.get(cid, None)
        if seg_info is not None:
            seg_name = seg_info["segment_name"]
            is_prem = bool(seg_info["is_premium"])
        else:
            seg_name = "Not Segmented"
            is_prem = False

        if not raw_t.strip():
            status = "skipped_empty"
            cat = None
            t_used = 0
        elif mock_llm.est_tokens(raw_t) > 400:
            status = "flagged_long"
            cat = None
            t_used = 0
        elif accumulated_tokens_final >= final_budget:
            status = "budget_exhausted"
            cat = None
            t_used = 0
        else:
            messages = [
                {"role": "system", "content": prompt_few_shot},
                {"role": "user", "content": red_t}
            ]
            resp = call_llm_with_retries(messages, tid, retry_log_collector=retry_logs)
            t_used = resp.usage.total_tokens
            accumulated_tokens_final += t_used
            content = resp.choices[0].message.content
            status, cat = validate_model_output(content)

        # Routing rules:
        # skipped_empty, flagged_long, flagged_invalid_output, budget_exhausted -> human_review, NORMAL
        # fraud -> security_team, HIGH
        # otherwise Premium customer -> premium_desk, HIGH
        # otherwise support_queue, NORMAL
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route = "human_review"
            priority = "NORMAL"
        elif cat == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_prem:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        triage_final_rows.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat if cat is not None else "",
            "segment_name": seg_name,
            "priority": priority,
            "route": route,
            "tokens_used": t_used
        })

    triage_results_df = pd.DataFrame(triage_final_rows)
    assert len(triage_results_df) == 164, f"Expected exactly 164 rows, got {len(triage_results_df)}"
    assert list(triage_results_df.columns) == [
        "ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"
    ], f"Column mismatch in triage_results.csv: {list(triage_results_df.columns)}"

    triage_results_df.to_csv("outputs/triage_results.csv", index=False)
    print("Saved outputs/triage_results.csv (exactly 164 rows).")

    route_counts = triage_results_df["route"].value_counts().to_dict()
    priority_counts = triage_results_df["priority"].value_counts().to_dict()
    status_final_counts = triage_results_df["status"].value_counts().to_dict()
    print(f"Final Route Counts: {route_counts}")
    print(f"Final Priority Counts: {priority_counts}")
    print(f"Final Status Counts: {status_final_counts}")

    metrics["final_few_shot_triage"] = {
        "total_tickets": len(triage_results_df),
        "total_tokens_used": accumulated_tokens_final,
        "status_distribution": status_final_counts,
        "route_distribution": route_counts,
        "priority_distribution": priority_counts
    }

    # Save metrics and retry log
    with open("outputs/triage_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("outputs/triage_retry_log.txt", "w") as f:
        f.write("\n".join(retry_logs) + "\n")

    print("Saved outputs/triage_metrics.json and outputs/triage_retry_log.txt.")
    print("ALL PART C DELIVERABLES COMPLETED.")


if __name__ == "__main__":
    run_part_c()
