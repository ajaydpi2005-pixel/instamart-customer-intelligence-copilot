"""src/part_d.py - Policy RAG Assistant and Resilient Refund Agent for Instamart Copilot.

This script implements:
1. Deterministic text chunking of 5 policy documents (40-word chunks, overlap 10, step 30).
2. TF-IDF vectorization and cosine similarity retrieval (k=3).
3. Grounded question answering using mock_llm.chat under POLICY ASSISTANT role.
4. Evaluation of policy_eval.csv (overlap 10 vs overlap 0), reporting retrieval hits, fact hits, and broken questions.
5. Five registered tools: get_order, search_policy, calculate_refund, issue_refund, escalate_to_human.
6. ReAct agent loop with max_iterations, token_limit, and timeout guardrails.
7. Evaluation across all 7 scenarios in agent_scenarios.csv, exported to outputs/agent_runs.jsonl.
8. Safe stop demonstrations (token limit cutoff and timeout).
9. Persistent checkpointing in state/ and crash recovery via resume_agent(run_id).
10. Metric exports to outputs/part_d_metrics.json.
"""

import os
import sys
import glob
import json
import time
import uuid
import datetime
import sqlite3

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


# =====================================================================
# 1. POLICY CHUNKING & RETRIEVAL (RAG)
# =====================================================================

def chunk_policy_documents(overlap=10):
    """Chunks policy documents into 40-word chunks with specified overlap.
    Step = 40 - overlap.
    """
    step = 40 - overlap
    policy_files = sorted(glob.glob("data/policies/*.txt"))
    chunks = []

    for fpath in policy_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            words = f.read().split()

        i = 0
        while i < len(words):
            c = words[i : i + 40]
            chunks.append({
                "source": fname,
                "text": " ".join(c)
            })
            if i + 40 >= len(words):
                break
            i += step

    return chunks


class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.chunk_texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(sims[i]),
                "text": self.chunks[i]["text"]
            }
            for i in top_indices
        ]

    def answer(self, question, k=3):
        retrieved = self.retrieve(question, k=k)
        context_str = "\n".join(f"[{item['source']}]\n{item['text']}" for item in retrieved)
        user_message_content = f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"

        messages = [
            {"role": "system", "content": "POLICY ASSISTANT. Answer only using the context provided. If not in context say you cannot find it."},
            {"role": "user", "content": user_message_content}
        ]
        resp = mock_llm.chat(messages)
        final_answer = resp.choices[0].message.content
        sources = list(dict.fromkeys(item["source"] for item in retrieved))
        return final_answer, sources


# =====================================================================
# 2. TOOL DEFINITIONS & LEDGER
# =====================================================================

DB_PATH = "data/instamart.db"
LEDGER_PATH = "outputs/refund_ledger.jsonl"


def get_order(order_id):
    """Queries orders table in SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"error": f"Order {order_id} not found."}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": float(row[3]) if row[3] is not None else None
    }


def search_policy(query):
    """Retrieves top grounded chunks for a policy query."""
    retriever = PolicyRetriever(chunk_policy_documents(overlap=10))
    results = retriever.retrieve(query, k=3)
    return {"query": query, "top_matches": [r["text"] for r in results]}


def calculate_refund(order_amount, delivery_minutes):
    """Calculates refund based on late delivery threshold of >45 minutes."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def issue_refund(order_id, refund_inr):
    """Writes an approved refund transaction to the ledger."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "success", "order_id": order_id, "refund_inr": float(refund_inr)}


def escalate_to_human(order_id, refund_inr):
    """Escalates high-value refund to a human support lead and logs to ledger."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "escalated", "order_id": order_id, "refund_inr": float(refund_inr)}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Fetch order details from SQLite database"},
    {"name": "search_policy", "description": "Search customer refund and service policies"},
    {"name": "calculate_refund", "description": "Compute eligible refund amount based on delivery time"},
    {"name": "issue_refund", "description": "Execute automated refund for eligible orders up to INR 1000"},
    {"name": "escalate_to_human", "description": "Escalate refund above INR 1000 to human support lead"}
]


# =====================================================================
# 3. AGENT RUNNER & RESUMABILITY
# =====================================================================

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, tools_executed, steps):
    os.makedirs("state", exist_ok=True)
    state_payload = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_num": step_num,
        "tools_executed": tools_executed,
        "steps": steps
    }
    with open(f"state/{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(state_payload, f, indent=2)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool_count=None):
    """Executes the ReAct refund agent loop with safety limits and checkpointing."""
    start_time = time.monotonic()
    run_id = run_id or f"run_{scenario_id}_{uuid.uuid4().hex[:6]}"

    messages = [
        {"role": "system", "content": "AGENT for customer refund evaluation and policy enforcement."},
        {"role": "user", "content": user_message}
    ]
    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []
    step_num = 0

    for iteration in range(max_iterations):
        # Time guardrail check at top of iteration
        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": f"Handover to human agent: operation timed out after {llm_calls} LLM calls.",
                "steps": steps,
                "tools_executed": tools_executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        choice = resp.choices[0]

        # Case 1: Final natural language response
        if not choice.message.tool_calls:
            ans = choice.message.content
            if "REFUND:" in ans:
                status = "REFUND"
            elif "ESCALATE:" in ans:
                status = "ESCALATE"
            elif "DENIED:" in ans:
                status = "DENIED"
            elif "NOT_FOUND:" in ans:
                status = "NOT_FOUND"
            else:
                status = "COMPLETED"

            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": ans,
                "steps": steps,
                "tools_executed": tools_executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        # Case 2: Tool call request
        tc = choice.message.tool_calls[0]

        # Token limit guardrail check before executing requested tool
        if tokens_used > token_limit:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": f"Handover to human agent: token budget exceeded ({tokens_used} > {token_limit}).",
                "steps": steps,
                "tools_executed": tools_executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        # Append assistant tool call message
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })

        # Execute tool
        tool_fn = TOOL_MAP[tc.name]
        res = tool_fn(**tc.arguments)
        steps.append({"tool": tc.name, "args": tc.arguments, "result": res})
        tools_executed.append(tc.name)
        step_num += 1

        # Append tool observation
        messages.append({
            "role": "tool",
            "name": tc.name,
            "tool_call_id": tc.id,
            "content": json.dumps(res)
        })

        # Save complete checkpoint under state/
        save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, tools_executed, steps)

        # Simulated crash demonstration hook
        if crash_after_tool_count and len(tools_executed) >= crash_after_tool_count:
            raise RuntimeError(f"Simulated process crash after {len(tools_executed)} tool executions.")

    # Exceeded max iterations fallback
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": f"Handover to human agent: resolution exceeded {max_iterations} iteration ceiling.",
        "steps": steps,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Resumes an agent session from a state/ checkpoint file."""
    start_time = time.monotonic()
    checkpoint_file = f"state/{run_id}.json"
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found.")

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    scenario_id = state["scenario_id"]
    messages = state["messages"]
    tokens_used = state["tokens_used"]
    step_num = state["step_num"]
    tools_executed = state["tools_executed"]
    steps = state["steps"]
    llm_calls = len([m for m in messages if m.get("role") == "assistant"])

    while len(tools_executed) < max_iterations:
        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": f"Handover to human agent: operation timed out after resume ({llm_calls} calls).",
                "steps": steps,
                "tools_executed": tools_executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        choice = resp.choices[0]

        if not choice.message.tool_calls:
            ans = choice.message.content
            if "REFUND:" in ans:
                status = "REFUND"
            elif "ESCALATE:" in ans:
                status = "ESCALATE"
            elif "DENIED:" in ans:
                status = "DENIED"
            elif "NOT_FOUND:" in ans:
                status = "NOT_FOUND"
            else:
                status = "COMPLETED"

            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": ans,
                "steps": steps,
                "tools_executed": tools_executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        tc = choice.message.tool_calls[0]
        if tokens_used > token_limit:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": f"Handover to human agent: token limit exceeded ({tokens_used} > {token_limit}).",
                "steps": steps,
                "tools_executed": tools_executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })

        tool_fn = TOOL_MAP[tc.name]
        res = tool_fn(**tc.arguments)
        steps.append({"tool": tc.name, "args": tc.arguments, "result": res})
        tools_executed.append(tc.name)
        step_num += 1

        messages.append({
            "role": "tool",
            "name": tc.name,
            "tool_call_id": tc.id,
            "content": json.dumps(res)
        })

        save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, tools_executed, steps)

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": f"Handover to human agent: resolution exceeded {max_iterations} iteration ceiling.",
        "steps": steps,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


# =====================================================================
# 4. MAIN EXECUTION ROUTINE
# =====================================================================

def run_part_d():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("state", exist_ok=True)
    metrics = {}

    print("=================================================================")
    print("PART D.1 - D.4: POLICY RAG EVALUATION")
    print("=================================================================")

    # 1. Chunk documents with overlap 10
    chunks_10 = chunk_policy_documents(overlap=10)
    chunk_counts_by_doc = {}
    for c in chunks_10:
        chunk_counts_by_doc[c["source"]] = chunk_counts_by_doc.get(c["source"], 0) + 1

    print(f"Total chunks (overlap=10): {len(chunks_10)}")
    print(f"Chunk distribution: {chunk_counts_by_doc}")

    assert len(chunks_10) == 15, f"Expected 15 chunks, got {len(chunks_10)}"
    assert chunk_counts_by_doc.get("refund_policy.txt") == 4
    assert chunk_counts_by_doc.get("delivery_policy.txt") == 3
    assert chunk_counts_by_doc.get("escalation_policy.txt") == 3
    assert chunk_counts_by_doc.get("payment_policy.txt") == 3
    assert chunk_counts_by_doc.get("privacy_policy.txt") == 2
    print("Assertion passed: Exact chunk counts verified (15 total).")

    # 2. Retriever and query check
    retriever_10 = PolicyRetriever(chunks_10)
    test_q_top3 = retriever_10.retrieve("refund policy late delivery", k=3)
    test_q_sources = [r["source"] for r in test_q_top3]
    print(f"Top 3 sources for 'refund policy late delivery': {test_q_sources}")
    assert test_q_sources == ["refund_policy.txt", "refund_policy.txt", "refund_policy.txt"]

    # 4. Evaluate policy_eval.csv on overlap 10
    eval_df = pd.read_csv("data/raw/policy_eval.csv")
    top3_hits_10 = 0
    fact_hits_10 = 0
    fallback_hits_10 = 0

    eval_results_10 = []
    for idx, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]
        ans, sources = retriever_10.answer(q, k=3)

        if exp_doc == "NONE":
            is_fb = ans.startswith("I could not find")
            if is_fb:
                fallback_hits_10 += 1
            eval_results_10.append({"question": q, "status": "unanswerable", "fallback_correct": is_fb, "answer": ans})
        else:
            doc_hit = exp_doc in sources
            fact_hit = exp_fact in ans
            if doc_hit:
                top3_hits_10 += 1
            if fact_hit:
                fact_hits_10 += 1
            eval_results_10.append({"question": q, "doc_hit": doc_hit, "fact_hit": fact_hit, "answer": ans})

    print(f"Overlap 10 Evaluation: Top-3 Doc Hits: {top3_hits_10}/12, Fact Hits: {fact_hits_10}/12, Fallback Hits: {fallback_hits_10}/2")
    assert top3_hits_10 == 12, f"Expected 12/12 doc hits, got {top3_hits_10}"
    assert fact_hits_10 == 12, f"Expected 12/12 fact hits, got {fact_hits_10}"
    assert fallback_hits_10 == 2, f"Expected 2/2 fallback hits, got {fallback_hits_10}"

    # Repeat with overlap 0
    chunks_0 = chunk_policy_documents(overlap=0)
    retriever_0 = PolicyRetriever(chunks_0)
    top3_hits_0 = 0
    fact_hits_0 = 0
    fallback_hits_0 = 0
    broken_questions_0 = []

    for idx, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]
        ans, sources = retriever_0.answer(q, k=3)

        if exp_doc == "NONE":
            if ans.startswith("I could not find"):
                fallback_hits_0 += 1
        else:
            doc_hit = exp_doc in sources
            fact_hit = exp_fact in ans
            if doc_hit:
                top3_hits_0 += 1
            if fact_hit:
                fact_hits_0 += 1
            else:
                broken_questions_0.append({
                    "question": q,
                    "expected_doc": exp_doc,
                    "expected_fact": exp_fact,
                    "returned_answer": ans
                })

    print(f"Overlap 0 Evaluation: Top-3 Doc Hits: {top3_hits_0}/12, Fact Hits: {fact_hits_0}/12, Fallback Hits: {fallback_hits_0}/2")
    print(f"Broken questions with Overlap 0: {len(broken_questions_0)}")
    for bq in broken_questions_0:
        print(f"  Question: '{bq['question']}' | Missing Fact: '{bq['expected_fact']}'")

    metrics["rag_evaluation"] = {
        "overlap_10": {
            "chunk_count": len(chunks_10),
            "distribution": chunk_counts_by_doc,
            "top3_doc_hits": f"{top3_hits_10}/12",
            "fact_hits": f"{fact_hits_10}/12",
            "fallback_hits": f"{fallback_hits_10}/2"
        },
        "overlap_0": {
            "chunk_count": len(chunks_0),
            "top3_doc_hits": f"{top3_hits_0}/12",
            "fact_hits": f"{fact_hits_0}/12",
            "fallback_hits": f"{fallback_hits_0}/2",
            "broken_questions": broken_questions_0
        }
    }

    print("\n=================================================================")
    print("PART D.5 - D.7: RUN 7 AGENT SCENARIOS")
    print("=================================================================")

    # Reset refund ledger for clean run
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)

    scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv")
    agent_run_records = []
    scenarios_summary = []

    for idx, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_out = row["expected_outcome"]
        exp_amt = row["expected_refund_inr"]

        res = run_agent(msg, sid)
        agent_run_records.append(res)
        scenarios_summary.append({
            "scenario_id": sid,
            "status": res["status"],
            "expected_outcome": exp_out,
            "expected_refund_inr": exp_amt,
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "tools_executed": res["tools_executed"]
        })

        print(f"{sid}: Status={res['status']} | LLM Calls={res['llm_calls']} | Tokens={res['tokens_used']} | Answer='{res['final_answer']}'")

    # Assertions for scenario outcomes
    s_map = {item["scenario_id"]: item for item in scenarios_summary}
    assert s_map["S1"]["status"] == "REFUND" and "128.25" in s_map["S1"]["final_answer"]
    assert s_map["S2"]["status"] == "REFUND" and "135" in s_map["S2"]["final_answer"]
    assert s_map["S3"]["status"] == "ESCALATE" and "1000" in s_map["S3"]["final_answer"]
    assert s_map["S4"]["status"] == "DENIED" and "45 minutes" in s_map["S4"]["final_answer"]
    assert s_map["S5"]["status"] == "DENIED" and "completed" in s_map["S5"]["final_answer"]
    assert s_map["S6"]["status"] == "NOT_FOUND"
    assert s_map["S7"]["status"] == "FALLBACK_MAX_ITERATIONS" and s_map["S7"]["llm_calls"] == 6

    # Save outputs/agent_runs.jsonl
    with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as f:
        for r in agent_run_records:
            f.write(json.dumps(r) + "\n")
    print("Saved outputs/agent_runs.jsonl successfully.")

    print("\n=================================================================")
    print("PART D.8: SAFE STOPS DEMONSTRATION")
    print("=================================================================")

    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].iloc[0]

    # Token limit stop
    res_token_stop = run_agent(s1_msg, "S1", token_limit=200)
    print(f"Token Stop Test: Status={res_token_stop['status']} | Tokens={res_token_stop['tokens_used']} | Answer='{res_token_stop['final_answer']}'")
    assert res_token_stop["status"] == "FALLBACK_TOKEN_LIMIT"

    # Timeout stop
    res_timeout_stop = run_agent(s1_msg, "S1", time_limit_seconds=0)
    print(f"Timeout Stop Test: Status={res_timeout_stop['status']} | Answer='{res_timeout_stop['final_answer']}'")
    assert res_timeout_stop["status"] == "FALLBACK_TIMEOUT"

    metrics["safe_stops"] = {
        "token_limit_demo": {
            "token_limit": 200,
            "status": res_token_stop["status"],
            "tokens_used": res_token_stop["tokens_used"],
            "final_answer": res_token_stop["final_answer"]
        },
        "timeout_demo": {
            "time_limit_seconds": 0,
            "status": res_timeout_stop["status"],
            "final_answer": res_timeout_stop["final_answer"]
        }
    }

    print("\n=================================================================")
    print("PART D.9: RESUMABILITY AND CRASH RECOVERY")
    print("=================================================================")

    crash_run_id = f"crash_demo_{uuid.uuid4().hex[:6]}"
    crash_occurred = False
    try:
        run_agent(s1_msg, "S1", run_id=crash_run_id, crash_after_tool_count=2)
    except RuntimeError as err:
        crash_occurred = True
        print(f"Successfully simulated crash: {err}")

    assert crash_occurred, "Simulated crash failed to trigger!"
    assert os.path.exists(f"state/{crash_run_id}.json"), "Checkpoint state file missing after crash!"

    with open(f"state/{crash_run_id}.json", "r", encoding="utf-8") as f:
        pre_resume_state = json.load(f)
    print(f"Pre-resume checkpoint saved tools: {pre_resume_state['tools_executed']}")
    assert pre_resume_state["tools_executed"] == ["get_order", "search_policy"]

    # Resume from checkpoint
    resumed_res = resume_agent(crash_run_id)
    print(f"Resumed run result: Status={resumed_res['status']} | Answer='{resumed_res['final_answer']}'")
    print(f"Tools executed in post-resume session: {resumed_res['tools_executed']}")

    # Across crash and resume, tool executions must be exactly get_order, search_policy, calculate_refund, issue_refund
    assert resumed_res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed_res["status"] == "REFUND" and "128.25" in resumed_res["final_answer"]
    print("Assertion passed: Exact tool sequence executed across crash and resume with zero duplication!")

    metrics["resumability"] = {
        "run_id": crash_run_id,
        "pre_crash_tools": pre_resume_state["tools_executed"],
        "post_resume_tools": resumed_res["tools_executed"],
        "final_status": resumed_res["status"],
        "final_answer": resumed_res["final_answer"]
    }

    metrics["scenarios_summary"] = scenarios_summary

    with open("outputs/part_d_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved outputs/part_d_metrics.json.")
    print("ALL PART D DELIVERABLES COMPLETED.")


if __name__ == "__main__":
    run_part_d()

