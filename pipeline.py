"""The LangGraph pipeline. Try it without the web app:  python pipeline.py [submission.json]"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import audit
import compliance
import decision
import fraud
import intake
import llm
import risk

SAMPLES = {
    "approval_with_aliases": {
        "Application ID": "A-1", "dependents": 2, "education": "graduate", "self_employed": "No",
        "income": 9600000, "loan_amount": 29900000, "tenure": 12, "credit_score": 778,
        "home_value": 2400000, "business_assets": 17600000, "luxury_assets_value": 22700000,
        "bank_asset_value": 8000000,
    },
    "schema_lending_club": {
        "application_type": "individual", "annual_income": 90000, "verified_income": "Verified",
        "dti": 18.01, "loan_amount": 28000, "installment": 652.53, "term": 60, "grade": "C",
        "interest_rate": 14.07, "credit_limit": 70795, "credit_used": 38767, "delinq_2y": 0,
        "months_since_last_delinq": 38, "inquiries_last_12m": 6, "accounts_opened_24m": 5,
        "account_never_delinq_percent": 92.9, "loan_purpose": "moving", "homeownership": "MORTGAGE",
    },
    "approval_missing_fields": {"credit_score": 550, "income": -100, "loan_amount": 500000},
    "approval_low_credit": {
        "no_of_dependents": 3, "education": "Graduate", "self_employed": "Yes", "income_annum": 4000000,
        "loan_amount": 14000000, "loan_term": 10, "cibil_score": 420, "residential_assets_value": 3000000,
        "commercial_assets_value": 2000000, "luxury_assets_value": 9000000, "bank_asset_value": 2500000,
    },
}


class State(TypedDict, total=False):
    raw: dict
    dataset: str | None
    llm_decision: bool
    intake: dict
    risk: dict
    compliance: dict
    fraud: dict
    decision: dict
    decision_llm: dict


def intake_node(s: State):
    return {"intake": intake.run(s["raw"], s.get("dataset"))}


def risk_node(s: State):
    return {"risk": risk.run(s["intake"])}


def compliance_node(s: State):
    return {"compliance": compliance.run(s["intake"])}


def fraud_node(s: State):
    return {"fraud": fraud.run(s["intake"], s["risk"])}


def decision_node(s: State):
    return {"decision": decision.run(s["intake"], s.get("risk", {}), s.get("compliance"), s.get("fraud"))}


def decision_llm_node(s: State):
    return {"decision_llm": decision.run_llm(s["intake"], s["risk"], s["compliance"], s["fraud"], s["decision"])}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("intake", intake_node), ("risk", risk_node), ("compliance", compliance_node),
                     ("fraud", fraud_node), ("decision", decision_node), ("decision_llm", decision_llm_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    # Invalid submissions skip scoring and go straight to the decision (MANUAL_REVIEW).
    g.add_conditional_edges("intake", lambda s: "risk" if s["intake"]["valid"] else "decision", ["risk", "decision"])
    g.add_edge("risk", "compliance")
    g.add_edge("risk", "fraud")  # compliance and fraud run in parallel
    g.add_edge(["compliance", "fraud"], "decision")  # waits for both
    g.add_conditional_edges("decision", lambda s: "decision_llm" if s.get("llm_decision") and s["intake"]["valid"] else END,
                            ["decision_llm", END])
    g.add_edge("decision_llm", END)
    return g.compile()


graph = build_graph()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def stream_run(raw: dict, dataset: str = None, llm_decision: bool = True, audit_id: str = None):
    """Yields (agent, output) as each agent finishes; always writes one audit record, even if an agent crashes."""
    audit_id = audit_id or uuid.uuid4().hex
    started_at, t0 = _now(), time.perf_counter()
    steps, outputs, error = [], {}, None
    try:
        for update in graph.stream({"raw": raw, "dataset": dataset, "llm_decision": llm_decision}, stream_mode="updates"):
            for node_output in update.values():
                for agent, out in node_output.items():
                    steps.append({"agent": agent, "finished_at": _now(),
                                  "elapsed_ms": round(1000 * (time.perf_counter() - t0)), "output": out})
                    outputs[agent] = out
                    yield agent, out
    except Exception as e:
        error = repr(e)
        raise
    finally:
        audit.write({
            "audit_id": audit_id,
            "pipeline": "MALAS",
            "application_id": outputs.get("intake", {}).get("application_id"),
            "dataset": outputs.get("intake", {}).get("dataset"),
            "started_at": started_at,
            "finished_at": _now(),
            "input": raw,
            "options": {"dataset": dataset, "llm_decision": llm_decision},
            "models": {"llm": llm.MODEL, "risk": outputs.get("risk", {}).get("model")},
            "steps": steps,
            "final_decision": outputs.get("decision", {}).get("final_decision"),
            "llm_decision": outputs.get("decision_llm", {}).get("final_decision"),
            "completed": error is None and "decision" in outputs,
            "error": error,
        })


def run(raw: dict, dataset: str = None, llm_decision: bool = True) -> dict:
    audit_id = uuid.uuid4().hex
    result = {agent: {k: v for k, v in out.items() if k != "llm_call"}
              for agent, out in stream_run(raw, dataset, llm_decision, audit_id)}
    return {"audit_id": audit_id, **result}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            submissions = {sys.argv[1]: json.load(f)}
    else:
        submissions = SAMPLES
    for name, raw in submissions.items():
        print(f"===== {name}")
        print(json.dumps(run(raw), indent=2, default=str))
