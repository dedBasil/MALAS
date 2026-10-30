"""MALAS backend: FastAPI app that runs the LangGraph pipeline for the Streamlit frontend.

    POST /applications/stream   body: form JSON -> one JSON line per agent as it finishes, then the result
    GET  /cases/{case_id}       -> result JSON of a past application (from the audit log)
    GET  /audit?case_id=&agent= -> audit rows (every agent action of every application)

Run from this folder:  uvicorn backend:app --port 8010      (interactive API docs at /docs)

The form is mapped onto the Lending Club schema the models were trained on. Fields the form does not ask
for are filled with portfolio-typical values (see to_lending_club), so scores are indicative, not exact.
shortcut: no login, and /cases and /audit return applicant data, so only run this on your own machine.
"""
import json
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import audit
import risk
from pipeline import stream_run

PURPOSE = {"Debt consolidation": "debt_consolidation", "Home improvement": "home_improvement", "Car": "car",
           "Medical": "medical", "Business": "small_business", "Education": "other", "Other": "other"}
HOME = {"Rent": "RENT", "Own": "OWN", "Mortgage": "MORTGAGE"}
OUTCOME = {"APPROVE": "Approved", "MANUAL_REVIEW": "Manual review", "REJECT": "Rejected"}
AGENT = {"intake": ("Intake", "validated_input"), "risk": ("Risk", "scored"), "fraud": ("Fraud", "llm_review"),
         "compliance": ("Compliance", "rule_checks"), "decision": ("Decision", "decided"),
         "decision_llm": ("Decision", "llm_opinion")}
NOT_USED = ["full_name", "age", "employment_status", "employment_years", "credit_history_years", "open_accounts"]


@asynccontextmanager
async def lifespan(app):
    # Load the models and portfolio data once at startup, so the first application is not slower.
    risk._approval_model()
    risk._schema_models()
    yield


app = FastAPI(title="MALAS", description="Multi-Agent Loan Approval System", lifespan=lifespan)


# --------------------------------------------------------------------------
# Form -> pipeline input
# --------------------------------------------------------------------------
@lru_cache
def _portfolio_norms():
    portfolio = risk._schema_models()[2]
    income = portfolio["annual_income"].where(portfolio["annual_income"] > 0)
    limit = portfolio["total_credit_limit"].where(portfolio["total_credit_limit"] > 0)
    return {"interest_rate": float(portfolio["interest_rate"].median()),
            "limit_to_income": float((portfolio["total_credit_limit"] / income).median()),
            "utilization": float((portfolio["total_credit_utilized"] / limit).median())}


def to_lending_club(p: dict) -> dict:
    """MALAS form -> Lending Club application. Assumed (form does not ask): income not verified, the portfolio's
    median interest rate for the installment, and a portfolio-typical credit limit and utilization for this income."""
    norms = _portfolio_norms()
    income, loan, months = p["annual_income"], p["loan_amount"], p["term_months"]
    i = norms["interest_rate"] / 100 / 12
    limit = round(norms["limit_to_income"] * income)
    return {
        "application_type": "individual",
        "annual_income": income,
        "verified_income": "Not Verified",
        "debt_to_income": round(p["monthly_debt"] * 12 / income * 100, 2) if income else None,
        "loan_amount": loan,
        "term": months,
        "installment": round(loan * i / (1 - (1 + i) ** -months), 2),
        "total_credit_limit": limit,
        "total_credit_utilized": round(limit * norms["utilization"]),
        "delinq_2y": p["delinquencies"],
        # The form counts delinquencies in the last 2 years but not when; without a date the model reads the
        # combination as a data inconsistency, so assume the midpoint of that window.
        "months_since_last_delinq": 12 if p["delinquencies"] else None,
        "homeownership": HOME.get(p["home_ownership"]),
        "loan_purpose": PURPOSE.get(p["purpose"], "other"),
    }


# --------------------------------------------------------------------------
# Pipeline output -> result JSON for the frontend
# --------------------------------------------------------------------------
def _rule_detail(rule: str, r: dict) -> tuple:
    used, limit = r.get("total_credit_utilized"), r.get("total_credit_limit")
    return {
        "max_debt_to_income": ("DTI at most 43%", f"DTI = {r.get('debt_to_income')}%"),
        "income_verification_required": ("Verified income for loans over 100,000",
                                         f"Amount = {r.get('loan_amount'):,.0f}, income {r.get('verified_income')}"),
        "max_credit_utilization": ("Credit utilization at most 100%",
                                   f"Utilization = {used / limit:.0%}" if limit else "No credit limit"),
        "bankruptcy_manual_review": ("No public-record bankruptcy", f"Bankruptcies = {r.get('public_record_bankrupt')}"),
    }.get(rule, (rule.replace("_", " "), ""))


def _summary(agent: str, o: dict) -> str:
    if o.get("status") == "error":
        return f"error: {o.get('error')}"
    return {
        "intake": lambda: f"dataset={o['dataset']}, {len(o['issues'])} issue(s)",
        "risk": lambda: f"score = {o.get('risk_score')}, PD = {o.get('p_default')}",
        "fraud": lambda: f"{o['recommendation']}, {len(o['red_flags'])} flag(s)",
        "compliance": lambda: f"{sum(r['passed'] is True for r in o['rules'])}/{len(o['rules'])} passed",
        "decision": lambda: o["final_decision"],
        "decision_llm": lambda: f"{o['final_decision']} (confidence {o['confidence']})",
    }[agent]()


def _notes(agent: str, o: dict) -> list:
    """The few lines worth showing live under each agent."""
    if o.get("status") == "error":
        return [str(o.get("error"))]
    if agent == "intake":
        return [f"{i['field']}: {i['message']}" for i in o["issues"]]
    if agent == "risk":
        return [f"flag: {f}" for f in o.get("flags", [])]
    if agent == "fraud":
        return o["red_flags"] or [o["explanation"]]
    if agent == "compliance":
        return o["adverse_action_reasons"]
    if agent == "decision":
        return [r.split("] ", 1)[-1] for r in o["reasons"]]
    if agent == "decision_llm":
        return [o.get("reasoning", "")]
    return []


def to_result(record: dict) -> dict:
    """One audit record -> the result JSON the frontend renders."""
    out = {s["agent"]: s["output"] for s in record["steps"]}
    intake = out["intake"]
    res = {"case_id": record["audit_id"], "created_at": record["started_at"],
           "intake": {"status": "Validated" if intake["valid"] else "Incomplete",
                      "issues": [f"{i['field']}: {i['message']}" for i in intake["issues"]],
                      "normalized": {**intake["record"], "form_fields_not_used_by_models": NOT_USED}},
           "audit": [{"timestamp": s["finished_at"], "agent": AGENT[s["agent"]][0], "action": AGENT[s["agent"]][1],
                      "detail": _summary(s["agent"], s["output"])} for s in record["steps"]]}

    r = out.get("risk", {})
    if r.get("status") == "ok" and "top_factors" in r:
        band = "High" if r["risk_score"] >= 90 else "Medium" if r["risk_score"] >= 70 else "Low"
        res["risk"] = {"default_probability": r["p_default"], "band": band, "top_factors": r["top_factors"]}
    else:
        res["risk"] = {"default_probability": 0.0, "band": "High",
                       "top_factors": [{"factor": r.get("error", "not scored"), "impact": 0}]}

    f = out.get("fraud", {})
    if not f:  # incomplete application: intake stopped the pipeline before fraud ran
        res["fraud"] = {"status": "Not run", "flags": []}
    else:
        res["fraud"] = ({"status": "Clear" if f["recommendation"] == "APPROVE" else "Flagged", "flags": f["red_flags"]}
                        if f.get("status") == "ok" else {"status": "Flagged", "flags": [f"Fraud check unavailable: {f.get('error')}"]})

    c = out.get("compliance")
    rules = []
    for rule in (c or {}).get("rules", []):
        name, detail = _rule_detail(rule["rule"], intake["record"])
        rules.append({"rule": name, "passed": rule["passed"] is True,
                      "detail": detail if rule["passed"] is not None else "could not be evaluated"})
    res["compliance"] = {"passed": bool(c and c["compliant"] and all(x["passed"] for x in rules)), "rules": rules}

    d, llm = out.get("decision", {}), out.get("decision_llm", {})
    final = d.get("final_decision", "MANUAL_REVIEW")
    conflicts = []
    views = [("Fraud agent", f.get("recommendation"))]
    if c:
        views.append(("Compliance agent", max(c["rule_severity"].values(), key=["APPROVE", "MANUAL_REVIEW", "REJECT"].index,
                                              default="APPROVE")))
    for who, verdict in views:
        if verdict and verdict != final:
            conflicts.append(f"{who} alone would say {OUTCOME[verdict]}; the strictest signal wins ({OUTCOME[final]}).")
    if llm.get("status") == "ok" and llm["final_decision"] != final:
        conflicts.append(f"LLM opinion is {OUTCOME[llm['final_decision']]}; the rule decision stays official.")
    reasons = ". ".join(x.split("] ", 1)[-1].rstrip(".") for x in d.get("reasons", [])) + "." if d.get("reasons") else "No issues found by any agent."
    res["decision"] = {"outcome": OUTCOME[final], "conflicts": conflicts,
                       "rationale": f"{reasons} {llm['reasoning']}" if llm.get("status") == "ok" else reasons}
    if llm.get("status") == "ok":
        res["decision"]["confidence"] = llm["confidence"]
    return res


def _find(case_id: str) -> dict:
    records = audit.find(case_id)
    if not records:
        raise HTTPException(404, f"Case {case_id} not found")
    return records[-1]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.post("/applications/stream")
def submit_stream(payload: dict = Body(...)):
    """Plain def, so FastAPI runs the blocking LLM calls on a worker thread."""
    try:
        application = to_lending_club(payload)
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"Missing or invalid form field: {e}")
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"

    def events():
        yield json.dumps({"type": "start", "case_id": case_id}) + "\n"
        try:
            for agent, out in stream_run(application, audit_id=case_id):
                yield json.dumps({"type": "step", "agent": agent, "detail": _summary(agent, out),
                                  "notes": _notes(agent, out)}, default=str) + "\n"
            yield json.dumps({"type": "result", "result": to_result(_find(case_id))}, default=str) + "\n"
        except Exception as e:  # the audit record is already written by stream_run
            yield json.dumps({"type": "error", "error": repr(e)}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    return to_result(_find(case_id))


@app.get("/audit")
def get_audit(case_id: str = "", agent: str = ""):
    records = audit.find(case_id) if case_id else audit.find_all()
    rows = [{"case_id": r["audit_id"], **row} for r in records for row in to_result(r)["audit"]]
    return [row for row in rows if not agent or row["agent"] == agent]
