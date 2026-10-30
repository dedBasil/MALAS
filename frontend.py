"""
MALAS - Multi-Agent Loan Approval System
Streamlit frontend

Run from this folder (backend first, in its own terminal):
    uvicorn backend:app --port 8010
    streamlit run frontend.py

Two pages: "New application" (the form, prefilled from demo_applications/current.json; replace that
file to present a different applicant) and "Pipeline in action" (watch each agent run live, then the result).

Backend contract: POST /applications/stream  body: application JSON -> one JSON line per agent, then the result.
Turn on "Demo mode" in the sidebar to run without a backend.
"""

import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MALAS - Loan Approval", page_icon="🏦", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.5rem; max-width: 1100px;}

    /* Sidebar: logo, link-style navigation */
    .brand {display:flex; align-items:center; gap:10px; margin:-1.5rem 0 1rem;}
    .brand-mark {width:34px; height:34px; border-radius:8px; background:#4f46e5; color:#fff; font-weight:700;
                 display:flex; align-items:center; justify-content:center;}
    .brand-name {font-weight:700; line-height:1.1; color:#0f172a;}
    .brand-sub {font-size:.72rem; color:#64748b;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:4px;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:8px 14px; border-radius:8px; width:100%; margin:0;}
    [data-testid="stSidebar"] [role="radiogroup"] label > div > div:not([data-testid]) {display:none;}
    [data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {background:#eef2ff;}
    [data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {color:#4f46e5; font-weight:600;}

    /* Upload box: dashed drop zone with arrow, title, hint, then "Browse files" */
    [data-testid="stFileUploaderDropzone"] {display:flex; flex-direction:column; align-items:center; gap:8px;
        padding:20px; border:2px dashed #a5b4fc; border-radius:6px; background:#eef2ff;}
    [data-testid="stFileUploaderDropzone"]::before {content:"↑"; width:40px; height:40px; border-radius:50%;
        background:#e0e7ff; color:#4f46e5; font-size:1.3rem; font-weight:700;
        display:flex; align-items:center; justify-content:center;}
    [data-testid="stFileUploaderDropzoneInstructions"] {order:1; text-align:center; display:block !important;}
    [data-testid="stFileUploaderDropzoneInstructions"] span {display:none;}
    [data-testid="stFileUploaderDropzoneInstructions"]::before {content:"Drag & drop your application file here";
        display:block; font-weight:700; font-size:.95rem; color:#0f172a;}
    [data-testid="stFileUploaderDropzoneInstructions"]::after {content:"CSV, JSON, PDF or DOCX · up to 10 MB";
        display:block; font-size:.72rem; color:#64748b; margin-top:4px;}
    [data-testid="stFileUploaderDropzone"] > span {order:2; margin-top:14px;}
    [data-testid="stFileUploaderDropzone"] button {background:#fff;}
    [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {display:none;}
    [data-testid="stFileUploaderDropzone"] button p {font-size:0;}
    [data-testid="stFileUploaderDropzone"] button p::after {content:"Browse files"; font-size:.85rem;
        font-weight:600; color:#4f46e5;}

    /* Form: small grey labels, no +/- buttons on number fields */
    [data-testid="stWidgetLabel"] p {font-size:.78rem; color:#64748b;}
    [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display:none;}
    .section-title {font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:.35rem;}
    .or-divider {display:flex; align-items:center; gap:12px; color:#64748b; font-size:.78rem; margin:.25rem 0 .5rem;}
    .or-divider::before, .or-divider::after {content:""; flex:1; border-top:1px solid #e2e8f0;}
    .agent-card {border: 1px solid rgba(128,128,128,.3); border-radius: 12px;
                 padding: 14px 16px; height: 100%;}
    .agent-title {font-weight: 600; font-size: 1.05rem; margin-bottom: 4px;}
    .pill {display:inline-block; padding: 2px 10px; border-radius: 999px;
           font-size: .75rem; font-weight: 600;}
    .pill-ok   {background:#dcfce7; color:#166534;}
    .pill-warn {background:#fef9c3; color:#854d0e;}
    .pill-bad  {background:#fee2e2; color:#991b1b;}
    .pill-idle {background:#e5e7eb; color:#374151;}
    .flow {display:flex; align-items:center; justify-content:center; gap:10px;
           flex-wrap:wrap; margin: 6px 0 14px 0;}
    .flow-box {border:1.5px solid #4f46e5; background:#eef2ff; color:#3730a3;
               border-radius:10px; padding:8px 16px; font-weight:600;}
    .flow-par {border:1.5px dashed #a5b4fc; border-radius:12px; padding:6px 10px;
               display:flex; gap:8px; background:rgba(79,70,229,.04);}
    .flow-par .flow-box {border-color:#818cf8; background:#e0e7ff;}
    .flow-arrow {font-size:1.3rem; opacity:.6;}
    .flow-box.waiting {border:1.5px dashed rgba(128,128,128,.5); background:transparent; opacity:.55;}
    .flow-box.running {border-color:#4f46e5; background:#c7d2fe; animation: pulse 1s ease-in-out infinite;}
    .flow-box.skipped {border:1.5px dashed rgba(128,128,128,.5); background:transparent; opacity:.35;
                       text-decoration: line-through;}
    @keyframes pulse {50% {box-shadow: 0 0 0 6px rgba(79,70,229,.25);}}
    @media (prefers-reduced-motion: reduce) {.flow-box.running {animation: none;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# API layer
# --------------------------------------------------------------------------
def api_submit_stream(base_url: str, payload: dict):
    """Yields the backend's events as they arrive: start, one step per agent, then result (or error)."""
    with requests.post(f"{base_url}/applications/stream", json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                yield json.loads(line)


def api_submit(base_url: str, payload: dict) -> dict:
    """Runs the whole pipeline and returns only the final result (the classic page shows no live steps)."""
    for event in api_submit_stream(base_url, payload):
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if event["type"] == "result":
            return event["result"]
    raise RuntimeError("The pipeline finished without a result.")


# --------------------------------------------------------------------------
# Demo mode (fake backend so the UI can be developed standalone)
# --------------------------------------------------------------------------
def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def mock_submit(p: dict) -> dict:
    time.sleep(1.2)
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    dti = p["monthly_debt"] * 12 / max(p["annual_income"], 1) * 100
    pd_est = min(0.95, max(0.02, 0.05 + dti / 200 + p["delinquencies"] * 0.06
                           - p["employment_years"] * 0.005
                           + p["loan_amount"] / max(p["annual_income"], 1) * 0.15))
    risk_band = "Low" if pd_est < 0.15 else "Medium" if pd_est < 0.35 else "High"

    fraud_flags = []
    if p["loan_amount"] > p["annual_income"] * 0.8:
        fraud_flags.append("Loan amount is unusually large relative to income")
    if p["age"] < 21 and p["annual_income"] > 150000:
        fraud_flags.append("Income looks implausible for stated age")
    fraud_status = "Flagged" if fraud_flags else "Clear"

    rules = [
        ("Minimum age 18", p["age"] >= 18, f"Age = {p['age']}"),
        ("DTI below 45%", dti < 45, f"DTI = {dti:.1f}%"),
        ("Loan amount within product limit", p["loan_amount"] <= 50000, f"Amount = {p['loan_amount']:,}"),
        ("No more than 2 recent delinquencies", p["delinquencies"] <= 2, f"Delinquencies = {p['delinquencies']}"),
        ("Employment history 1+ year", p["employment_years"] >= 1, f"{p['employment_years']} yrs"),
    ]
    comp_pass = all(r[1] for r in rules)

    conflicts = []
    if risk_band == "Low" and fraud_flags:
        conflicts.append("Risk agent says low risk, but Fraud agent flagged the application.")
    if risk_band == "High" and comp_pass:
        conflicts.append("Compliance passed, but Risk agent estimates high default probability.")

    if fraud_flags or not comp_pass or risk_band == "High":
        outcome = "Rejected" if (not comp_pass or risk_band == "High") else "Manual review"
    else:
        outcome = "Approved"
    if fraud_flags and outcome == "Rejected":
        outcome = "Rejected"

    rationale = (
        f"Estimated default probability {pd_est:.1%} ({risk_band} risk). "
        f"Fraud checks: {fraud_status.lower()}. "
        f"Compliance: {'all rules passed' if comp_pass else 'one or more rules failed'}."
    )

    audit = [
        {"timestamp": _now(), "agent": "Intake", "action": "validated_input", "detail": "Fields cleaned and typed"},
        {"timestamp": _now(), "agent": "Risk", "action": "scored", "detail": f"PD = {pd_est:.3f}"},
        {"timestamp": _now(), "agent": "Fraud", "action": "rule_checks", "detail": f"{len(fraud_flags)} flag(s)"},
        {"timestamp": _now(), "agent": "Compliance", "action": "rule_checks",
         "detail": f"{sum(r[1] for r in rules)}/{len(rules)} passed"},
        {"timestamp": _now(), "agent": "Decision", "action": "decided", "detail": outcome},
    ]
    return {
        "case_id": case_id,
        "created_at": _now(),
        "intake": {"status": "Validated", "issues": [], "normalized": {**p, "dti_pct": round(dti, 1)}},
        "risk": {"default_probability": round(pd_est, 4), "band": risk_band,
                 "top_factors": [
                     {"factor": "Debt-to-income", "impact": round(dti / 100, 2)},
                     {"factor": "Delinquencies", "impact": round(p["delinquencies"] * 0.06, 2)},
                     {"factor": "Employment length", "impact": round(-p["employment_years"] * 0.005, 2)},
                 ]},
        "fraud": {"status": fraud_status, "flags": fraud_flags},
        "compliance": {"passed": comp_pass,
                       "rules": [{"rule": a, "passed": b, "detail": c} for a, b, c in rules]},
        "decision": {"outcome": outcome, "rationale": rationale, "conflicts": conflicts,
                     "confidence": round(random.uniform(0.7, 0.95), 2)},
        "audit": audit,
    }


DEMO_FILE = Path(__file__).resolve().parent / "demo_applications" / "current.json"
FORM_DEFAULTS = dict(full_name="Jane Doe", age=34, home_ownership="Rent", employment_status="Employed",
                     employment_years=5.0, annual_income=60000, loan_amount=15000, term_months=36,
                     purpose="Debt consolidation", monthly_debt=800, credit_history_years=8, open_accounts=4,
                     delinquencies=0)
BOUNDS = dict(age=(18, 100), employment_years=(0.0, 60.0), annual_income=(0, 5_000_000), loan_amount=(500, 500_000),
              monthly_debt=(0, 100_000), credit_history_years=(0, 60), open_accounts=(0, 50), delinquencies=(0, 20))


def read_upload(file) -> dict:
    """Uploaded JSON/CSV -> form values. Only known fields are taken, and numbers must fit the form's limits."""
    name = file.name.lower()
    if not name.endswith((".json", ".csv")):
        raise ValueError("PDF and DOCX files are not supported yet. Upload a JSON or CSV file.")
    if name.endswith(".json"):
        data = json.loads(file.getvalue())
        data = data[0] if isinstance(data, list) and data else data
    else:
        rows = pd.read_csv(file)
        if rows.empty:
            raise ValueError("the CSV has no data rows")
        data = rows.iloc[0].to_dict()  # first row = the application
    if not isinstance(data, dict):
        raise ValueError("expected one application (a JSON object or one CSV row)")
    app = {}
    for key, default in FORM_DEFAULTS.items():
        if key in data:
            value = type(default)(data[key])
            low, high = BOUNDS.get(key, (value, value))
            if not low <= value <= high:
                raise ValueError(f"{key} = {value} is outside {low}–{high}")
            app[key] = value
    return app


def api_get(base_url: str, path: str, **params):
    """GET from the backend; None for 404."""
    r = requests.get(f"{base_url}{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def load_current_application() -> dict:
    """The form's starting values come from current.json, so presenters switch applicants outside the page."""
    try:
        return {**FORM_DEFAULTS, **json.loads(DEMO_FILE.read_text(encoding="utf-8"))}
    except (OSError, ValueError) as exc:
        st.warning(f"Could not read {DEMO_FILE.name} ({exc}); using built-in defaults.")
        return dict(FORM_DEFAULTS)

# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def render_flow():
    st.markdown(
        """
        <div class="flow">
          <div class="flow-box">Intake</div><div class="flow-arrow">→</div>
          <div class="flow-par">
            <div class="flow-box">Risk</div>
            <div class="flow-box">Fraud</div>
            <div class="flow-box">Compliance</div>
          </div>
          <div class="flow-arrow">→</div><div class="flow-box">Final Decision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(res: dict):
    d = res["decision"]
    outcome = d["outcome"]
    msg = f"**{outcome}** · Case `{res['case_id']}`"
    if outcome == "Approved":
        st.success(msg, icon="✅")
    elif outcome == "Rejected":
        st.error(msg, icon="⛔")
    else:
        st.warning(msg, icon="🔎")
    c1, c2 = st.columns([3, 1])
    c1.write(d["rationale"].replace("$", "\\$"))
    if "confidence" in d:
        c2.metric("Confidence", f"{d['confidence']:.0%}")
    for conflict in d.get("conflicts", []):
        st.info(f"Conflict resolved: {conflict}", icon="⚖️")


def render_agent_cards(res: dict):
    st.markdown("##### Stage 1 · Intake")
    intake = res["intake"]
    with st.container(border=True):
        top = st.columns([1, 4])
        top[0].markdown(pill(intake["status"], "ok" if not intake["issues"] else "warn"),
                        unsafe_allow_html=True)
        top[1].write("No data issues found." if not intake["issues"] else "; ".join(intake["issues"]))
        with st.expander("Normalized application data"):
            st.json(intake["normalized"])

    st.markdown("##### Stage 2 · Parallel checks")
    c_risk, c_fraud, c_comp = st.columns(3)

    with c_risk:
        r = res["risk"]
        kind = {"Low": "ok", "Medium": "warn", "High": "bad"}[r["band"]]
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown('<div class="agent-title">Risk Detection</div>', unsafe_allow_html=True)
        st.markdown(pill(f"{r['band']} risk", kind), unsafe_allow_html=True)
        st.metric("Probability of default", f"{r['default_probability']:.1%}")
        st.progress(min(1.0, r["default_probability"]))
        with st.expander("Top factors"):
            st.dataframe(pd.DataFrame(r["top_factors"]), hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_fraud:
        f = res["fraud"]
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown('<div class="agent-title">Fraud Detection</div>', unsafe_allow_html=True)
        st.markdown(pill(f["status"], "ok" if f["status"] == "Clear" else "bad"), unsafe_allow_html=True)
        st.metric("Flags raised", len(f["flags"]))
        if f["flags"]:
            for flag in f["flags"]:
                st.write(f"⚠️ {flag}".replace("$", "\\$"))
        else:
            st.caption("No anomalies detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_comp:
        c = res["compliance"]
        n_pass = sum(r["passed"] for r in c["rules"])
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown('<div class="agent-title">Compliance</div>', unsafe_allow_html=True)
        st.markdown(pill("Passed" if c["passed"] else "Failed", "ok" if c["passed"] else "bad"),
                    unsafe_allow_html=True)
        st.metric("Rules passed", f"{n_pass}/{len(c['rules'])}")
        with st.expander("Rule details"):
            for rule in c["rules"]:
                st.write(f"{'✅' if rule['passed'] else '❌'} **{rule['rule']}**  \n{rule['detail']}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_audit(rows: list, key: str):
    if not rows:
        st.caption("No audit entries.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download audit log (CSV)", df.to_csv(index=False),
                       file_name="audit_log.csv", mime="text/csv", key=f"dl_{key}")


def render_result(res: dict, key: str):
    render_decision_banner(res)
    st.divider()
    render_flow()
    render_agent_cards(res)
    st.divider()
    st.markdown("##### Audit trail")
    render_audit(res.get("audit", []), key)


# --------------------------------------------------------------------------
# Live pipeline view
# --------------------------------------------------------------------------
STAGES = {"intake": "Intake", "risk": "Risk", "fraud": "Fraud", "compliance": "Compliance",
          "decision": "Final Decision", "decision_llm": "LLM opinion"}


def render_live_flow(slot, state: dict):
    box = lambda key: f'<div class="flow-box {state[key]}">{STAGES[key]}</div>'
    slot.markdown(
        f"""
        <div class="flow">
          {box("intake")}<div class="flow-arrow">→</div>{box("risk")}<div class="flow-arrow">→</div>
          <div class="flow-par">{box("fraud")}{box("compliance")}</div>
          <div class="flow-arrow">→</div>{box("decision")}<div class="flow-arrow">→</div>{box("decision_llm")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_steps(slot, steps: list):
    with slot.container():
        for step in steps:
            st.markdown(f"✅ **{STAGES[step['agent']]}** · {step['detail']}  ·  :gray[+{step['elapsed']:.1f}s]")
            for note in step["notes"]:
                if note:
                    st.caption(note.replace("$", "\\$"))  # '$' would otherwise start LaTeX


def run_live(base_url: str, payload: dict):
    """Runs the application and redraws the flow as each agent finishes. Returns (result, steps)."""
    state = {key: "waiting" for key in STAGES}
    state["intake"] = "running"
    flow_slot, steps_slot = st.empty(), st.empty()
    render_live_flow(flow_slot, state)
    steps, result, t0 = [], None, time.perf_counter()
    for event in api_submit_stream(base_url, payload):
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if event["type"] == "result":
            result = event["result"]
        if event["type"] != "step":
            continue
        agent = event["agent"]
        state[agent] = "done"
        # Light up what LangGraph runs next: risk, then fraud + compliance in parallel, then the two decisions.
        if agent == "intake":
            state["risk"] = "running"
        if agent == "risk":
            state["fraud"] = state["compliance"] = "running"
        if agent in ("fraud", "compliance") and state["fraud"] == state["compliance"] == "done":
            state["decision"] = "running"
        if agent == "decision":
            state["risk"] = "skipped" if state["risk"] != "done" else "done"  # incomplete: intake went straight here
            if state["risk"] == "done":
                state["decision_llm"] = "running"
        steps.append({**event, "elapsed": time.perf_counter() - t0})
        render_live_flow(flow_slot, state)
        render_steps(steps_slot, steps)
    for key, value in state.items():
        if value != "done":
            state[key] = "skipped"
    render_live_flow(flow_slot, state)
    return result, steps


def go_to_pipeline():
    st.session_state["page"] = "Pipeline in action"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">M</div><div><div class="brand-name">MALAS</div>'
                '<div class="brand-sub">Loan approval</div></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["New application", "Pipeline in action", "Classic application", "Case lookup",
                                 "Audit log"], key="page", label_visibility="collapsed")
    st.divider()
    demo = st.toggle("Demo mode", value=False)
    api_url = st.text_input("API URL", "localhost:8010", disabled=demo).strip()
    api_url = api_url if "://" in api_url else f"http://{api_url}"

# --------------------------------------------------------------------------
# Pages: New application (mockup design) and Classic application (the original basic page)
# --------------------------------------------------------------------------
if page in ("New application", "Classic application"):
    classic = page == "Classic application"
    st.header("New loan application")
    app = load_current_application()
    if classic:
        render_flow()
    else:
        st.caption("Upload an application file or enter the applicant details manually.")
        upload = st.file_uploader("Application file", type=["csv", "json", "pdf", "docx"],
                                  label_visibility="collapsed")
        if upload:
            try:
                app.update(read_upload(upload))
                st.success(f"Form filled from {upload.name}. Check the details, then submit.")
            except (ValueError, IndexError) as exc:
                st.error(f"Could not read {upload.name}: {exc}")
        st.markdown('<div class="or-divider">or enter details manually</div>', unsafe_allow_html=True)
    HOMES = ["Rent", "Own", "Mortgage", "Other"]
    STATUSES = ["Employed", "Self-employed", "Unemployed", "Retired"]
    TERMS = [12, 24, 36, 48, 60]
    PURPOSES = ["Debt consolidation", "Home improvement", "Car", "Education", "Medical", "Business", "Other"]
    pick = lambda options, value: options.index(value) if value in options else 0
    label = lambda new, old: old if classic else new  # the classic page keeps its original wording

    def card(title):
        """Mockup: a bordered card with a small bold title. Classic: a plain subheader."""
        box = st.container(border=not classic)
        if classic:
            box.subheader(title)
        else:
            box.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        return box

    with st.form("application_form", border=classic):
        with card("Applicant"):
            a1, a2, a3 = st.columns(3)
            full_name = a1.text_input("Full name", app["full_name"])
            age = a2.number_input("Age", *BOUNDS["age"], int(app["age"]))
            home = a3.selectbox("Home ownership", HOMES, index=pick(HOMES, app["home_ownership"]))

        with card("Employment & income"):
            e1, e2, e3 = st.columns(3)
            employment_status = e1.selectbox("Employment status", STATUSES,
                                             index=pick(STATUSES, app["employment_status"]))
            employment_years = e2.number_input("Years employed", *BOUNDS["employment_years"],
                                               float(app["employment_years"]), step=0.5, format="%.1f")
            annual_income = e3.number_input("Annual income", *BOUNDS["annual_income"], int(app["annual_income"]),
                                            step=1000)

        with card("Loan request"):
            l1, l2, l3 = st.columns(3)
            loan_amount = l1.number_input("Loan amount", *BOUNDS["loan_amount"], int(app["loan_amount"]), step=500)
            term = l2.selectbox("Term (months)", TERMS, index=pick(TERMS, app["term_months"]))
            purpose = l3.selectbox("Purpose", PURPOSES, index=pick(PURPOSES, app["purpose"]))

        with card("Credit profile"):
            k1, k2, k3, k4 = st.columns(4)
            monthly_debt = k1.number_input(label("Monthly debt", "Monthly debt payments"), *BOUNDS["monthly_debt"],
                                           int(app["monthly_debt"]), step=50)
            credit_history_years = k2.number_input("Credit history (yrs)", *BOUNDS["credit_history_years"],
                                                   int(app["credit_history_years"]))
            open_accounts = k3.number_input("Open accounts", *BOUNDS["open_accounts"], int(app["open_accounts"]))
            delinquencies = k4.number_input(label("Delinquencies (2 yrs)", "Delinquencies (last 2 yrs)"),
                                            *BOUNDS["delinquencies"], int(app["delinquencies"]))

        consent = st.checkbox(label("I consent to automated processing and audit logging",
                                    "Applicant consents to automated processing and audit logging"))
        if classic:
            submitted = st.form_submit_button("Submit application", type="primary", width="stretch")
        else:
            _, right = st.columns([3, 1])
            submitted = right.form_submit_button("Submit application", type="primary", width="stretch")

    if submitted:
        if not consent:
            st.error("Consent is required to proceed.")
        else:
            payload = dict(
                full_name=full_name, age=int(age), home_ownership=home,
                employment_status=employment_status, employment_years=float(employment_years),
                annual_income=float(annual_income), loan_amount=float(loan_amount), term_months=int(term),
                purpose=purpose, monthly_debt=float(monthly_debt),
                credit_history_years=int(credit_history_years), open_accounts=int(open_accounts),
                delinquencies=int(delinquencies),
            )
            if not classic:
                st.session_state["pending"] = payload
                st.success(f"Application for {full_name} submitted.", icon="📨")
            else:
                try:
                    with st.status("Running pipeline...", expanded=True) as status:
                        st.write("Intake: validating and structuring input")
                        st.write("Risk, then Fraud and Compliance in parallel")
                        st.write("Decision: combining agent outputs")
                        result = mock_submit(payload) if demo else api_submit(api_url, payload)
                        st.session_state["classic_result"] = result
                        status.update(label="Pipeline complete", state="complete", expanded=False)
                except Exception as exc:
                    st.error(f"Could not process the application: {exc}")
    if not classic and "pending" in st.session_state:
        st.button("Watch it run in the pipeline →", type="primary", on_click=go_to_pipeline)
    if classic and "classic_result" in st.session_state:
        st.divider()
        render_result(st.session_state["classic_result"], key="classic")

# --------------------------------------------------------------------------
# Pages: Case lookup, Audit log (read past runs back from the backend)
# --------------------------------------------------------------------------
elif page in ("Case lookup", "Audit log"):
    st.header(page)
    if demo:
        st.info(f"{page} reads saved cases from the backend. Turn off Demo mode to use it.")
        st.stop()
    last_case = st.session_state["last_run"][1]["case_id"] if "last_run" in st.session_state else ""
    try:
        if page == "Case lookup":
            st.caption("Open the result of an earlier application by its case ID.")
            case_id = st.text_input("Case ID", last_case, placeholder="CASE-1A2B3C4D").strip()
            if case_id:
                result = api_get(api_url, f"/cases/{requests.utils.quote(case_id, safe='')}")
                if result is None:
                    st.warning(f"No case called {case_id}.")
                else:
                    st.divider()
                    render_result(result, key="lookup")
        else:
            st.caption("Every agent step of every application, newest first.")
            case_id = st.text_input("Filter by case ID", placeholder="All cases").strip()
            render_audit(list(reversed(api_get(api_url, "/audit", case_id=case_id))), key="audit")
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {api_url}: {exc}")

# --------------------------------------------------------------------------
# Page: Pipeline in action
# --------------------------------------------------------------------------
else:
    st.header("Pipeline in action")
    pending = st.session_state.pop("pending", None)
    if pending:
        st.caption(f"Processing the application for **{pending['full_name']}**")
        try:
            if demo:
                with st.spinner("Running the demo pipeline..."):
                    result, steps = mock_submit(pending), []
                render_flow()
            else:
                result, steps = run_live(api_url, pending)
            st.session_state["last_run"] = (pending["full_name"], result, steps)
        except Exception as exc:
            st.error(f"Could not process the application: {exc}")
            st.stop()
    elif "last_run" in st.session_state:
        name, result, steps = st.session_state["last_run"]
        st.caption(f"Last run: **{name}**. Submit a new application to watch one live.")
        if steps:
            ran = {s["agent"] for s in steps}
            render_live_flow(st.empty(), {key: "done" if key in ran else "skipped" for key in STAGES})
            render_steps(st.empty(), steps)
    else:
        st.info("No application yet. Submit one on the New application page, then come back here to watch it run.")
        st.stop()

    if result is None:
        st.error("The pipeline finished without a result.")
        st.stop()
    st.divider()
    render_result(result, key="pipeline")
