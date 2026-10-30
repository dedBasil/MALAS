from fraud import ratio_report
from llm import ask_json

RANK = {"APPROVE": 0, "MANUAL_REVIEW": 1, "REJECT": 2}
# Approval score = 100 x (1 - P(approved)). Schema score = default-risk percentile, too weak (AUC ~0.65) to reject on.
APPROVAL_REJECT_SCORE = 80
APPROVAL_REVIEW_SCORE = 50
SCHEMA_REVIEW_PERCENTILE = 90

KEY_FIELDS = {
    "approval": ["income_annum", "loan_amount", "loan_term", "cibil_score", "no_of_dependents", "self_employed",
                 "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"],
    "schema": ["annual_income", "verified_income", "debt_to_income", "loan_amount", "installment", "term",
               "grade", "interest_rate", "total_credit_limit", "total_credit_utilized", "delinq_2y",
               "public_record_bankrupt", "inquiries_last_12m", "loan_purpose"],
}
CURRENCY = {"approval": "Indian rupees, credit score is CIBIL (300-900)", "schema": "US dollars (Lending Club loans)"}


def run(intake: dict, risk: dict, compliance: dict, fraud: dict) -> dict:
    signals = []  # (decision, source, reason)

    if not intake["valid"]:
        signals.append(("MANUAL_REVIEW", "intake", "Application is incomplete or unparseable; cannot be assessed automatically"))
    for issue in intake["issues"]:
        if issue["type"] == "impossible_value":
            signals.append(("MANUAL_REVIEW", "intake", f"{issue['field']}: {issue['message']}"))

    if compliance:
        for rule, reason in zip(compliance["violated_rules"], compliance["adverse_action_reasons"]):
            signals.append((compliance["rule_severity"][rule], "compliance", reason))
        for rule in compliance["unevaluated_rules"]:
            signals.append(("MANUAL_REVIEW", "compliance", f"Rule '{rule}' could not be evaluated (missing data)"))

    if fraud and fraud.get("status") == "ok":
        if fraud["recommendation"] != "APPROVE":
            signals.append((fraud["recommendation"], "fraud", fraud["explanation"]))
    elif intake["valid"]:
        signals.append(("MANUAL_REVIEW", "fraud", f"Fraud check unavailable: {(fraud or {}).get('error', 'not run')}"))

    # The ML score only decides when nothing else fired.
    if not signals:
        score = risk.get("risk_score")
        if risk.get("status") != "ok":
            signals.append(("MANUAL_REVIEW", "ml_risk", f"ML risk score unavailable: {risk.get('error')}"))
        elif intake["dataset"] == "approval" and score >= APPROVAL_REJECT_SCORE:
            signals.append(("REJECT", "ml_risk", f"Risk score {score} indicates rejection under past policy"))
        elif intake["dataset"] == "approval" and score >= APPROVAL_REVIEW_SCORE:
            signals.append(("MANUAL_REVIEW", "ml_risk", f"Risk score {score} is borderline"))
        elif intake["dataset"] == "schema" and score >= SCHEMA_REVIEW_PERCENTILE:
            signals.append(("MANUAL_REVIEW", "ml_risk",
                            f"Predicted default risk is higher than {score:.0f}% of past loans"))

    final = max((s[0] for s in signals), key=RANK.get, default="APPROVE")
    return {
        "final_decision": final,
        "reasons": [f"[{src}] {reason}" for dec, src, reason in signals if dec == final],
        "signals": [{"decision": d, "source": s, "reason": r} for d, s, r in signals],
    }


DECISION_PROMPT = """You are the final decision-making agent for a lending institution. Four upstream agents and a
rule engine have already evaluated this application. Read all of their work, then give one final decision,
a confidence, and an explanation a loan officer can act on.

Dataset: {dataset} (amounts in {currency})

Key application figures:
{application_data}

{norms}
The compliance agent's limits ARE the lending policy. Do not apply credit-score, loan-to-income or other
thresholds of your own: a figure that passes compliance and sits inside the ranges above is acceptable.

Intake agent issues: {intake_issues}

ML risk agent:
  Score: {risk_score} / 100 -- {score_meaning}
  {risk_extra}
  Flags that fired: {risk_flags}

Compliance agent:
  Violated rules (with the decision each one maps to): {violations}
  Rules that could not be evaluated: {unevaluated}

Fraud agent:
  {fraud_summary}

Rule engine decision (takes the strictest signal): {rule_decision}
  Reasons: {rule_reasons}

How to weigh this (defaults you may deviate from ONLY with clear justification in your reasoning):
  - A compliance rule that maps to REJECT or MANUAL_REVIEW should not be downgraded to an approval.
  - A fraud agent REJECT should not be overridden by a clean compliance check.
  - The ML score is the weakest signal; it mainly matters when compliance and fraud raised nothing.
    A LOW score is good (low risk); read its meaning above before citing it.
  - Look for what the rule engine cannot see: several individually minor findings (risk flags, borderline
    ratios, a manual-review rule) that together point to a bigger problem, or a clean-looking case
    whose figures sit at the edge of several limits at once.
  - Without a REJECT from compliance or fraud, the strictest you may go is MANUAL_REVIEW.
  - Cite the actual figures in your reasoning, not just the other agents' conclusions.

Respond ONLY with valid JSON in this exact shape:
{{
    "final_decision": "APPROVE" | "MANUAL_REVIEW" | "REJECT",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<2-4 sentence explanation a loan officer could read and act on>",
    "deviated_from_default_guidance": <true if your decision differs from the rule engine decision, false otherwise>,
    "deviation_explanation": "<if deviated_from_default_guidance is true, explain why; otherwise empty string>"
}}"""


def _fraud_summary(fraud):
    if not fraud or fraud.get("status") != "ok":
        return f"Not available ({(fraud or {}).get('error', 'not run')})"
    return (f"Recommendation: {fraud['recommendation']}\n  Red flags: {fraud['red_flags'] or 'none'}\n"
            f"  Explanation: {fraud['explanation']}")


def run_llm(intake: dict, risk: dict, compliance: dict, fraud: dict, rule_decision: dict) -> dict:
    """Synthesizes all upstream work into a verdict with confidence. The deterministic run() stays the system of record."""
    if not compliance:
        return {"status": "skipped", "error": "no compliance result (intake invalid)"}
    dataset, record = intake["dataset"], intake["record"]
    risk_ok = risk.get("status") == "ok"
    extra = (f"Only a score of {SCHEMA_REVIEW_PERCENTILE}+ counts as elevated; the model is weak (AUC ~0.65), "
             f"so a lower score is ordinary and is no reason for review on its own. Anomaly score "
             f"{risk['anomaly_score']} (top-5% threshold {risk['anomaly_threshold']}).") if risk_ok and dataset == "schema" else ""
    prompt = DECISION_PROMPT.format(
        dataset=dataset,
        currency=CURRENCY[dataset],
        norms=ratio_report(dataset, record),
        application_data="\n".join(f"- {f}: {record[f]}" for f in KEY_FIELDS[dataset] if record.get(f) is not None),
        intake_issues="; ".join(f"{i['field']}: {i['message']}" for i in intake["issues"]) or "none",
        risk_score=risk["risk_score"] if risk_ok else f"not available ({risk.get('error')})",
        score_meaning=risk.get("score_meaning"),
        risk_extra=extra,
        risk_flags=", ".join(risk.get("flags") or []) or "none",
        violations=", ".join(f"{r} -> {s}" for r, s in compliance["rule_severity"].items()) or "none",
        unevaluated=", ".join(compliance["unevaluated_rules"]) or "none",
        fraud_summary=_fraud_summary(fraud),
        rule_decision=rule_decision["final_decision"],
        rule_reasons="; ".join(rule_decision["reasons"]) or "none (no signal fired)",
    )
    result = ask_json("You are the final decision-making agent for loan applications.", prompt, {
        "final_decision": tuple(RANK), "confidence": None, "reasoning": None,
        "deviated_from_default_guidance": (True, False), "deviation_explanation": None,
    })
    if result["status"] == "ok":
        # The LLM sometimes claims a deviation while agreeing with the rule engine; the comparison is a fact.
        result["deviated_from_default_guidance"] = result["final_decision"] != rule_decision["final_decision"]
        if not result["deviated_from_default_guidance"]:
            result["deviation_explanation"] = ""
    return result
