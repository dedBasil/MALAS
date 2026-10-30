from llm import ask_json

PROFILES = {
    "schema": {
        "fields": [
            "application_type", "annual_income", "annual_income_joint", "verified_income",
            "verification_income_joint", "debt_to_income", "loan_amount", "installment", "term",
            "grade", "sub_grade", "interest_rate", "total_credit_limit", "total_credit_utilized",
            "delinq_2y", "months_since_last_delinq", "num_collections_last_12m",
            "num_historical_failed_to_pay", "public_record_bankrupt", "tax_liens",
            "inquiries_last_12m", "accounts_opened_24m", "account_never_delinq_percent",
            "loan_purpose", "homeownership", "emp_title",
        ],
        "considerations": (
            "1. Whether the anomaly has an innocent explanation (non-earning co-applicant, new-to-credit, retiree)\n"
            "2. Internal consistency of the data (e.g. delinquency fields that contradict each other, "
            "a joint application without joint income)\n"
            "3. Whether the installment, loan amount and term fit together\n"
            "Amounts are in US dollars."
        ),
    },
    "approval": {
        "fields": [
            "no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
            "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value",
        ],
        "considerations": (
            "1. Separate 'poor credit' (high risk score from low CIBIL) from 'suspicious' "
            "(inconsistent or implausible data). Poor credit alone is not a fraud indicator.\n"
            "2. Asset inconsistencies: do declared assets plausibly match stated income?\n"
            "3. Data integrity issues (negative values, implausible combinations)\n"
            "4. Self-employment status versus income level\n"
            "Amounts are in Indian rupees (CIBIL is India's credit bureau)."
        ),
    },
}

FRAUD_PROMPT = """You are a fraud and data-integrity reviewer for a lending institution. Your ONLY question is:
does this application look misrepresented, fabricated or internally inconsistent?

You are NOT judging creditworthiness. Other agents already handle that: a compliance agent enforces
credit policy (minimum credit score, loan-to-income and loan-to-asset caps, debt-to-income) and an
ML model scores repayment risk. A low credit score, a large loan or weak finances are NOT fraud
signals by themselves -- an honest applicant with bad credit should get APPROVE from you.

ML Risk Score: {risk_score} / 100 ({score_meaning}) -- context only, it does not measure fraud.

Portfolio flags that fired: {flags}
{flag_notes}
A flag that fires on a large share of honest applicants is weak evidence on its own.

Intake issues: {intake_issues}

Application Data:
{application_data}

{norms}

Things worth checking:
{considerations}

Decide with these rules:
- APPROVE: no specific integrity concern. This is the expected answer for most applications,
  including risky-but-plausible ones.
- MANUAL_REVIEW: you can name a SPECIFIC inconsistency that a human could check against a document
  (e.g. stated income cannot plausibly fund the declared assets by a wide margin, fields contradict
  each other). Vague unease or a single common flag is not enough.
- REJECT: the data is impossible or almost certainly fabricated (negative amounts, values that
  cannot coexist).
MANUAL_REVIEW and REJECT must list at least one red flag that cites the exact field values.

Respond ONLY with valid JSON in this exact shape (fill the fields in this order):
{{
    "red_flags": [<specific integrity concerns citing field values, empty list if none>],
    "explanation": "<2-3 sentences on why the data is or is not consistent>",
    "recommendation": "APPROVE" | "MANUAL_REVIEW" | "REJECT"
}}"""


# What each risk-agent flag means and how often it fires, so the LLM can weigh a flag against its base rate.
FLAG_NOTES = {
    "loan_exceeds_total_assets": "loan larger than all declared assets combined (~0.6% of applicants)",
    "luxury_dominated_assets_low_income": "luxury assets >50% of total and below-median income "
                                          "(common: ~18% of applicants, not predictive on its own)",
    "unverified_income": "income not verified (common: roughly a third of applicants)",
    "high_credit_utilization": "credit utilization above 85% (top ~5% of portfolio)",
    "rapid_new_accounts": "10+ accounts opened in 24 months (top ~5%)",
    "adverse_credit_history": "2+ adverse items (collections, failed payments, bankruptcy, liens)",
    "delinquency_inconsistency": "delinquency fields contradict each other",
    "grade_rate_mismatch": "interest rate unusual for the stated grade (top ~5%)",
    "unusual_profile": "overall profile is in the top 5% most unusual of the portfolio (anomaly model)",
}

def _div(a, b):
    return None if a is None or not b else a / b


def _total_assets(r):
    values = [r.get(a) for a in ("residential_assets_value", "commercial_assets_value",
                                 "luxury_assets_value", "bank_asset_value")]
    return None if None in values else sum(values)


# ratio -> (how to compute it, portfolio median, portfolio 99th percentile). ~22% of honest applicants
# exceed the 95th percentile on at least one ratio, so only the top 1% counts as unusual.
RATIO_NORMS = {
    "approval": {
        "luxury assets / income": (lambda r: _div(r.get("luxury_assets_value"), r.get("income_annum")), 3.0, 4.0),
        "residential assets / income": (lambda r: _div(r.get("residential_assets_value"), r.get("income_annum")), 1.5, 2.97),
        "total assets / income": (lambda r: _div(_total_assets(r), r.get("income_annum")), 6.4, 9.22),
        "loan / income": (lambda r: _div(r.get("loan_amount"), r.get("income_annum")), 3.0, 4.0),
        "loan / total assets": (lambda r: _div(r.get("loan_amount"), _total_assets(r)), 0.47, 0.94),
    },
    "schema": {
        "debt-to-income %": (lambda r: r.get("debt_to_income"), 17.6, 58.6),
        "credit utilization": (lambda r: _div(r.get("total_credit_utilized"), r.get("total_credit_limit")), 0.36, 1.05),
        "inquiries last 12m": (lambda r: r.get("inquiries_last_12m"), 1, 11),
        "accounts opened 24m": (lambda r: r.get("accounts_opened_24m"), 4, 14),
        "loan / income": (lambda r: _div(r.get("loan_amount"), r.get("annual_income")), 0.21, 1.0),
        "yearly installments / income": (lambda r: _div(12 * (r.get("installment") or 0), r.get("annual_income")), 0.07, 0.36),
        "credit limit / income": (lambda r: _div(r.get("total_credit_limit"), r.get("annual_income")), 1.8, 9.7),
    },
}


def ratio_report(dataset, record):
    """Ratios compared to the portfolio in code: the LLM kept misreading e.g. 3.4x as above a 3.9x percentile."""
    lines = []
    for name, (fn, median, p99) in RATIO_NORMS[dataset].items():
        value = fn(record)
        if value is None:
            continue
        verdict = "UNUSUAL: in the top 1% of the portfolio" if value > p99 else "normal (within the portfolio's usual range)"
        lines.append(f"- {name}: {value:.2f} (median {median}, 99th pct {p99}) -> {verdict}")
    return ("Pre-computed ratios vs this portfolio (already compared for you; trust these labels):\n"
            + "\n".join(lines) + "\nOnly a ratio marked UNUSUAL may be cited as out of the ordinary.")


def run(intake: dict, risk: dict) -> dict:
    profile = PROFILES[intake["dataset"]]
    record = intake["record"]
    flags = risk.get("flags") or []
    prompt = FRAUD_PROMPT.format(
        risk_score=risk["risk_score"] if risk.get("status") == "ok" else "not available (ML agent did not score this application)",
        score_meaning=risk.get("score_meaning") or "n/a",
        flags=", ".join(flags) or "none",
        flag_notes="\n".join(f"  - {f}: {FLAG_NOTES.get(f, '')}" for f in flags) or "  none",
        intake_issues="; ".join(f"{i['field']}: {i['message']}" for i in intake["issues"]) or "none",
        application_data="\n".join(f"- {f}: {record[f]}" for f in profile["fields"] if record.get(f) is not None),
        considerations=profile["considerations"],
        norms=ratio_report(intake["dataset"], record),
    )
    result = ask_json("You are a fraud-review agent for loan applications.", prompt, {
        "red_flags": None, "explanation": None, "recommendation": ("APPROVE", "MANUAL_REVIEW", "REJECT"),
    })
    if result["status"] == "ok" and result["recommendation"] != "APPROVE" and not result["red_flags"]:
        # A review nobody can act on: without a named inconsistency, treat it as clean.
        result["downgraded_from"] = result["recommendation"]
        result["recommendation"] = "APPROVE"
    return result
