ASSETS = ["residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]

# Approval limits come from loan_approval_dataset.csv: 99-100% of applicants with CIBIL 550-600 were approved
# vs ~10% below 550; no applicant borrowed more than 4.0x income, and only ~0.6% more than their total assets.
RULE_SEVERITY = {
    "approval": {
        "min_cibil_score": "REJECT",
        "max_loan_to_income": "MANUAL_REVIEW",
        "loan_exceeds_total_assets": "MANUAL_REVIEW",
        "negative_asset_value": "MANUAL_REVIEW",
    },
    "schema": {
        "max_debt_to_income": "REJECT",
        "income_verification_required": "MANUAL_REVIEW",
        "max_credit_utilization": "MANUAL_REVIEW",
        "bankruptcy_manual_review": "MANUAL_REVIEW",
    },
}


def _approval_rules(r):
    assets = [r.get(a) for a in ASSETS]
    total = None if None in assets else sum(assets)
    income, loan = r.get("income_annum"), r.get("loan_amount")
    # rule -> (violated: True/False/None when not evaluable, adverse-action reason)
    return {
        "min_cibil_score": (None if r.get("cibil_score") is None else r["cibil_score"] < 550,
                            "Credit score (CIBIL) is below the minimum of 550"),
        "max_loan_to_income": (None if None in (income, loan) else loan > 4 * income,
                               "Requested loan exceeds 4 times annual income"),
        "loan_exceeds_total_assets": (None if None in (total, loan) else loan > total,
                                      "Requested loan exceeds total declared assets"),
        "negative_asset_value": (None if total is None else any(a < 0 for a in assets),
                                 "A declared asset value is negative and must be verified"),
    }


def _schema_rules(r):
    limit, used = r.get("total_credit_limit"), r.get("total_credit_utilized")
    return {
        "max_debt_to_income": (None if r.get("debt_to_income") is None else r["debt_to_income"] > 43,
                               "Debt-to-income ratio exceeds 43%"),
        "income_verification_required": (None if r.get("loan_amount") is None
                                         else r["loan_amount"] > 100_000
                                         and r.get("verified_income") not in ("Verified", "Source Verified"),
                                         "Loans over 100,000 require verified income"),
        "max_credit_utilization": (None if None in (limit, used) else (used > limit if limit else used > 0),
                                   "Credit utilization exceeds 100% of available credit"),
        "bankruptcy_manual_review": ((r.get("public_record_bankrupt") or 0) > 0,
                                     "Public record bankruptcy requires manual review"),
    }


def run(intake: dict) -> dict:
    dataset = intake["dataset"]
    rules = (_approval_rules if dataset == "approval" else _schema_rules)(intake["record"])
    violated = [name for name, (hit, _) in rules.items() if hit]
    return {
        "status": "ok",
        "compliant": not violated,
        "violated_rules": violated,
        "rule_severity": {name: RULE_SEVERITY[dataset][name] for name in violated},
        "unevaluated_rules": [name for name, (hit, _) in rules.items() if hit is None],
        "adverse_action_reasons": [rules[name][1] for name in violated],
        # Every rule with its outcome (passed is None when it could not be evaluated), for the UI's rule list.
        "rules": [{"rule": name, "passed": None if hit is None else not hit, "reason": reason,
                   "severity": RULE_SEVERITY[dataset][name]} for name, (hit, reason) in rules.items()],
    }
