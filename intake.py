import uuid

SIGNATURE_COLUMNS = {
    "approval": {"cibil_score", "residential_assets_value", "commercial_assets_value"},
    "schema": {"verified_income", "debt_to_income", "annual_income_joint"},
}

# field -> (required/optional, default when missing)
SCHEMA = {
    "approval": {
        "loan_id": ("optional", None),
        "no_of_dependents": ("required", None),
        "education": ("required", None),
        "self_employed": ("required", None),
        "income_annum": ("required", None),
        "loan_amount": ("required", None),
        "loan_term": ("required", None),
        "cibil_score": ("required", None),
        "residential_assets_value": ("required", None),
        "commercial_assets_value": ("required", None),
        "luxury_assets_value": ("required", None),
        "bank_asset_value": ("required", None),
    },
    "schema": {
        "row_id": ("optional", None),
        "application_type": ("required", None),
        "annual_income": ("required", None),
        "annual_income_joint": ("optional", None),
        "verified_income": ("required", None),
        "verification_income_joint": ("optional", None),
        "debt_to_income": ("required", None),
        "loan_amount": ("required", None),
        "installment": ("required", None),
        "term": ("required", None),
        "grade": ("optional", None),
        "sub_grade": ("optional", None),
        "interest_rate": ("optional", None),
        "total_credit_limit": ("required", None),
        "total_credit_utilized": ("required", None),
        "delinq_2y": ("optional", 0),
        "months_since_last_delinq": ("optional", None),
        "num_collections_last_12m": ("optional", 0),
        "num_historical_failed_to_pay": ("optional", 0),
        "public_record_bankrupt": ("optional", 0),
        "tax_liens": ("optional", 0),
        "inquiries_last_12m": ("optional", 0),
        "accounts_opened_24m": ("optional", 0),
        "account_never_delinq_percent": ("optional", None),
        "loan_purpose": ("optional", None),
        "homeownership": ("optional", None),
        "emp_title": ("optional", None),
    },
}

ALIASES = {
    "approval": {
        "dependents": "no_of_dependents", "num_dependents": "no_of_dependents",
        "annual_income": "income_annum", "income": "income_annum", "yearly_income": "income_annum",
        "credit_score": "cibil_score", "cibil": "cibil_score",
        "term": "loan_term", "tenure": "loan_term", "loan_term_months": "loan_term",
        "home_value": "residential_assets_value", "residence_assets": "residential_assets_value",
        "business_assets": "commercial_assets_value",
        "self_employed_status": "self_employed", "employment_type": "self_employed",
        "id": "loan_id", "application_id": "loan_id",
    },
    "schema": {
        "income": "annual_income", "yearly_income": "annual_income",
        "joint_income": "annual_income_joint",
        "income_verification": "verified_income", "verified": "verified_income",
        "dti": "debt_to_income",
        "monthly_payment": "installment",
        "loan_term": "term", "term_months": "term",
        "credit_limit": "total_credit_limit",
        "credit_utilized": "total_credit_utilized", "credit_used": "total_credit_utilized",
        "bankruptcies": "public_record_bankrupt", "bankruptcy": "public_record_bankrupt",
        "delinquencies_2y": "delinq_2y",
        "recent_inquiries": "inquiries_last_12m",
        "new_accounts": "accounts_opened_24m",
        "purpose": "loan_purpose", "homeowner_status": "homeownership",
        "employer": "emp_title", "job_title": "emp_title",
        "id": "row_id", "application_id": "row_id",
    },
}

NON_NEGATIVE_FIELDS = {
    "approval": ["income_annum", "loan_amount", "residential_assets_value",
                 "commercial_assets_value", "luxury_assets_value", "bank_asset_value"],
    "schema": ["annual_income", "loan_amount", "total_credit_limit", "total_credit_utilized"],
}

TEXT_FIELDS = {
    "approval": {"education", "self_employed"},
    "schema": {"application_type", "verified_income", "verification_income_joint", "grade",
               "sub_grade", "loan_purpose", "homeownership", "emp_title"},
}
ALLOWED_VALUES = {
    "education": {"Graduate", "Not Graduate"},
    "self_employed": {"Yes", "No"},
    "application_type": {"individual", "joint"},
    "verified_income": {"Verified", "Source Verified", "Not Verified"},
}
ID_FIELD = {"approval": "loan_id", "schema": "row_id"}


def _norm_key(key) -> str:
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def _canonical(keys, dataset):
    return {ALIASES[dataset].get(k, k) for k in keys}


def detect_dataset(keys):
    def score(ds):
        canon = _canonical(keys, ds)
        return len(canon & SIGNATURE_COLUMNS[ds]), len(canon & SCHEMA[ds].keys())
    best = max(SCHEMA, key=score)
    return best if score(best)[0] > 0 else None


def run(raw: dict, dataset: str = None) -> dict:
    issues = []
    submission = {_norm_key(k): v.strip() if isinstance(v, str) else v for k, v in raw.items()}
    submission = {k: v for k, v in submission.items() if v not in ("", None) and v == v}  # v == v drops NaN

    dataset = dataset or detect_dataset(submission.keys())
    if dataset is None:
        return {"application_id": None, "dataset": None, "record": {}, "valid": False,
                "issues": [{"field": None, "type": "unrecognized_schema",
                            "message": "Submission matches neither the approval nor the schema dataset"}]}

    record = {}
    for key, value in submission.items():
        canon = ALIASES[dataset].get(key, key)
        if canon not in SCHEMA[dataset]:
            issues.append({"field": key, "type": "unknown_field", "message": "Ignored: not part of the schema"})
        else:
            record[canon] = value

    for field, (status, default) in SCHEMA[dataset].items():
        if field in record:
            continue
        if status == "required":
            issues.append({"field": field, "type": "missing_required", "message": "Required field is missing"})
        else:
            record[field] = default

    for field, value in list(record.items()):
        if value is None or field in TEXT_FIELDS[dataset] or field == ID_FIELD[dataset]:
            continue
        try:
            record[field] = float(value)
        except (TypeError, ValueError):
            issues.append({"field": field, "type": "not_numeric", "message": f"Expected a number, got {value!r}"})
            record[field] = None

    for field, allowed in ALLOWED_VALUES.items():
        if record.get(field) is not None:
            match = {a.lower(): a for a in allowed}.get(str(record[field]).lower())
            if match is None:
                issues.append({"field": field, "type": "invalid_value",
                               "message": f"{record[field]!r} not in {sorted(allowed)}"})
            else:
                record[field] = match

    for field in NON_NEGATIVE_FIELDS[dataset]:
        if isinstance(record.get(field), float) and record[field] < 0:
            issues.append({"field": field, "type": "impossible_value", "message": f"Negative value {record[field]}"})
    cibil = record.get("cibil_score")
    if isinstance(cibil, float) and not 300 <= cibil <= 900:
        issues.append({"field": "cibil_score", "type": "impossible_value", "message": f"CIBIL {cibil} outside 300-900"})

    app_id = record.get(ID_FIELD[dataset]) or f"APP-{uuid.uuid4().hex[:8]}"
    record[ID_FIELD[dataset]] = app_id
    # Impossible values are passed on (compliance/decision handle them); missing or unparseable data halts scoring.
    blocking = {"missing_required", "not_numeric", "invalid_value"}
    return {"application_id": app_id, "dataset": dataset, "record": record, "issues": issues,
            "valid": not any(i["type"] in blocking for i in issues)}
