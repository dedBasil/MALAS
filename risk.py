from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS, DATA = HERE / "models", HERE / "data"
BAD_STATUSES = {"Charged Off", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"}
ASSETS = ["residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]

# Flag thresholds ~ the 95th percentile of the schema portfolio (loans_full_schema.csv).
UTILIZATION_FLAG = 0.85
NEW_ACCOUNTS_FLAG = 10
ADVERSE_HISTORY_FLAG = 2
GRADE_RATE_Z_FLAG = 1.65

SCORE_MEANING = {
    "approval": "100 x (1 - P(approved)) from a logistic regression trained on past decisions. "
                "Likelihood of rejection under past policy, NOT fraud probability.",
    "schema": "Percentile of predicted early-default risk (late / charged off) within the loan portfolio: "
              "a score of S means riskier than S% of past loans. NOT fraud probability.",
}

# Same features the isolation forest (models/isolation_forest_model.joblib) was trained on.
ANOMALY_FEATURES = [
    "income_verification_risk", "zero_income_flag", "loan_to_income_ratio", "debt_to_income_clean",
    "installment_to_income", "credit_utilization", "grade_rate_zscore", "adverse_history_score",
    "inquiries_last_12m", "accounts_opened_24m", "delinq_inconsistency", "delinq_missing_inconsistency",
    "credit_limit_to_income", "joint_consistency_issue",
]
DEFAULT_FEATURES = ANOMALY_FEATURES + ["interest_rate", "term", "grade_num"]


def build_features(df):
    """Approval-dataset features (models/logistic_regression_model.joblib expects these)."""
    X = df.drop(columns=["loan_id", "loan_status"], errors="ignore")
    X["loan_to_income"] = X["loan_amount"] / X["income_annum"]
    X["total_assets"] = X[ASSETS].sum(axis=1)
    return X


def engineer_features(df):
    """Lending Club features for the default and anomaly models. Group stats come from df, so score rows inside the portfolio."""
    feat = pd.DataFrame(index=df.index)
    feat["income_verification_risk"] = df["verified_income"].map(
        {"Verified": 0, "Source Verified": 1, "Not Verified": 2}).fillna(1)
    feat["zero_income_flag"] = (df["annual_income"] <= 0).astype(int)
    safe_income = df["annual_income"].replace(0, np.nan)
    feat["loan_to_income_ratio"] = (df["loan_amount"] / safe_income).fillna(df["loan_amount"].max() / 1000)
    feat["debt_to_income_clean"] = df["debt_to_income"].fillna(df["debt_to_income"].median())
    feat["installment_to_income"] = (df["installment"] * 12 / safe_income).fillna(feat["loan_to_income_ratio"])
    utilization = df["total_credit_utilized"] / df["total_credit_limit"].replace(0, np.nan)
    feat["credit_utilization"] = utilization.fillna(0).clip(0, 3)

    rate = pd.to_numeric(df["interest_rate"], errors="coerce")
    grade_mean = rate.groupby(df["grade"]).transform("mean")
    grade_std = rate.groupby(df["grade"]).transform("std").replace(0, np.nan)
    feat["grade_rate_zscore"] = ((rate - grade_mean) / grade_std).fillna(0)

    feat["adverse_history_score"] = sum(df[c].fillna(0) for c in [
        "num_collections_last_12m", "num_historical_failed_to_pay", "delinq_2y", "public_record_bankrupt", "tax_liens"])
    feat["inquiries_last_12m"] = df["inquiries_last_12m"].fillna(0)
    feat["accounts_opened_24m"] = df["accounts_opened_24m"].fillna(0)

    has_delinq = df["delinq_2y"].fillna(0) > 0
    feat["delinq_inconsistency"] = ((df["account_never_delinq_percent"].fillna(100) >= 99) & has_delinq).astype(int)
    feat["delinq_missing_inconsistency"] = (has_delinq & df["months_since_last_delinq"].isna()).astype(int)
    feat["credit_limit_to_income"] = (df["total_credit_limit"] / safe_income).fillna(0).clip(0, 50)
    feat["joint_consistency_issue"] = (
        ((df["application_type"] == "individual") & df["annual_income_joint"].notna())
        | ((df["application_type"] == "joint") & df["annual_income_joint"].isna())
    ).astype(int)

    feat["interest_rate"] = rate  # missing values are imputed inside the default model
    feat["term"] = pd.to_numeric(df["term"], errors="coerce")
    feat["grade_num"] = df["grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    return feat


def _load(path):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing from {path.parent}")
    return joblib.load(path)


@lru_cache
def _approval_model():
    model = _load(MODELS / "logistic_regression_model.joblib")
    df = pd.read_csv(DATA / "loan_approval_dataset.csv", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return model, df["income_annum"].median()


@lru_cache
def _schema_models():
    default = _load(MODELS / "default_model.joblib")
    iso = _load(MODELS / "isolation_forest_model.joblib")
    portfolio = pd.read_csv(DATA / "loans_full_schema.csv")
    feat = engineer_features(portfolio)
    raw = -iso["model"].decision_function(iso["scaler"].transform(feat[iso["feature_columns"]].astype(float)))
    # risk2.py min-max scaled over the training set but did not save the bounds, so recover them.
    lo, hi = raw.min(), raw.max()
    anomaly_threshold = np.quantile((raw - lo) / (hi - lo) * 100, 0.95)
    bad_rate = portfolio["loan_status"].isin(BAD_STATUSES).mean()
    return default, iso, portfolio, lo, hi, anomaly_threshold, bad_rate


def _score_approval(record):
    model, median_income = _approval_model()
    X = build_features(pd.DataFrame([record]))[list(model.feature_names_in_)]
    p_approved = model.predict_proba(X)[0, 1]

    total_assets = sum(record[a] for a in ASSETS)
    checks = {
        "loan_exceeds_total_assets": record["loan_amount"] > total_assets,
        "luxury_dominated_assets_low_income": total_assets > 0
            and record["luxury_assets_value"] > 0.5 * total_assets
            and record["income_annum"] < median_income,
    }
    return round(float(100 * (1 - p_approved)), 1), checks, {}


def _score_schema(record):
    default, iso, portfolio, lo, hi, anomaly_threshold, bad_rate = _schema_models()
    feat = engineer_features(pd.concat([portfolio, pd.DataFrame([record])], ignore_index=True)).iloc[[-1]]

    x = feat[DEFAULT_FEATURES].astype(float)
    p_bad = default["model"].predict_proba(x)[0, 1]
    # Balanced class weights multiply the odds by n_good/n_bad; undo that to get a real probability.
    odds = p_bad / (1 - p_bad) * bad_rate / (1 - bad_rate)
    p_calibrated = odds / (1 + odds)
    imputer, scaler, lr = (step for _, step in default["model"].steps)
    contributions = lr.coef_[0] * scaler.transform(imputer.transform(x))[0]  # log-odds pushed by each feature
    top = np.argsort(-np.abs(contributions))[:3]
    top_factors = [{"factor": DEFAULT_FEATURES[i], "impact": round(float(contributions[i]), 2)} for i in top]
    percentile = 100 * np.searchsorted(default["portfolio_probs"], p_bad) / len(default["portfolio_probs"])

    raw = -iso["model"].decision_function(iso["scaler"].transform(feat[iso["feature_columns"]].astype(float)))[0]
    anomaly = float(np.clip((raw - lo) / (hi - lo) * 100, 0, 100))

    f = feat.iloc[0]
    checks = {
        "unverified_income": f["income_verification_risk"] == 2,
        "high_credit_utilization": f["credit_utilization"] > UTILIZATION_FLAG,
        "rapid_new_accounts": f["accounts_opened_24m"] >= NEW_ACCOUNTS_FLAG,
        "adverse_credit_history": f["adverse_history_score"] >= ADVERSE_HISTORY_FLAG,
        "delinquency_inconsistency": bool(f["delinq_inconsistency"] or f["delinq_missing_inconsistency"]),
        "grade_rate_mismatch": abs(f["grade_rate_zscore"]) > GRADE_RATE_Z_FLAG,
        "unusual_profile": anomaly >= anomaly_threshold,
    }
    extra = {"p_default": round(float(p_calibrated), 4), "top_factors": top_factors, "anomaly_score": round(anomaly, 1),
             "anomaly_threshold": round(float(anomaly_threshold), 1)}
    return round(float(percentile), 1), checks, extra


def run(intake: dict) -> dict:
    dataset = intake["dataset"]
    base = {"model": "logistic_regression" if dataset == "approval" else "default_logistic_regression",
            "score_meaning": SCORE_MEANING.get(dataset)}
    if not intake["valid"]:
        return {**base, "status": "skipped", "error": "intake record invalid", "risk_score": None, "flags": []}
    try:
        score, checks, extra = (_score_approval if dataset == "approval" else _score_schema)(intake["record"])
    except Exception as e:
        return {**base, "status": "error", "error": str(e), "risk_score": None, "flags": []}
    return {**base, "status": "ok", "risk_score": score, **extra, "flags": [n for n, hit in checks.items() if hit]}
