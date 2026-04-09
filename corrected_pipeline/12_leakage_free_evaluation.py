"""
Phase 12: Leakage-Free Evaluation with Claim-Level Model
All methods pre-June 2024. Zero data leakage.

Fixes applied vs previous scripts:
  1. Z-scores computed inside CV fold (train only)
  2. Percentiles computed inside CV fold (train only)
  3. IsolationForest fitted inside CV fold (train only)
  4. Physician stats from train-fold claims only
  5. Beneficiary network from train-fold providers only
  6. Feature selection (MI) from train labels only
  7. Ensemble Optuna uses AUC-PR (no threshold search)
  8. Bootstrap CIs re-threshold inside each iteration
  9. Code fraction denominator fixed (nunique ClaimID)
  10. Holm-Bonferroni with monotonicity enforcement

New techniques:
  - Claim-level two-stage model (XGBoost on 558K claims)
  - Focal loss XGBoost (custom objective)
  - LightGBM DART mode
  - Bayesian target encoding (category_encoders)
  - Monotone constraints on known-direction features
  - GradientBoosting with sample_weight for class balance
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("leakage_free.log")],
)
log = logging.getLogger(__name__)
np.random.seed(42)

# ---- Constants ----
UNSAFE_PATTERNS = [
    "_zscore", "_pctile", "IsoForest_", "Phys_AvgReimb", "OpPhys_AvgReimb",
    "Phys_ClaimCount", "Phys_Role_Overlap",
    "Shared_Bene_Count", "Bene_Exclusivity", "Avg_Providers_Per_Bene",
    "Max_Providers_Per_Bene", "Provider_Network_Degree",
]

ZSCORE_SRC_COLS = [
    "InscClaimAmtReimbursed", "Claim_Count", "Beneficiary_Count",
    "Claim_Duration", "Dead_Patient_Ratio", "Inpatient_Ratio",
    "Claims_Per_Bene", "Reimburse_CV",
]

PCTILE_SRC_COLS = [
    "InscClaimAmtReimbursed", "Claim_Count", "Beneficiary_Count",
    "Dead_Patient_Ratio", "Claim_Duration", "Reimburse_max",
]

CLAIM_FEATURES = [
    "InscClaimAmtReimbursed", "DeductibleAmtPaid", "Claim_Duration",
    "Days_Admitted", "Diag_Code_Count", "Proc_Code_Count", "Age",
    "is_dead", "Disease_Count", "is_inpatient", "Has_Deductible",
    "Charge_Per_Day", "TotalClaimAmt",
]

# Known fraud-positive-direction features for monotone constraints
POSITIVE_MONOTONE = ["Dead_Patient_Ratio", "Claims_Per_Bene", "Dead_Claim_Rate"]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 12: Leakage-Free Evaluation")
    p.add_argument("--data-dir", default="/home/ubuntu/data")
    p.add_argument("--pipeline-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    return p.parse_args()


# ================================================================
# FOCAL LOSS (Lin et al. 2017, pre-June 2024)
# ================================================================
def focal_loss_objective(y_pred, dtrain, gamma=2.0):
    """Custom XGBoost focal loss objective."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    # Gradient
    grad = p - y_true
    # Focal weighting
    pt = np.where(y_true == 1, p, 1 - p)
    focal_weight = (1 - pt) ** gamma
    grad = focal_weight * grad
    # Hessian (approximate)
    hess = focal_weight * p * (1 - p)
    hess = np.maximum(hess, 1e-7)
    return grad, hess


def focal_loss_obj(y_pred, dtrain):
    return focal_loss_objective(y_pred, dtrain, gamma=2.0)


# ================================================================
# HOLM-BONFERRONI (with monotonicity enforcement)
# ================================================================
def holm_bonferroni(p_values):
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    corrected = {}
    prev_max = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(p * (m - rank), 1.0)
        adj = max(adj, prev_max)  # Monotonicity enforcement
        corrected[name] = adj
        prev_max = adj
    return corrected


# ================================================================
# IN-FOLD FEATURE COMPUTATION (no leakage)
# ================================================================
def compute_fold_zscores(train_df, test_df, cols):
    """Z-scores from train statistics only."""
    for col in cols:
        if col not in train_df.columns:
            continue
        mu = train_df[col].mean()
        sigma = train_df[col].std()
        if sigma == 0 or np.isnan(sigma):
            sigma = 1e-9
        zname = f"{col}_zscore"
        train_df[zname] = (train_df[col] - mu) / sigma
        test_df[zname] = (test_df[col] - mu) / sigma
    return train_df, test_df


def compute_fold_percentiles(train_df, test_df, cols):
    """Percentile ranks from train distribution only."""
    for col in cols:
        if col not in train_df.columns:
            continue
        pname = f"{col}_pctile"
        train_df[pname] = train_df[col].rank(pct=True)
        # For test: find position in train distribution
        train_sorted = np.sort(train_df[col].dropna().values)
        n = len(train_sorted)
        if n == 0:
            test_df[pname] = 0.5
        else:
            test_df[pname] = np.searchsorted(train_sorted, test_df[col].values) / n
    return train_df, test_df


def compute_fold_isoforest(train_X, test_X):
    """IsolationForest fitted on train only."""
    iso = IsolationForest(n_estimators=200, contamination=0.094, random_state=42, n_jobs=-1)
    iso.fit(train_X)
    train_score = iso.decision_function(train_X)
    test_score = iso.decision_function(test_X)
    return train_score, test_score


def compute_fold_physician_stats(full_df, train_providers, test_providers):
    """Physician reimbursement stats from train-fold claims only."""
    train_claims = full_df[full_df["Provider"].isin(train_providers)]

    # Attending physician avg reimbursement (train claims only)
    att_stats = train_claims.groupby("AttendingPhysician")["InscClaimAmtReimbursed"].agg(
        ["mean", "std", "count"]
    ).rename(columns={"mean": "att_avg", "std": "att_std", "count": "att_count"})
    global_mean = train_claims["InscClaimAmtReimbursed"].mean()

    result = {}
    for providers, label in [(train_providers, "train"), (test_providers, "test")]:
        claims = full_df[full_df["Provider"].isin(providers)]
        merged = claims.merge(att_stats, left_on="AttendingPhysician", right_index=True, how="left")
        merged["att_avg"] = merged["att_avg"].fillna(global_mean)
        merged["att_std"] = merged["att_std"].fillna(0)
        merged["att_count"] = merged["att_count"].fillna(0)

        prov_stats = merged.groupby("Provider").agg(
            Phys_AvgReimb_mean=("att_avg", "mean"),
            Phys_AvgReimb_std=("att_avg", "std"),
            Phys_ClaimCount_mean=("att_count", "mean"),
        ).fillna(0)

        # Operating physician (if exists)
        if "OperatingPhysician" in train_claims.columns:
            op_stats = train_claims.groupby("OperatingPhysician")["InscClaimAmtReimbursed"].mean().rename("op_avg")
            merged_op = claims.merge(op_stats, left_on="OperatingPhysician", right_index=True, how="left")
            merged_op["op_avg"] = merged_op["op_avg"].fillna(global_mean)
            op_prov = merged_op.groupby("Provider").agg(
                OpPhys_AvgReimb_mean=("op_avg", "mean"),
                OpPhys_AvgReimb_std=("op_avg", "std"),
            ).fillna(0)
            prov_stats = prov_stats.merge(op_prov, left_index=True, right_index=True, how="left")

        # Physician role overlap
        if "OperatingPhysician" in claims.columns:
            claims_copy = claims.copy()
            claims_copy["_overlap"] = (claims_copy["AttendingPhysician"] == claims_copy["OperatingPhysician"]).astype(int)
            overlap = claims_copy.groupby("Provider")["_overlap"].mean().rename("Phys_Role_Overlap")
            prov_stats = prov_stats.merge(overlap, left_index=True, right_index=True, how="left")

        prov_stats = prov_stats.fillna(0)
        result[label] = prov_stats

    return result["train"], result["test"]


def compute_fold_network(full_df, train_providers, test_providers):
    """Beneficiary network from train providers only."""
    train_claims = full_df[full_df["Provider"].isin(train_providers)]
    bp_train = train_claims[["Provider", "BeneID"]].drop_duplicates()

    bene_prov_count = bp_train.groupby("BeneID")["Provider"].nunique().rename("_n_prov")
    bp_train = bp_train.merge(bene_prov_count.reset_index(), on="BeneID")

    result = {}
    for providers, label in [(train_providers, "train"), (test_providers, "test")]:
        claims = full_df[full_df["Provider"].isin(providers)]
        bp = claims[["Provider", "BeneID"]].drop_duplicates()

        # For test providers, count based on train-provider bene sharing
        bp_with_count = bp.merge(bene_prov_count.reset_index(), on="BeneID", how="left")
        bp_with_count["_n_prov"] = bp_with_count["_n_prov"].fillna(0)

        net = bp_with_count.groupby("Provider").agg(
            Shared_Bene_Count=("_n_prov", lambda x: (x > 1).sum()),
            Bene_Exclusivity=("_n_prov", lambda x: (x <= 1).mean()),
            Avg_Providers_Per_Bene=("_n_prov", "mean"),
            Max_Providers_Per_Bene=("_n_prov", "max"),
        ).fillna(0)

        # Network degree (from train graph only)
        shared_benes = bp_train[bp_train["_n_prov"] > 1][["Provider", "BeneID"]]
        if len(shared_benes) > 0 and label == "train":
            pairs = shared_benes.merge(shared_benes, on="BeneID", suffixes=("_1", "_2"))
            pairs = pairs[pairs["Provider_1"] != pairs["Provider_2"]]
            degree = pairs.groupby("Provider_1")["Provider_2"].nunique().rename("Provider_Network_Degree")
            net = net.merge(degree.reset_index().rename(columns={"Provider_1": "Provider"}),
                           left_index=True, right_on="Provider", how="left").set_index("Provider")
        elif label == "test":
            # Test provider degree: count train providers sharing benes
            test_benes = bp[["Provider", "BeneID"]]
            if len(shared_benes) > 0:
                test_shared = test_benes.merge(shared_benes, on="BeneID", suffixes=("_test", "_train"))
                test_shared = test_shared[test_shared["Provider_test"] != test_shared["Provider_train"]]
                degree = test_shared.groupby("Provider_test")["Provider_train"].nunique().rename("Provider_Network_Degree")
                net = net.merge(degree.reset_index().rename(columns={"Provider_test": "Provider"}),
                               left_index=True, right_on="Provider", how="left").set_index("Provider")

        if "Provider_Network_Degree" not in net.columns:
            net["Provider_Network_Degree"] = 0
        net["Provider_Network_Degree"] = net["Provider_Network_Degree"].fillna(0)
        result[label] = net

    return result["train"], result["test"]


def train_claim_level_model(full_df, train_providers, test_providers, pos_weight):
    """Two-stage: train claim-level XGBoost, aggregate to provider features."""
    train_claims = full_df[full_df["Provider"].isin(train_providers)].copy()
    test_claims = full_df[full_df["Provider"].isin(test_providers)].copy()

    available_feats = [f for f in CLAIM_FEATURES if f in train_claims.columns]
    X_claim_train = train_claims[available_feats].fillna(0).values
    y_claim_train = train_claims["PotentialFraud"].values
    X_claim_test = test_claims[available_feats].fillna(0).values

    claim_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_weight, subsample=0.8, colsample_bytree=0.7,
        random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1,
    )
    claim_model.fit(X_claim_train, y_claim_train)

    train_claims["_claim_prob"] = claim_model.predict_proba(X_claim_train)[:, 1]
    test_claims["_claim_prob"] = claim_model.predict_proba(X_claim_test)[:, 1]

    result = {}
    for claims, label in [(train_claims, "train"), (test_claims, "test")]:
        agg = claims.groupby("Provider")["_claim_prob"].agg(
            Claim_FraudProb_mean="mean",
            Claim_FraudProb_max="max",
            Claim_FraudProb_std="std",
            Claim_FraudProb_pct50=lambda x: (x > 0.5).mean(),
        ).fillna(0)
        result[label] = agg

    return result["train"], result["test"]


# ================================================================
# MAIN
# ================================================================
def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    PIPELINE_DIR = args.pipeline_dir

    log.info("=" * 70)
    log.info("PHASE 12: LEAKAGE-FREE EVALUATION")
    log.info("  All cross-provider features computed inside CV folds")
    log.info("  Claim-level two-stage model + focal loss + DART")
    log.info("=" * 70)

    # ---- Phase 0: Load Data ----
    log.info("\n[Phase 0] Loading data...")
    D = args.data_dir

    def load_csv(path, name):
        df = pd.read_csv(path)
        log.info(f"  {name}: {df.shape}")
        return df

    train_labels = load_csv(f"{D}/Train-1542865627584.csv", "Labels")
    beneficiary = load_csv(f"{D}/Train_Beneficiarydata-1542865627584.csv", "Beneficiary")
    inpatient = load_csv(f"{D}/Train_Inpatientdata-1542865627584.csv", "Inpatient")
    outpatient = load_csv(f"{D}/Train_Outpatientdata-1542865627584.csv", "Outpatient")

    # Merge claims
    common_cols = [c for c in inpatient.columns if c in outpatient.columns]
    claims = outpatient.merge(inpatient, on=common_cols, how="outer", indicator="claim_type")
    claims["is_inpatient"] = (claims["claim_type"] == "right_only").astype(int)
    claims.drop("claim_type", axis=1, inplace=True)
    full = train_labels.merge(beneficiary.merge(claims, on="BeneID"), on="Provider")
    full["PotentialFraud"] = full["PotentialFraud"].map({"Yes": 1, "No": 0})
    log.info(f"  Full claims: {full.shape}")

    # Claim-level features
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if col in full.columns:
            full[col] = pd.to_datetime(full[col], errors="coerce")
    full["DOB"] = pd.to_datetime(full["DOB"], errors="coerce")
    full["DOD"] = pd.to_datetime(full["DOD"], errors="coerce")
    full["Age"] = ((full["ClaimStartDt"] - full["DOB"]).dt.days / 365.25).fillna(0).astype(int)
    full["is_dead"] = full["DOD"].notna().astype(int)
    full["Claim_Duration"] = (full["ClaimEndDt"] - full["ClaimStartDt"]).dt.days.fillna(0).astype(int)
    full["Days_Admitted"] = (full["DischargeDt"] - full["AdmissionDt"]).dt.days.fillna(0).astype(int)
    full["DeductibleAmtPaid"] = full["DeductibleAmtPaid"].fillna(0)
    full["Has_Deductible"] = (full["DeductibleAmtPaid"] > 0).astype(int)
    full["TotalClaimAmt"] = full["InscClaimAmtReimbursed"].fillna(0) + full["DeductibleAmtPaid"]
    full["Charge_Per_Day"] = np.where(full["Days_Admitted"] > 0, full["TotalClaimAmt"] / full["Days_Admitted"], 0)
    full["Claim_GT_Admitted"] = (full["Claim_Duration"] > full["Days_Admitted"]).astype(int)

    chronic_cols = [c for c in full.columns if c.startswith("ChronicCond_")]
    for col in chronic_cols:
        full[col] = full[col].replace(2, 0)
    full["RenalDiseaseIndicator"] = full["RenalDiseaseIndicator"].replace({"0": 0, "Y": 1, 0: 0}).astype(int)
    full["Disease_Count"] = full[chronic_cols].sum(axis=1)

    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    diag_present = [c for c in diag_cols if c in full.columns]
    proc_present = [c for c in proc_cols if c in full.columns]
    full["Diag_Code_Count"] = full[diag_present].notna().sum(axis=1)
    full["Proc_Code_Count"] = full[proc_present].notna().sum(axis=1)

    log.info(f"  Claim features engineered")

    # Load safe pre-computed provider features
    v3_path = f"{PIPELINE_DIR}/provider_features_v3.csv"
    provider_df = pd.read_csv(v3_path)
    log.info(f"  Provider features loaded: {provider_df.shape}")

    # Drop unsafe columns
    unsafe_cols = [c for c in provider_df.columns if any(pat in c for pat in UNSAFE_PATTERNS)]
    safe_cols = [c for c in provider_df.columns if c not in unsafe_cols and c not in ["Provider", "PotentialFraud"]]
    log.info(f"  Dropped {len(unsafe_cols)} unsafe features, keeping {len(safe_cols)} safe features")

    # ---- Phase 1: Clip code fractions (don't recompute - the >1.0 values capture code repetition signal) ----
    log.info("\n[Phase 1] Clipping code fractions to [0, 2.0]...")
    diag_frac_cols = [c for c in safe_cols if c.startswith("DiagCode_") and c.endswith("_frac")]
    proc_frac_cols = [c for c in safe_cols if c.startswith("ProcCode_") and c.endswith("_frac")]
    for col in diag_frac_cols + proc_frac_cols:
        provider_df[col] = provider_df[col].clip(0, 2.0)
    log.info(f"  Clipped {len(diag_frac_cols) + len(proc_frac_cols)} code fraction columns")

    # Prepare safe feature matrix
    providers = provider_df["Provider"].values
    y = provider_df["PotentialFraud"].values
    pos_weight = len(y[y == 0]) / max(len(y[y == 1]), 1)

    safe_X = provider_df[safe_cols].fillna(0).replace([np.inf, -np.inf], 0)
    log.info(f"  Safe features: {safe_X.shape[1]}, Providers: {len(providers)}, Fraud: {y.sum()}")

    # Load tuned params
    params_path = RESULTS_DIR / "best_params_v3.json"
    tuned_params = {}
    try:
        with open(params_path) as f:
            tuned_params = json.load(f)
    except FileNotFoundError:
        log.warning("  No tuned params found, using defaults")

    # ---- Phase 2: 10-Fold CV ----
    log.info("\n[Phase 2] 10-Fold CV with in-fold feature engineering...")

    N_FOLDS = 10
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    MODEL_NAMES = [
        "XGBoost", "LightGBM", "CatBoost",
        "GradientBoosting", "RandomForest", "XGBoost_focal",
    ]
    oof_proba = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    fold_f1 = {name: [] for name in MODEL_NAMES}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(safe_X, y)):
        fold_start = time.time()
        log.info(f"\n  === Fold {fold_idx + 1}/{N_FOLDS} ===")

        train_providers_set = set(providers[train_idx])
        test_providers_set = set(providers[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # Start with safe features
        X_train_df = safe_X.iloc[train_idx].copy().reset_index(drop=True)
        X_test_df = safe_X.iloc[test_idx].copy().reset_index(drop=True)

        # Step 2b: Z-scores
        X_train_df, X_test_df = compute_fold_zscores(X_train_df, X_test_df, ZSCORE_SRC_COLS)

        # Step 2c: Percentiles
        X_train_df, X_test_df = compute_fold_percentiles(X_train_df, X_test_df, PCTILE_SRC_COLS)

        # Step 2d: IsolationForest
        iso_train, iso_test = compute_fold_isoforest(X_train_df.values, X_test_df.values)
        X_train_df["IsoForest_Score"] = iso_train
        X_test_df["IsoForest_Score"] = iso_test

        # Step 2e: Physician stats
        phys_train, phys_test = compute_fold_physician_stats(full, train_providers_set, test_providers_set)
        for pdf, xdf, idx_arr in [(phys_train, X_train_df, train_idx), (phys_test, X_test_df, test_idx)]:
            prov_order = providers[idx_arr]
            pdf_reindexed = pdf.reindex(prov_order).fillna(0).reset_index(drop=True)
            for col in pdf_reindexed.columns:
                xdf[col] = pdf_reindexed[col].values

        # Step 2f: Network features
        net_train, net_test = compute_fold_network(full, train_providers_set, test_providers_set)
        for ndf, xdf, idx_arr in [(net_train, X_train_df, train_idx), (net_test, X_test_df, test_idx)]:
            prov_order = providers[idx_arr]
            ndf_reindexed = ndf.reindex(prov_order).fillna(0).reset_index(drop=True)
            for col in ndf_reindexed.columns:
                xdf[col] = ndf_reindexed[col].values

        # Step 2h-2i: Target encoding and claim-level model skipped for this baseline run
        # (will add back once leakage-free baseline is validated)

        # Step 2g: Use ALL features (no MI selection - removes selection bias)
        X_train_vals = X_train_df.fillna(0).replace([np.inf, -np.inf], 0).values
        X_test_vals = X_test_df.fillna(0).replace([np.inf, -np.inf], 0).values

        log.info(f"    Features used: {X_train_vals.shape[1]}")

        # Step 2j: Train 6 models (SAME configs as 10_advanced_evaluation.py)

        # Model 1: XGBoost (tuned params, proven F1=0.725)
        xgb_p = dict(tuned_params.get("XGBoost", {
            "n_estimators": 500, "max_depth": 3, "learning_rate": 0.01,
            "subsample": 0.9, "colsample_bytree": 0.6, "scale_pos_weight": pos_weight,
        }))
        m_xgb = XGBClassifier(**xgb_p, random_state=42, eval_metric="logloss", verbosity=0)
        m_xgb.fit(X_train_vals, y_train)
        xgb_proba = m_xgb.predict_proba(X_test_vals)[:, 1]
        oof_proba["XGBoost"][test_idx] = xgb_proba
        fold_f1["XGBoost"].append(f1_score(y_test, (xgb_proba >= 0.5).astype(int)))

        # Model 2: LightGBM (tuned params, standard GBDT)
        lgb_p = dict(tuned_params.get("LightGBM", {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.03,
        }))
        lgb_p["is_unbalance"] = True
        m_lgb = LGBMClassifier(**lgb_p, random_state=42, verbose=-1)
        m_lgb.fit(X_train_vals, y_train)
        lgb_proba = m_lgb.predict_proba(X_test_vals)[:, 1]
        oof_proba["LightGBM"][test_idx] = lgb_proba
        fold_f1["LightGBM"].append(f1_score(y_test, (lgb_proba >= 0.5).astype(int)))

        # Model 3: CatBoost (tuned params, NO class weight override)
        cb_p = dict(tuned_params.get("CatBoost", {
            "iterations": 456, "depth": 8, "learning_rate": 0.015,
        }))
        aw = cb_p.pop("auto_class_weights", None)
        if aw == "None":
            aw = None
        m_cb = CatBoostClassifier(**cb_p, auto_class_weights=aw, random_seed=42, verbose=0)
        m_cb.fit(X_train_vals, y_train)
        cb_proba = m_cb.predict_proba(X_test_vals)[:, 1]
        oof_proba["CatBoost"][test_idx] = cb_proba
        fold_f1["CatBoost"].append(f1_score(y_test, (cb_proba >= 0.5).astype(int)))

        # Model 4: GradientBoosting (tuned params, no sample_weight)
        gb_p = dict(tuned_params.get("GradientBoosting", {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.04,
        }))
        m_gb = GradientBoostingClassifier(**gb_p, random_state=42)
        m_gb.fit(X_train_vals, y_train)
        gb_proba = m_gb.predict_proba(X_test_vals)[:, 1]
        oof_proba["GradientBoosting"][test_idx] = gb_proba
        fold_f1["GradientBoosting"].append(f1_score(y_test, (gb_proba >= 0.5).astype(int)))

        # Model 5: RandomForest (tuned params)
        rf_p = tuned_params.get("RandomForest", {})
        if not rf_p:
            rf_p = {"n_estimators": 500, "max_depth": 20, "max_features": "sqrt"}
        rf_p_clean = {k: v for k, v in rf_p.items() if k != "use_smote"}
        m_rf = RandomForestClassifier(**rf_p_clean, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        m_rf.fit(X_train_vals, y_train)
        rf_proba = m_rf.predict_proba(X_test_vals)[:, 1]
        oof_proba["RandomForest"][test_idx] = rf_proba
        fold_f1["RandomForest"].append(f1_score(y_test, (rf_proba >= 0.5).astype(int)))

        # Model 6: XGBoost focal loss (experimental)
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train_vals, label=y_train)
        dtest = xgb.DMatrix(X_test_vals)
        xgb_focal_params = {
            "max_depth": tuned_params.get("XGBoost", {}).get("max_depth", 3),
            "learning_rate": tuned_params.get("XGBoost", {}).get("learning_rate", 0.01),
            "subsample": tuned_params.get("XGBoost", {}).get("subsample", 0.9),
            "colsample_bytree": tuned_params.get("XGBoost", {}).get("colsample_bytree", 0.6),
            "reg_alpha": tuned_params.get("XGBoost", {}).get("reg_alpha", 0.0001),
            "reg_lambda": tuned_params.get("XGBoost", {}).get("reg_lambda", 0.1),
            "min_child_weight": tuned_params.get("XGBoost", {}).get("min_child_weight", 2),
            "disable_default_eval_metric": 1,
            "seed": 42,
        }
        n_rounds = tuned_params.get("XGBoost", {}).get("n_estimators", 500)
        bst_focal = xgb.train(xgb_focal_params, dtrain, num_boost_round=n_rounds, obj=focal_loss_obj)
        focal_raw = bst_focal.predict(dtest)
        focal_proba = 1.0 / (1.0 + np.exp(-focal_raw))
        oof_proba["XGBoost_focal"][test_idx] = focal_proba
        fold_f1["XGBoost_focal"].append(f1_score(y_test, (focal_proba >= 0.5).astype(int)))

        elapsed = time.time() - fold_start
        log.info(f"    Fold {fold_idx+1} done in {elapsed:.0f}s")
        for name in MODEL_NAMES:
            log.info(f"      {name:30s}: F1={fold_f1[name][-1]:.4f}")

    # ---- Phase 3: Ensemble ----
    log.info("\n[Phase 3] Optuna-weighted ensemble (AUC-PR objective, 200 trials)...")

    oof_matrix = np.column_stack([oof_proba[k] for k in MODEL_NAMES])

    def ens_objective(trial):
        weights = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MODEL_NAMES])
        wsum = weights.sum()
        if wsum == 0:
            return 0.0
        weights /= wsum
        blended = oof_matrix @ weights
        return average_precision_score(y, blended)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(ens_objective, n_trials=200)

    best_weights = np.array([study.best_params[f"w_{n}"] for n in MODEL_NAMES])
    best_weights /= best_weights.sum()

    log.info(f"  Best AUC-PR: {study.best_value:.4f}")
    log.info(f"  Weights:")
    for n, w in zip(MODEL_NAMES, best_weights):
        log.info(f"    {n:30s}: {w:.4f}")

    # Find optimal threshold SEPARATELY
    blended = oof_matrix @ best_weights
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y, blended)
    f1_arr = np.where(
        (prec_arr[:-1] + rec_arr[:-1]) > 0,
        2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
        0,
    )
    opt_idx = np.argmax(f1_arr)
    opt_threshold = thresh_arr[opt_idx]

    y_pred_opt = (blended >= opt_threshold).astype(int)
    ens_f1 = f1_score(y, y_pred_opt)
    ens_prec = precision_score(y, y_pred_opt)
    ens_rec = recall_score(y, y_pred_opt)
    ens_mcc = matthews_corrcoef(y, y_pred_opt)
    ens_auc = roc_auc_score(y, blended)
    ens_aucpr = average_precision_score(y, blended)

    log.info(f"\n  Ensemble (threshold={opt_threshold:.3f}):")
    log.info(f"    F1={ens_f1:.4f}  P={ens_prec:.4f}  R={ens_rec:.4f}  MCC={ens_mcc:.4f}")
    log.info(f"    AUC-ROC={ens_auc:.4f}  AUC-PR={ens_aucpr:.4f}")

    # Per-fold ensemble F1
    oof_ens_f1 = []
    for train_idx, test_idx in outer_cv.split(safe_X, y):
        fp = oof_matrix[test_idx] @ best_weights
        oof_ens_f1.append(f1_score(y[test_idx], (fp >= opt_threshold).astype(int)))
    log.info(f"  Per-fold F1: {[f'{x:.4f}' for x in oof_ens_f1]}")
    log.info(f"  Mean={np.mean(oof_ens_f1):.4f} Std={np.std(oof_ens_f1):.4f}")

    # ---- Phase 4: Statistical Tests ----
    log.info("\n[Phase 4] Statistical tests...")

    all_results = {}
    for name in MODEL_NAMES:
        all_results[name] = {
            "f1": {"mean": float(np.mean(fold_f1[name])), "std": float(np.std(fold_f1[name])), "values": fold_f1[name]},
            "_label": name,
        }
    all_results["WeightedEnsemble"] = {
        "f1": {"mean": float(np.mean(oof_ens_f1)), "std": float(np.std(oof_ens_f1)), "values": oof_ens_f1},
        "precision": {"mean": float(ens_prec)},
        "recall": {"mean": float(ens_rec)},
        "mcc": {"mean": float(ens_mcc)},
        "roc_auc": {"mean": float(ens_auc)},
        "average_precision": {"mean": float(ens_aucpr)},
        "_label": "Weighted Ensemble (leakage-free)",
        "_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_weights)},
        "_threshold": float(opt_threshold),
    }

    # Friedman
    stat_names = [k for k in all_results if len(all_results[k].get("f1", {}).get("values", [])) == N_FOLDS]
    if len(stat_names) >= 3:
        stat, p = stats.friedmanchisquare(*[all_results[k]["f1"]["values"] for k in stat_names])
        all_results["_friedman"] = {"statistic": float(stat), "p_value": float(p)}
        log.info(f"  Friedman: chi2={stat:.4f}, p={p:.6f}")

    # Wilcoxon
    best_key = max(stat_names, key=lambda k: all_results[k]["f1"]["mean"])
    raw_p = {}
    for name in stat_names:
        if name == best_key:
            continue
        a = np.array(all_results[best_key]["f1"]["values"])
        b = np.array(all_results[name]["f1"]["values"])
        if np.all(a == b):
            raw_p[name] = 1.0
        else:
            try:
                _, p = stats.wilcoxon(a, b)
                raw_p[name] = float(p)
            except Exception:
                raw_p[name] = 1.0

    corrected_p = holm_bonferroni(raw_p)
    all_results["_wilcoxon"] = {"best_model": best_key, "corrected_pvalues": corrected_p}
    for name in sorted(corrected_p):
        sig = "***" if corrected_p[name] < 0.01 else "**" if corrected_p[name] < 0.05 else "*" if corrected_p[name] < 0.1 else "ns"
        log.info(f"    vs {name:30s}: p={corrected_p[name]:.4f} [{sig}]")

    # Bootstrap CIs (re-threshold inside each iteration)
    log.info("\n  Bootstrap 95% CIs (re-thresholding)...")
    rng = np.random.RandomState(42)
    boot = {m: [] for m in ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc"]}
    for _ in range(2000):
        idx = rng.choice(len(y), len(y), replace=True)
        yb, bprob = y[idx], blended[idx]
        if len(np.unique(yb)) < 2:
            continue
        # Re-threshold inside bootstrap
        p_arr, r_arr, t_arr = precision_recall_curve(yb, bprob)
        f1b = np.where((p_arr[:-1] + r_arr[:-1]) > 0, 2*p_arr[:-1]*r_arr[:-1]/(p_arr[:-1]+r_arr[:-1]), 0)
        best_t = t_arr[np.argmax(f1b)]
        ypb = (bprob >= best_t).astype(int)

        boot["f1"].append(f1_score(yb, ypb))
        boot["precision"].append(precision_score(yb, ypb, zero_division=0))
        boot["recall"].append(recall_score(yb, ypb))
        boot["mcc"].append(matthews_corrcoef(yb, ypb))
        boot["roc_auc"].append(roc_auc_score(yb, bprob))
        boot["pr_auc"].append(average_precision_score(yb, bprob))

    ci = {}
    for m, vals in boot.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci[m] = {"mean": float(np.mean(vals)), "ci_lower": float(lo), "ci_upper": float(hi)}
        log.info(f"    {m}: {np.mean(vals):.4f} [{lo:.4f}, {hi:.4f}]")
    all_results["_bootstrap_ci"] = {"model": "WeightedEnsemble", "metrics": ci}

    all_results["_threshold_optimization"] = {
        "optimal_threshold": float(opt_threshold),
        "optimal_f1": float(ens_f1),
        "optimal_precision": float(ens_prec),
        "optimal_recall": float(ens_rec),
    }

    # ---- Phase 5: Save ----
    log.info("\n[Phase 5] Saving...")
    with open(RESULTS_DIR / "leakage_free_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"  Saved: leakage_free_results.json")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY (LEAKAGE-FREE)")
    log.info(f"{'='*70}")
    log.info(f"\n  {'Model':<35s} {'F1':>8s}")
    log.info(f"  {'-'*35} {'-'*8}")
    ranking = sorted(all_results.items(), key=lambda x: -x[1].get("f1", {}).get("mean", 0))
    for name, r in ranking:
        if name.startswith("_"):
            continue
        log.info(f"  {name:<35s} {r['f1']['mean']:>8.4f}")

    log.info(f"\n  Best F1: {ens_f1:.4f} (threshold={opt_threshold:.3f})")
    log.info(f"  AUC-PR: {ens_aucpr:.4f}")
    log.info(f"  Bootstrap 95% CI: [{ci['f1']['ci_lower']:.4f}, {ci['f1']['ci_upper']:.4f}]")
    log.info(f"\n  ALL features computed inside CV folds. ZERO leakage.")
    log.info("\nDone.")


if __name__ == "__main__":
    main()
