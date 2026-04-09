"""
Phase 7: Attempt to improve results by:
1. Rebuild ensembles WITHOUT SMOTE (fix from 05_full_evaluation.py bug)
2. Add interaction features
3. Re-tune LightGBM with more budget
4. Try CatBoost stacking

Runs locally - no GPU needed for 5,410 samples.
"""
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

np.random.seed(42)


def main():
    RESULTS_DIR = Path("results")

    log.info("=" * 60)
    log.info("PHASE 7: IMPROVING RESULTS")
    log.info("=" * 60)

    # Load data
    df = pd.read_csv("provider_features.csv")
    feature_cols = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    X = df[feature_cols].values
    y = df["PotentialFraud"].values
    pos_weight = len(y[y == 0]) / len(y[y == 1])

    log.info(f"  {X.shape[0]} providers, {X.shape[1]} features, fraud={y.mean()*100:.1f}%")

    # Load best params
    with open(RESULTS_DIR / "best_params.json") as f:
        optuna_results = json.load(f)

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "average_precision": "average_precision",
        "mcc": make_scorer(matthews_corrcoef),
    }

    # ---- 1. Rebuild tuned individual models (NO SMOTE) ----
    log.info("\n[1/4] Rebuilding tuned individual models (no SMOTE)...")

    xgb_params = dict(optuna_results["models"]["XGBoost"]["params"])
    xgb_params.pop("use_smote", None)
    xgb = XGBClassifier(**xgb_params, random_state=42, eval_metric="logloss", verbosity=0)

    lgb_params = dict(optuna_results["models"]["LightGBM"]["params"])
    lgb_params.pop("use_smote", None)
    lgb = LGBMClassifier(**lgb_params, random_state=42, verbose=-1)

    gb_params = dict(optuna_results["models"]["GradientBoosting"]["params"])
    gb_params.pop("use_smote", None)
    gb = GradientBoostingClassifier(**gb_params, random_state=42)

    rf_params = dict(optuna_results["models"]["RandomForest"]["params"])
    rf_params.pop("use_smote", None)
    rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)

    # ---- 2. Ensembles WITHOUT SMOTE (the fix) ----
    log.info("\n[2/4] Building ensembles WITHOUT SMOTE (the improvement)...")

    estimators = [
        ("xgb", XGBClassifier(**xgb_params, random_state=42, eval_metric="logloss", verbosity=0)),
        ("lgb", LGBMClassifier(**lgb_params, random_state=42, verbose=-1)),
        ("gb", GradientBoostingClassifier(**gb_params, random_state=42)),
        ("rf", RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)),
    ]

    # SoftVoting without SMOTE
    soft_vote_no_smote = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

    t0 = time.time()
    cv_soft = cross_validate(soft_vote_no_smote, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    elapsed = time.time() - t0
    soft_f1 = float(np.mean(cv_soft["test_f1"]))
    soft_pr = float(np.mean(cv_soft["test_average_precision"]))
    soft_mcc = float(np.mean(cv_soft["test_mcc"]))
    log.info(f"  SoftVoting (NO SMOTE): F1={soft_f1:.4f}, AUC-PR={soft_pr:.4f}, MCC={soft_mcc:.4f} ({elapsed:.1f}s)")

    # Stacking without SMOTE
    stack_no_smote = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=False, n_jobs=1,
    )

    t0 = time.time()
    cv_stack = cross_validate(stack_no_smote, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    elapsed = time.time() - t0
    stack_f1 = float(np.mean(cv_stack["test_f1"]))
    stack_pr = float(np.mean(cv_stack["test_average_precision"]))
    stack_mcc = float(np.mean(cv_stack["test_mcc"]))
    log.info(f"  Stacking (NO SMOTE):   F1={stack_f1:.4f}, AUC-PR={stack_pr:.4f}, MCC={stack_mcc:.4f} ({elapsed:.1f}s)")

    # ---- 3. Feature interactions ----
    log.info("\n[3/4] Testing feature interactions...")

    # Add top interaction features based on SHAP
    X_enhanced = np.column_stack([
        X,
        df["Claims_Per_Bene"].values * df["Dead_Patient_Ratio"].values,
        df["Reimburse_max"].values * df["Has_Deductible"].values,
        df["ClaimDur_max"].values / (df["Claim_Count"].values + 1),
        np.log1p(df["Reimburse_max"].values) * df["Inpatient_Ratio"].values,
        df["Reimburse_std"].values / (df["InscClaimAmtReimbursed"].values + 1),
    ])

    t0 = time.time()
    cv_enhanced = cross_validate(xgb, X_enhanced, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    elapsed = time.time() - t0
    enhanced_f1 = float(np.mean(cv_enhanced["test_f1"]))
    enhanced_pr = float(np.mean(cv_enhanced["test_average_precision"]))
    enhanced_mcc = float(np.mean(cv_enhanced["test_mcc"]))
    log.info(f"  XGBoost + 5 interactions: F1={enhanced_f1:.4f}, AUC-PR={enhanced_pr:.4f}, MCC={enhanced_mcc:.4f} ({elapsed:.1f}s)")

    # Stacking with enhanced features
    t0 = time.time()
    cv_stack_enh = cross_validate(stack_no_smote, X_enhanced, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    elapsed = time.time() - t0
    stack_enh_f1 = float(np.mean(cv_stack_enh["test_f1"]))
    stack_enh_pr = float(np.mean(cv_stack_enh["test_average_precision"]))
    stack_enh_mcc = float(np.mean(cv_stack_enh["test_mcc"]))
    log.info(f"  Stacking + interactions: F1={stack_enh_f1:.4f}, AUC-PR={stack_enh_pr:.4f}, MCC={stack_enh_mcc:.4f} ({elapsed:.1f}s)")

    # ---- 4. Threshold optimization on best ----
    log.info("\n[4/4] Threshold optimization on best configuration...")

    # Find the best configuration
    configs = {
        "XGBoost (52 feat)": (xgb, X, 0.6842),
        "XGBoost + interactions": (xgb, X_enhanced, enhanced_f1),
        "SoftVoting no SMOTE": (soft_vote_no_smote, X, soft_f1),
        "Stacking no SMOTE": (stack_no_smote, X, stack_f1),
        "Stacking + interactions": (stack_no_smote, X_enhanced, stack_enh_f1),
    }

    best_name = max(configs.keys(), key=lambda k: configs[k][2])
    best_model, best_X, best_f1 = configs[best_name]
    log.info(f"  Best configuration: {best_name} (F1={best_f1:.4f})")

    # Get cross-validated probabilities for threshold optimization
    y_proba = cross_val_predict(best_model, best_X, y, cv=outer_cv, method="predict_proba")[:, 1]

    best_threshold = 0.5
    best_thresh_f1 = 0
    for t in np.arange(0.15, 0.85, 0.005):
        yp = (y_proba >= t).astype(int)
        if yp.sum() > 0:
            f1 = f1_score(y, yp)
            if f1 > best_thresh_f1:
                best_thresh_f1 = f1
                best_threshold = t

    y_opt = (y_proba >= best_threshold).astype(int)
    opt_prec = precision_score(y, y_opt)
    opt_rec = recall_score(y, y_opt)
    opt_mcc = matthews_corrcoef(y, y_opt)

    log.info(f"  Optimal threshold: {best_threshold:.3f}")
    log.info(f"  F1={best_thresh_f1:.4f}, Precision={opt_prec:.4f}, Recall={opt_rec:.4f}, MCC={opt_mcc:.4f}")

    # ---- Summary ----
    log.info("\n" + "=" * 80)
    log.info("IMPROVEMENT SUMMARY")
    log.info("=" * 80)
    log.info(f"  Previous best (XGBoost, no SMOTE):     F1=0.6842")
    log.info(f"  Previous best (threshold=0.348):       F1=0.6937")
    log.info(f"  SoftVoting (NO SMOTE, fixed):           F1={soft_f1:.4f}")
    log.info(f"  Stacking (NO SMOTE, fixed):             F1={stack_f1:.4f}")
    log.info(f"  XGBoost + interactions:                  F1={enhanced_f1:.4f}")
    log.info(f"  Stacking + interactions:                 F1={stack_enh_f1:.4f}")
    log.info(f"  Best + optimal threshold:                F1={best_thresh_f1:.4f}")
    log.info("=" * 80)

    # Save improvement results
    improvement = {
        "previous_best_f1": 0.6842,
        "previous_threshold_f1": 0.6937,
        "soft_voting_no_smote_f1": soft_f1,
        "stacking_no_smote_f1": stack_f1,
        "xgboost_interactions_f1": enhanced_f1,
        "stacking_interactions_f1": stack_enh_f1,
        "best_config": best_name,
        "best_threshold": float(best_threshold),
        "best_threshold_f1": float(best_thresh_f1),
        "best_threshold_precision": float(opt_prec),
        "best_threshold_recall": float(opt_rec),
        "best_threshold_mcc": float(opt_mcc),
    }

    with open(RESULTS_DIR / "improvement_results.json", "w") as f:
        json.dump(improvement, f, indent=2)

    log.info(f"\nResults saved to {RESULTS_DIR}/improvement_results.json")


if __name__ == "__main__":
    main()
