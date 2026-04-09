"""
Phase 8: Improved Evaluation - Fix SMOTE Bug + New Models + New Features
All methods pre-June 2024.

Parts:
  A: Fix ensemble SMOTE bug (52 features) - SoftVoting & Stacking without SMOTE
  B: New imbalanced learning models (52 features) - EasyEnsemble, BalancedRandomForest
  C: Calibration (52 features) - CalibratedClassifierCV on best XGBoost
  D: OOF Stacking (52 features) - explicit out-of-fold stacking with XGBoost meta
  E: New features (60 features) - re-run best configs on provider_features_v2.csv
  F: Threshold optimization on best configuration
  G: Statistical tests + bootstrap CIs across all configurations
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
from scipy import stats
from scipy.stats import skew as scipy_skew
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
)
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("improved_evaluation.log"),
    ],
)
log = logging.getLogger(__name__)

np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 8: Improved Evaluation")
    p.add_argument("--data-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    return p.parse_args()


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to raw p-values."""
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    corrected = {}
    for rank, (name, p) in enumerate(items):
        corrected[name] = min(p * (m - rank), 1.0)
    return corrected


def build_model_from_params(name, params):
    """Build a model instance from Optuna params (no SMOTE wrapping)."""
    p = dict(params)
    p.pop("use_smote", None)

    if name == "XGBoost":
        return XGBClassifier(**p, random_state=42, eval_metric="logloss", verbosity=0)
    elif name == "LightGBM":
        return LGBMClassifier(**p, random_state=42, verbose=-1)
    elif name == "GradientBoosting":
        return GradientBoostingClassifier(**p, random_state=42)
    elif name == "CatBoost":
        if p.get("auto_class_weights") == "None":
            p["auto_class_weights"] = None
        return CatBoostClassifier(**p, random_seed=42, verbose=0)
    elif name == "RandomForest":
        return RandomForestClassifier(**p, random_state=42, n_jobs=-1)
    elif name == "LogisticRegression":
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**p, max_iter=2000, class_weight="balanced", random_state=42)),
        ])
    else:
        raise ValueError(f"Unknown model: {name}")


def run_cv(model, X, y, outer_cv, scoring, name=""):
    """Run 10-fold CV and return results dict."""
    t0 = time.time()
    cv_results = cross_validate(
        model, X, y, cv=outer_cv, scoring=scoring,
        return_train_score=False, n_jobs=1,
    )
    elapsed = time.time() - t0
    result = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        result[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "values": scores.tolist(),
        }
    result["time_seconds"] = elapsed
    log.info(
        f"  {name:35s} | F1={result['f1']['mean']:.4f}+/-{result['f1']['std']:.4f} | "
        f"AUC-PR={result['average_precision']['mean']:.4f} | "
        f"MCC={result['mcc']['mean']:.4f} | {elapsed:.1f}s"
    )
    return result


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 70)
    log.info("PHASE 8: IMPROVED EVALUATION")
    log.info("  Fix SMOTE ensemble bug + New models + New features")
    log.info("=" * 70)

    # ---- Load data and params ----
    log.info("[Setup] Loading data and tuned parameters...")
    data_52_path = f"{args.data_dir}/provider_features.csv"
    data_60_path = f"{args.data_dir}/provider_features_v2.csv"
    params_path = RESULTS_DIR / "best_params.json"
    baseline_path = RESULTS_DIR / "tuned_cv_results.json"

    try:
        df_52 = pd.read_csv(data_52_path)
    except FileNotFoundError:
        log.error(f"File not found: {data_52_path}")
        sys.exit(1)

    try:
        with open(params_path) as f:
            optuna_results = json.load(f)
    except FileNotFoundError:
        log.error(f"File not found: {params_path}. Run 04_optuna_tuning.py first.")
        sys.exit(1)

    # Load baseline results for comparison
    baseline_results = {}
    try:
        with open(baseline_path) as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        log.warning("No baseline results found, will skip comparison.")

    feature_cols_52 = [c for c in df_52.columns if c not in ["Provider", "PotentialFraud"]]
    X_52 = df_52[feature_cols_52].values
    y = df_52["PotentialFraud"].values

    log.info(f"  52-feature dataset: {X_52.shape[0]} providers, {X_52.shape[1]} features")
    log.info(f"  Fraud rate: {y.mean()*100:.1f}%")

    N_FOLDS = 10
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "mcc": make_scorer(matthews_corrcoef),
    }

    all_results = {}

    # Record baseline XGBoost for reference
    if "XGBoost" in baseline_results:
        all_results["XGBoost_baseline"] = baseline_results["XGBoost"]
        all_results["XGBoost_baseline"]["_label"] = "XGBoost (tuned, baseline)"
        log.info(f"  Baseline XGBoost F1: {baseline_results['XGBoost']['f1']['mean']:.4f}")

    # Record buggy ensemble results for comparison
    if "SoftVoting" in baseline_results:
        all_results["SoftVoting_buggy"] = baseline_results["SoftVoting"]
        all_results["SoftVoting_buggy"]["_label"] = "SoftVoting (SMOTE bug)"
        log.info(f"  Buggy SoftVoting F1: {baseline_results['SoftVoting']['f1']['mean']:.4f}")
    if "Stacking" in baseline_results:
        all_results["Stacking_buggy"] = baseline_results["Stacking"]
        all_results["Stacking_buggy"]["_label"] = "Stacking (SMOTE bug)"
        log.info(f"  Buggy Stacking F1: {baseline_results['Stacking']['f1']['mean']:.4f}")

    # ---- Build individual tuned models (no SMOTE) ----
    tuned_estimators = {}
    for model_name, model_info in optuna_results["models"].items():
        params = dict(model_info["params"])
        tuned_estimators[model_name] = build_model_from_params(model_name, params)

    # ================================================================
    # PART A: Fix Ensemble SMOTE Bug (52 features)
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART A: Fix Ensemble SMOTE Bug (52 features)")
    log.info("  All Optuna models chose use_smote=false.")
    log.info("  05_full_evaluation.py hardcoded SMOTE-ENN in ensembles.")
    log.info("  Fix: build ensembles WITHOUT SMOTE wrapping.")
    log.info("=" * 70)

    # Get top models for ensemble by Optuna AUC-PR
    model_ranking = sorted(
        optuna_results["models"].items(),
        key=lambda x: x[1]["best_auc_pr"],
        reverse=True,
    )
    top_models = [name for name, _ in model_ranking[:5]]
    log.info(f"  Top 5 for ensembles: {top_models}")

    # Build fresh estimators for ensemble
    ensemble_list = []
    for name in top_models:
        params = dict(optuna_results["models"][name]["params"])
        est = build_model_from_params(name, params)
        # VotingClassifier needs (name, estimator) pairs; avoid pipeline wrappers
        # LogisticRegression is already wrapped in ImbPipeline with scaler
        ensemble_list.append((name.lower()[:4], est))

    # A1: Fixed SoftVoting (no SMOTE)
    log.info("\n  [A1] SoftVoting FIXED (no SMOTE)...")
    soft_vote_fixed = VotingClassifier(estimators=ensemble_list, voting="soft", n_jobs=1)
    result_sv_fixed = run_cv(soft_vote_fixed, X_52, y, outer_cv, scoring, "SoftVoting_fixed_52")
    result_sv_fixed["use_smote"] = False
    result_sv_fixed["_label"] = "SoftVoting FIXED (no SMOTE, 52 feat)"
    all_results["SoftVoting_fixed_52"] = result_sv_fixed

    # A2: Fixed Stacking with XGBoost meta-learner (no SMOTE)
    log.info("\n  [A2] Stacking FIXED (XGBoost meta, no SMOTE)...")
    xgb_meta = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    stack_fixed = StackingClassifier(
        estimators=ensemble_list,
        final_estimator=xgb_meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=False, n_jobs=1,
    )
    result_stack_fixed = run_cv(stack_fixed, X_52, y, outer_cv, scoring, "Stacking_XGBmeta_fixed_52")
    result_stack_fixed["use_smote"] = False
    result_stack_fixed["_label"] = "Stacking FIXED (XGB meta, no SMOTE, 52 feat)"
    all_results["Stacking_XGBmeta_fixed_52"] = result_stack_fixed

    # A3: Fixed Stacking with LR meta-learner for comparison
    log.info("\n  [A3] Stacking FIXED (LR meta, no SMOTE)...")
    stack_lr_fixed = StackingClassifier(
        estimators=ensemble_list,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=False, n_jobs=1,
    )
    result_stack_lr = run_cv(stack_lr_fixed, X_52, y, outer_cv, scoring, "Stacking_LRmeta_fixed_52")
    result_stack_lr["use_smote"] = False
    result_stack_lr["_label"] = "Stacking FIXED (LR meta, no SMOTE, 52 feat)"
    all_results["Stacking_LRmeta_fixed_52"] = result_stack_lr

    # ================================================================
    # PART B: New Imbalanced Learning Models (52 features)
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART B: New Imbalanced Learning Models (52 features)")
    log.info("=" * 70)

    # B1: EasyEnsembleClassifier
    log.info("\n  [B1] EasyEnsembleClassifier...")
    easy_ens = EasyEnsembleClassifier(
        n_estimators=20, random_state=42, n_jobs=-1,
    )
    result_easy = run_cv(easy_ens, X_52, y, outer_cv, scoring, "EasyEnsemble_52")
    result_easy["use_smote"] = False
    result_easy["_label"] = "EasyEnsembleClassifier (52 feat)"
    all_results["EasyEnsemble_52"] = result_easy

    # B2: BalancedRandomForestClassifier
    log.info("\n  [B2] BalancedRandomForestClassifier...")
    brf = BalancedRandomForestClassifier(
        n_estimators=500, max_depth=15,
        random_state=42, n_jobs=-1,
    )
    result_brf = run_cv(brf, X_52, y, outer_cv, scoring, "BalancedRF_52")
    result_brf["use_smote"] = False
    result_brf["_label"] = "BalancedRandomForest (52 feat)"
    all_results["BalancedRF_52"] = result_brf

    # ================================================================
    # PART C: Calibration (52 features)
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART C: Calibration - CalibratedClassifierCV (52 features)")
    log.info("=" * 70)

    xgb_params = dict(optuna_results["models"]["XGBoost"]["params"])
    xgb_params.pop("use_smote", None)
    xgb_base = XGBClassifier(**xgb_params, random_state=42, eval_metric="logloss", verbosity=0)
    cal_xgb = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)

    result_cal = run_cv(cal_xgb, X_52, y, outer_cv, scoring, "XGBoost_calibrated_52")
    result_cal["use_smote"] = False
    result_cal["_label"] = "XGBoost + Isotonic Calibration (52 feat)"
    all_results["XGBoost_calibrated_52"] = result_cal

    # ================================================================
    # PART D: OOF Stacking (52 features)
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART D: Out-of-Fold Stacking (52 features)")
    log.info("  Explicit OOF predictions from base models -> XGBoost meta")
    log.info("=" * 70)

    t0 = time.time()
    oof_preds = np.zeros((len(y), len(top_models)))
    oof_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for i, model_name in enumerate(top_models):
        log.info(f"  Generating OOF predictions for {model_name}...")
        params = dict(optuna_results["models"][model_name]["params"])
        model = build_model_from_params(model_name, params)
        try:
            oof_proba = cross_val_predict(
                model, X_52, y, cv=oof_cv, method="predict_proba",
            )[:, 1]
            oof_preds[:, i] = oof_proba
        except Exception as e:
            log.warning(f"  OOF failed for {model_name}: {e}, using predict")
            oof_pred = cross_val_predict(model, X_52, y, cv=oof_cv)
            oof_preds[:, i] = oof_pred.astype(float)

    # Train XGBoost meta-learner on OOF predictions
    log.info("  Training XGBoost meta-learner on OOF predictions...")
    meta_xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    result_oof = run_cv(meta_xgb, oof_preds, y, outer_cv, scoring, "OOF_Stacking_XGBmeta_52")
    result_oof["use_smote"] = False
    result_oof["time_seconds"] = time.time() - t0
    result_oof["_label"] = "OOF Stacking (XGB meta, 52 feat)"
    all_results["OOF_Stacking_52"] = result_oof

    # ================================================================
    # PART E: New Features (60 features)
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART E: Evaluation with 60 Features")
    log.info("=" * 70)

    has_60 = False
    try:
        df_60 = pd.read_csv(data_60_path)
        feature_cols_60 = [c for c in df_60.columns if c not in ["Provider", "PotentialFraud"]]
        X_60 = df_60[feature_cols_60].values
        y_60 = df_60["PotentialFraud"].values
        assert len(y_60) == len(y), "60-feature dataset row count mismatch"
        has_60 = True
        log.info(f"  60-feature dataset: {X_60.shape[0]} providers, {X_60.shape[1]} features")
    except FileNotFoundError:
        log.warning(f"  {data_60_path} not found. Run 01_preprocess.py --out-filename provider_features_v2.csv first.")
        log.warning("  Skipping Part E.")
    except Exception as e:
        log.warning(f"  Error loading 60-feature data: {e}")

    if has_60:
        # E1: XGBoost tuned on 60 features
        log.info("\n  [E1] XGBoost tuned (60 features)...")
        xgb_60 = build_model_from_params("XGBoost", dict(optuna_results["models"]["XGBoost"]["params"]))
        result_xgb60 = run_cv(xgb_60, X_60, y_60, outer_cv, scoring, "XGBoost_60feat")
        result_xgb60["use_smote"] = False
        result_xgb60["_label"] = "XGBoost tuned (60 feat)"
        all_results["XGBoost_60feat"] = result_xgb60

        # E2: Fixed SoftVoting on 60 features
        log.info("\n  [E2] SoftVoting FIXED (60 features)...")
        ensemble_list_60 = []
        for name in top_models:
            params = dict(optuna_results["models"][name]["params"])
            est = build_model_from_params(name, params)
            ensemble_list_60.append((name.lower()[:4], est))

        sv_60 = VotingClassifier(estimators=ensemble_list_60, voting="soft", n_jobs=1)
        result_sv60 = run_cv(sv_60, X_60, y_60, outer_cv, scoring, "SoftVoting_fixed_60feat")
        result_sv60["use_smote"] = False
        result_sv60["_label"] = "SoftVoting FIXED (no SMOTE, 60 feat)"
        all_results["SoftVoting_fixed_60feat"] = result_sv60

        # E3: Fixed Stacking + XGB meta on 60 features
        log.info("\n  [E3] Stacking FIXED + XGB meta (60 features)...")
        ensemble_list_60b = []
        for name in top_models:
            params = dict(optuna_results["models"][name]["params"])
            est = build_model_from_params(name, params)
            ensemble_list_60b.append((name.lower()[:4], est))

        xgb_meta_60 = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        stack_60 = StackingClassifier(
            estimators=ensemble_list_60b,
            final_estimator=xgb_meta_60,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=False, n_jobs=1,
        )
        result_stack60 = run_cv(stack_60, X_60, y_60, outer_cv, scoring, "Stacking_XGBmeta_fixed_60feat")
        result_stack60["use_smote"] = False
        result_stack60["_label"] = "Stacking FIXED (XGB meta, no SMOTE, 60 feat)"
        all_results["Stacking_XGBmeta_fixed_60feat"] = result_stack60

        # E4: EasyEnsemble on 60 features
        log.info("\n  [E4] EasyEnsembleClassifier (60 features)...")
        easy_60 = EasyEnsembleClassifier(n_estimators=20, random_state=42, n_jobs=-1)
        result_easy60 = run_cv(easy_60, X_60, y_60, outer_cv, scoring, "EasyEnsemble_60feat")
        result_easy60["use_smote"] = False
        result_easy60["_label"] = "EasyEnsembleClassifier (60 feat)"
        all_results["EasyEnsemble_60feat"] = result_easy60

        # E5: BalancedRandomForest on 60 features
        log.info("\n  [E5] BalancedRandomForestClassifier (60 features)...")
        brf_60 = BalancedRandomForestClassifier(
            n_estimators=500, max_depth=15, random_state=42, n_jobs=-1,
        )
        result_brf60 = run_cv(brf_60, X_60, y_60, outer_cv, scoring, "BalancedRF_60feat")
        result_brf60["use_smote"] = False
        result_brf60["_label"] = "BalancedRandomForest (60 feat)"
        all_results["BalancedRF_60feat"] = result_brf60

    # ================================================================
    # PART F: Threshold Optimization on Best Configuration
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART F: Threshold Optimization")
    log.info("=" * 70)

    # Find the best configuration by F1 (excluding buggy and baseline refs)
    eval_configs = {
        k: v for k, v in all_results.items()
        if not k.endswith("_buggy") and not k.endswith("_baseline") and "f1" in v
    }
    best_config_name = max(eval_configs, key=lambda k: eval_configs[k]["f1"]["mean"])
    best_f1 = eval_configs[best_config_name]["f1"]["mean"]
    log.info(f"  Best configuration: {best_config_name} (F1={best_f1:.4f})")

    # Determine which dataset and model to use for threshold optimization
    if "60feat" in best_config_name and has_60:
        X_best = X_60
        y_best = y_60
    else:
        X_best = X_52
        y_best = y

    # Rebuild the best model for cross_val_predict
    if "SoftVoting_fixed" in best_config_name:
        if "60" in best_config_name:
            el = []
            for name in top_models:
                params = dict(optuna_results["models"][name]["params"])
                el.append((name.lower()[:4], build_model_from_params(name, params)))
            best_model_obj = VotingClassifier(estimators=el, voting="soft", n_jobs=1)
        else:
            el = []
            for name in top_models:
                params = dict(optuna_results["models"][name]["params"])
                el.append((name.lower()[:4], build_model_from_params(name, params)))
            best_model_obj = VotingClassifier(estimators=el, voting="soft", n_jobs=1)
    elif "Stacking_XGBmeta" in best_config_name:
        el = []
        for name in top_models:
            params = dict(optuna_results["models"][name]["params"])
            el.append((name.lower()[:4], build_model_from_params(name, params)))
        meta = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        best_model_obj = StackingClassifier(
            estimators=el, final_estimator=meta,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=False, n_jobs=1,
        )
    elif "XGBoost" in best_config_name and "calibrated" not in best_config_name:
        best_model_obj = build_model_from_params(
            "XGBoost", dict(optuna_results["models"]["XGBoost"]["params"])
        )
    elif "EasyEnsemble" in best_config_name:
        best_model_obj = EasyEnsembleClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    elif "BalancedRF" in best_config_name:
        best_model_obj = BalancedRandomForestClassifier(
            n_estimators=500, max_depth=15, random_state=42, n_jobs=-1,
        )
    elif "calibrated" in best_config_name:
        xgb_p = dict(optuna_results["models"]["XGBoost"]["params"])
        xgb_p.pop("use_smote", None)
        best_model_obj = CalibratedClassifierCV(
            XGBClassifier(**xgb_p, random_state=42, eval_metric="logloss", verbosity=0),
            method="isotonic", cv=5,
        )
    elif "OOF_Stacking" in best_config_name:
        # For OOF stacking, threshold optimization is on the meta-learner
        log.info("  OOF Stacking: threshold optimization on meta predictions")
        best_model_obj = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        X_best = oof_preds
        y_best = y
    else:
        # Fallback to XGBoost
        best_model_obj = build_model_from_params(
            "XGBoost", dict(optuna_results["models"]["XGBoost"]["params"])
        )

    log.info(f"  Running cross_val_predict for threshold optimization...")
    try:
        y_pred_proba = cross_val_predict(
            best_model_obj, X_best, y_best, cv=outer_cv, method="predict_proba",
        )[:, 1]

        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        opt_threshold, opt_f1 = find_optimal_threshold(y_best, y_pred_proba)
        y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)

        log.info(f"  Default threshold (0.5):  F1={f1_score(y_best, y_pred_default):.4f}")
        log.info(f"  Optimal threshold ({opt_threshold:.3f}): F1={f1_score(y_best, y_pred_opt):.4f}")

        all_results["_threshold_optimization"] = {
            "best_config": best_config_name,
            "default_threshold": 0.5,
            "optimal_threshold": float(opt_threshold),
            "default_f1": float(f1_score(y_best, y_pred_default)),
            "optimal_f1": float(f1_score(y_best, y_pred_opt)),
            "default_precision": float(precision_score(y_best, y_pred_default)),
            "optimal_precision": float(precision_score(y_best, y_pred_opt)),
            "default_recall": float(recall_score(y_best, y_pred_default)),
            "optimal_recall": float(recall_score(y_best, y_pred_opt)),
        }
    except Exception as e:
        log.error(f"  Threshold optimization failed: {e}")

    # ================================================================
    # PART G: Statistical Tests + Bootstrap CIs
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("PART G: Statistical Tests + Bootstrap CIs")
    log.info("=" * 70)

    # Collect all model F1 arrays for Friedman test
    model_names = [
        k for k in all_results
        if not k.startswith("_") and not k.endswith("_buggy") and not k.endswith("_baseline")
        and "f1" in all_results[k]
    ]
    f1_arrays = [all_results[k]["f1"]["values"] for k in model_names]

    log.info(f"  Models in statistical comparison: {len(model_names)}")
    for mn in model_names:
        log.info(f"    {mn}: F1={all_results[mn]['f1']['mean']:.4f}")

    # Friedman test
    if len(f1_arrays) >= 3:
        stat, p_val = stats.friedmanchisquare(*f1_arrays)
        log.info(f"\n  Friedman test: chi2={stat:.4f}, p={p_val:.6f}")
        all_results["_friedman"] = {"statistic": float(stat), "p_value": float(p_val)}

    # Pairwise Wilcoxon against best
    best_model_key = max(model_names, key=lambda k: all_results[k]["f1"]["mean"])
    log.info(f"  Best model: {best_model_key} (F1={all_results[best_model_key]['f1']['mean']:.4f})")

    raw_pvalues = {}
    for name in model_names:
        if name == best_model_key:
            continue
        best_scores = np.array(all_results[best_model_key]["f1"]["values"])
        other_scores = np.array(all_results[name]["f1"]["values"])
        diff = best_scores - other_scores
        if np.all(diff == 0):
            raw_pvalues[name] = 1.0
            continue
        try:
            _, p = stats.wilcoxon(best_scores, other_scores)
            raw_pvalues[name] = float(p)
        except Exception:
            raw_pvalues[name] = 1.0

    corrected_pvalues = holm_bonferroni(raw_pvalues)
    log.info("\n  Pairwise Wilcoxon tests (Holm-Bonferroni corrected):")
    for name in sorted(corrected_pvalues.keys()):
        adj_p = corrected_pvalues[name]
        sig = "***" if adj_p < 0.01 else "**" if adj_p < 0.05 else "*" if adj_p < 0.1 else "ns"
        log.info(f"    vs {name:35s}: adjusted_p={adj_p:.4f} [{sig}]")

    all_results["_wilcoxon"] = {
        "best_model": best_model_key,
        "raw_pvalues": raw_pvalues,
        "corrected_pvalues": {k: float(v) for k, v in corrected_pvalues.items()},
        "correction": "holm-bonferroni",
    }

    # Bootstrap CIs on best model
    log.info("\n  Bootstrap 95% CIs on best model...")
    n_bootstrap = 2000
    rng = np.random.RandomState(42)

    # Get predictions for bootstrap
    try:
        if "y_pred_proba" in dir() and y_pred_proba is not None:
            boot_proba = y_pred_proba
            boot_pred = y_pred_opt
            boot_y = y_best
        else:
            raise ValueError("No predictions available")
    except Exception:
        # Fallback: use the best individual model's CV predictions
        log.info("  Generating predictions for bootstrap...")
        fallback_model = build_model_from_params(
            "XGBoost", dict(optuna_results["models"]["XGBoost"]["params"])
        )
        boot_proba = cross_val_predict(
            fallback_model, X_52, y, cv=outer_cv, method="predict_proba",
        )[:, 1]
        boot_pred = (boot_proba >= 0.5).astype(int)
        boot_y = y

    boot = {m: [] for m in ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc"]}
    for _ in range(n_bootstrap):
        idx = rng.choice(len(boot_y), size=len(boot_y), replace=True)
        yb, ypb, yppb = boot_y[idx], boot_pred[idx], boot_proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot["f1"].append(f1_score(yb, ypb))
        boot["precision"].append(precision_score(yb, ypb, zero_division=0))
        boot["recall"].append(recall_score(yb, ypb))
        boot["mcc"].append(matthews_corrcoef(yb, ypb))
        boot["roc_auc"].append(roc_auc_score(yb, yppb))
        boot["pr_auc"].append(average_precision_score(yb, yppb))

    bootstrap_cis = {}
    for metric_name, values in boot.items():
        if values:
            ci_lo, ci_hi = np.percentile(values, [2.5, 97.5])
            bootstrap_cis[metric_name] = {
                "mean": float(np.mean(values)),
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
            }
            log.info(f"    {metric_name}: {np.mean(values):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    all_results["_bootstrap_ci"] = {
        "model": best_model_key,
        "n_iterations": n_bootstrap,
        "metrics": bootstrap_cis,
    }

    # ================================================================
    # Save Results
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("Saving results...")
    log.info("=" * 70)

    out_path = RESULTS_DIR / "improved_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"  Saved: {out_path}")

    # Save best model
    log.info("  Training best model on full dataset for saving...")
    try:
        best_model_final = build_model_from_params(
            "XGBoost", dict(optuna_results["models"]["XGBoost"]["params"])
        )
        best_model_final.fit(X_52, y)
        model_path = RESULTS_DIR / "improved_best_model.pkl"
        joblib.dump(best_model_final, model_path)
        log.info(f"  Saved: {model_path}")
    except Exception as e:
        log.error(f"  Failed to save best model: {e}")

    # ================================================================
    # Summary
    # ================================================================
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    log.info(f"\n  {'Configuration':<40s} {'F1':>8s} {'AUC-PR':>8s} {'MCC':>8s}")
    log.info(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")

    for name in sorted(all_results.keys()):
        if name.startswith("_"):
            continue
        r = all_results[name]
        if "f1" not in r:
            continue
        label = r.get("_label", name)
        f1_val = r["f1"]["mean"]
        auc_val = r.get("average_precision", {}).get("mean", 0)
        mcc_val = r.get("mcc", {}).get("mean", 0)
        log.info(f"  {label:<40s} {f1_val:>8.4f} {auc_val:>8.4f} {mcc_val:>8.4f}")

    # Highlight the bug fix impact
    if "SoftVoting_buggy" in all_results and "SoftVoting_fixed_52" in all_results:
        buggy_f1 = all_results["SoftVoting_buggy"]["f1"]["mean"]
        fixed_f1 = all_results["SoftVoting_fixed_52"]["f1"]["mean"]
        log.info(f"\n  BUG FIX IMPACT:")
        log.info(f"    SoftVoting with SMOTE bug: F1={buggy_f1:.4f}")
        log.info(f"    SoftVoting FIXED:          F1={fixed_f1:.4f}")
        log.info(f"    Delta:                     {fixed_f1 - buggy_f1:+.4f}")

    if "Stacking_buggy" in all_results and "Stacking_XGBmeta_fixed_52" in all_results:
        buggy_f1 = all_results["Stacking_buggy"]["f1"]["mean"]
        fixed_f1 = all_results["Stacking_XGBmeta_fixed_52"]["f1"]["mean"]
        log.info(f"    Stacking with SMOTE bug:   F1={buggy_f1:.4f}")
        log.info(f"    Stacking FIXED (XGB meta): F1={fixed_f1:.4f}")
        log.info(f"    Delta:                     {fixed_f1 - buggy_f1:+.4f}")

    log.info(f"\n  Best overall: {best_model_key} (F1={all_results[best_model_key]['f1']['mean']:.4f})")
    log.info("\nDone.")


if __name__ == "__main__":
    main()
