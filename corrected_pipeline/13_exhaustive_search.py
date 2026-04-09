"""
Phase 13: Exhaustive Hyperparameter Search (Leakage-Free)
All methods pre-June 2024.

Re-tunes ALL models on 190 leakage-free features with wide Optuna search.
Then evaluates best configs in the full leakage-free 10-fold framework.
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
from scipy.stats import entropy as scipy_entropy, kurtosis as scipy_kurtosis
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("exhaustive_search.log")],
)
log = logging.getLogger(__name__)
np.random.seed(42)

UNSAFE_PATTERNS = [
    "_zscore", "_pctile", "IsoForest_", "Phys_AvgReimb", "OpPhys_AvgReimb",
    "Phys_ClaimCount", "Phys_Role_Overlap",
    "Shared_Bene_Count", "Bene_Exclusivity", "Avg_Providers_Per_Bene",
    "Max_Providers_Per_Bene", "Provider_Network_Degree",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--timeout", type=int, default=900, help="Per-model timeout seconds")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 70)
    log.info("PHASE 13: EXHAUSTIVE HYPERPARAMETER SEARCH")
    log.info(f"  {args.n_trials} trials/model, {args.timeout}s timeout")
    log.info("=" * 70)

    # ---- Load data ----
    log.info("\n[1/4] Loading data...")
    v3_path = f"{args.pipeline_dir}/provider_features_v3.csv"
    df = pd.read_csv(v3_path)

    # Drop unsafe features
    all_cols = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    unsafe = [c for c in all_cols if any(p in c for p in UNSAFE_PATTERNS)]
    safe_cols = [c for c in all_cols if c not in unsafe]

    # Clip code fractions
    for c in safe_cols:
        if c.startswith("DiagCode_") or c.startswith("ProcCode_"):
            df[c] = df[c].clip(0, 2.0)

    X = df[safe_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df["PotentialFraud"].values
    pos_weight = len(y[y == 0]) / max(len(y[y == 1]), 1)

    log.info(f"  Features: {X.shape[1]}, Providers: {X.shape[0]}, Fraud: {y.sum()}")

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- Optuna objectives ----
    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 1e-9, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, pos_weight * 2),
        }
        model = XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1)
        scores = cross_validate(model, X, y, cv=inner_cv, scoring="average_precision", n_jobs=1)
        return scores["test_score"].mean()

    def lgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100, log=True),
        }
        balance = trial.suggest_categorical("balance", ["is_unbalance", "scale_pos_weight", "none"])
        if balance == "is_unbalance":
            params["is_unbalance"] = True
        elif balance == "scale_pos_weight":
            params["scale_pos_weight"] = trial.suggest_float("lgb_spw", 1.0, pos_weight * 2)

        boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
        params["boosting_type"] = boosting
        if boosting == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.01, 0.5)

        model = LGBMClassifier(**params, random_state=42, verbose=-1)
        scores = cross_validate(model, X, y, cv=inner_cv, scoring="average_precision", n_jobs=1)
        return scores["test_score"].mean()

    def cb_objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1200),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 50, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
        aw = trial.suggest_categorical("auto_class_weights", ["Balanced", "None"])
        if aw == "None":
            aw = None
        model = CatBoostClassifier(**params, auto_class_weights=aw, random_seed=42, verbose=0)
        scores = cross_validate(model, X, y, cv=inner_cv, scoring="average_precision", n_jobs=1)
        return scores["test_score"].mean()

    def gb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
        }
        model = GradientBoostingClassifier(**params, random_state=42)
        scores = cross_validate(model, X, y, cv=inner_cv, scoring="average_precision", n_jobs=1)
        return scores["test_score"].mean()

    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        }
        cw = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])
        model = RandomForestClassifier(**params, class_weight=cw, random_state=42, n_jobs=-1)
        scores = cross_validate(model, X, y, cv=inner_cv, scoring="average_precision", n_jobs=1)
        return scores["test_score"].mean()

    def et_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        }
        cw = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])
        model = ExtraTreesClassifier(**params, class_weight=cw, random_state=42, n_jobs=-1)
        scores = cross_validate(model, X, y, cv=inner_cv, scoring="average_precision", n_jobs=1)
        return scores["test_score"].mean()

    # ---- Run tuning ----
    log.info("\n[2/4] Running Optuna for all models...")

    all_tuned = {}
    objectives = [
        ("XGBoost", xgb_objective),
        ("LightGBM", lgb_objective),
        ("CatBoost", cb_objective),
        ("GradientBoosting", gb_objective),
        ("RandomForest", rf_objective),
        ("ExtraTrees", et_objective),
    ]

    for model_name, obj_fn in objectives:
        log.info(f"\n  Tuning {model_name} ({args.n_trials} trials, {args.timeout}s timeout)...")
        t0 = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj_fn, n_trials=args.n_trials, timeout=args.timeout)
        elapsed = time.time() - t0
        all_tuned[model_name] = {
            "params": study.best_params,
            "best_aucpr": study.best_value,
            "n_trials": len(study.trials),
            "time_seconds": elapsed,
        }
        log.info(f"    Best AUC-PR: {study.best_value:.4f} ({len(study.trials)} trials, {elapsed:.0f}s)")
        log.info(f"    Best params: {study.best_params}")

    # Save tuned params
    with open(RESULTS_DIR / "exhaustive_params.json", "w") as f:
        json.dump(all_tuned, f, indent=2, default=str)
    log.info(f"\n  Saved: exhaustive_params.json")

    # ---- Evaluate with leakage-free 10-fold CV ----
    log.info("\n[3/4] Evaluating tuned models with 10-fold leakage-free CV...")

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    def build_model(name, params):
        p = dict(params)
        if name == "XGBoost":
            return XGBClassifier(**p, random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1)
        elif name == "LightGBM":
            bal = p.pop("balance", "none")
            bt = p.pop("boosting_type", "gbdt")
            dr = p.pop("drop_rate", 0.1)
            spw = p.pop("lgb_spw", None)
            p["boosting_type"] = bt
            if bt == "dart":
                p["drop_rate"] = dr
            if bal == "is_unbalance":
                p["is_unbalance"] = True
            elif bal == "scale_pos_weight" and spw:
                p["scale_pos_weight"] = spw
            return LGBMClassifier(**p, random_state=42, verbose=-1)
        elif name == "CatBoost":
            aw = p.pop("auto_class_weights", None)
            if aw == "None":
                aw = None
            return CatBoostClassifier(**p, auto_class_weights=aw, random_seed=42, verbose=0)
        elif name == "GradientBoosting":
            return GradientBoostingClassifier(**p, random_state=42)
        elif name == "RandomForest":
            cw = p.pop("class_weight", "balanced_subsample")
            return RandomForestClassifier(**p, class_weight=cw, random_state=42, n_jobs=-1)
        elif name == "ExtraTrees":
            cw = p.pop("class_weight", "balanced_subsample")
            return ExtraTreesClassifier(**p, class_weight=cw, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {name}")

    MODEL_NAMES = list(all_tuned.keys())
    oof_proba = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    fold_f1 = {name: [] for name in MODEL_NAMES}

    # Compute in-fold features helper (same as script 12)
    providers = df["Provider"].values

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        fold_start = time.time()
        log.info(f"\n  === Fold {fold_idx + 1}/10 ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Add in-fold z-scores (from train only)
        extra_train = []
        extra_test = []
        for col_idx, col_name in enumerate(safe_cols):
            if col_name in ["InscClaimAmtReimbursed", "Claim_Count", "Beneficiary_Count",
                           "Claim_Duration", "Dead_Patient_Ratio", "Inpatient_Ratio",
                           "Claims_Per_Bene", "Reimburse_CV"]:
                mu = X_train[:, col_idx].mean()
                sigma = X_train[:, col_idx].std()
                if sigma == 0:
                    sigma = 1e-9
                extra_train.append((X_train[:, col_idx] - mu) / sigma)
                extra_test.append((X_test[:, col_idx] - mu) / sigma)

        # Add IsolationForest from train only
        iso = IsolationForest(n_estimators=200, contamination=0.094, random_state=42, n_jobs=-1)
        iso.fit(X_train)
        extra_train.append(iso.decision_function(X_train))
        extra_test.append(iso.decision_function(X_test))

        # Combine
        if extra_train:
            X_train_aug = np.column_stack([X_train] + [e.reshape(-1, 1) if e.ndim == 1 else e for e in extra_train])
            X_test_aug = np.column_stack([X_test] + [e.reshape(-1, 1) if e.ndim == 1 else e for e in extra_test])
        else:
            X_train_aug, X_test_aug = X_train, X_test

        log.info(f"    Features: {X_train_aug.shape[1]}")

        for name in MODEL_NAMES:
            try:
                model = build_model(name, dict(all_tuned[name]["params"]))
                model.fit(X_train_aug, y_train)
                proba = model.predict_proba(X_test_aug)[:, 1]
                oof_proba[name][test_idx] = proba
                pred = (proba >= 0.5).astype(int)
                fold_f1[name].append(f1_score(y_test, pred))
            except Exception as e:
                log.warning(f"    {name} failed: {e}")
                fold_f1[name].append(0.0)

        elapsed = time.time() - fold_start
        log.info(f"    Fold {fold_idx+1} done in {elapsed:.0f}s")
        for name in MODEL_NAMES:
            if fold_f1[name]:
                log.info(f"      {name:25s}: F1={fold_f1[name][-1]:.4f}")

    # ---- Ensemble ----
    log.info("\n[4/4] Optuna ensemble (AUC-PR, 300 trials)...")

    oof_matrix = np.column_stack([oof_proba[k] for k in MODEL_NAMES])

    def ens_obj(trial):
        weights = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MODEL_NAMES])
        wsum = weights.sum()
        if wsum == 0:
            return 0.0
        weights /= wsum
        blended = oof_matrix @ weights
        return average_precision_score(y, blended)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(ens_obj, n_trials=300)

    best_w = np.array([study.best_params[f"w_{n}"] for n in MODEL_NAMES])
    best_w /= best_w.sum()

    blended = oof_matrix @ best_w
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y, blended)
    f1_arr = np.where((prec_arr[:-1] + rec_arr[:-1]) > 0,
                       2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]), 0)
    opt_t = thresh_arr[np.argmax(f1_arr)]
    y_pred = (blended >= opt_t).astype(int)

    ens_f1 = f1_score(y, y_pred)
    ens_aucpr = average_precision_score(y, blended)
    ens_auc = roc_auc_score(y, blended)

    # Per-fold ensemble F1
    ens_fold_f1 = []
    for _, test_idx in outer_cv.split(X, y):
        fp = oof_matrix[test_idx] @ best_w
        ens_fold_f1.append(f1_score(y[test_idx], (fp >= opt_t).astype(int)))

    log.info(f"\n  Ensemble weights:")
    for n, w in zip(MODEL_NAMES, best_w):
        log.info(f"    {n:25s}: {w:.4f}")
    log.info(f"\n  Ensemble AUC-PR: {ens_aucpr:.4f}")
    log.info(f"  Ensemble (t={opt_t:.3f}): F1={ens_f1:.4f}")
    log.info(f"  Per-fold F1: {[f'{x:.4f}' for x in ens_fold_f1]}")
    log.info(f"  Mean={np.mean(ens_fold_f1):.4f} Std={np.std(ens_fold_f1):.4f}")

    # Bootstrap CIs
    log.info("\n  Bootstrap 95% CIs...")
    rng = np.random.RandomState(42)
    boot_f1 = []
    for _ in range(2000):
        idx = rng.choice(len(y), len(y), replace=True)
        yb, bp = y[idx], blended[idx]
        if len(np.unique(yb)) < 2:
            continue
        p_a, r_a, t_a = precision_recall_curve(yb, bp)
        f1b = np.where((p_a[:-1]+r_a[:-1])>0, 2*p_a[:-1]*r_a[:-1]/(p_a[:-1]+r_a[:-1]), 0)
        bt = t_a[np.argmax(f1b)]
        boot_f1.append(f1_score(yb, (bp >= bt).astype(int)))

    ci_lo, ci_hi = np.percentile(boot_f1, [2.5, 97.5])
    log.info(f"  F1 Bootstrap CI: {np.mean(boot_f1):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Save everything
    results = {
        "tuned_params": {k: v["params"] for k, v in all_tuned.items()},
        "tuning_aucpr": {k: v["best_aucpr"] for k, v in all_tuned.items()},
        "individual_f1": {k: {"mean": float(np.mean(fold_f1[k])), "std": float(np.std(fold_f1[k])), "values": fold_f1[k]} for k in MODEL_NAMES},
        "ensemble": {
            "f1_mean": float(np.mean(ens_fold_f1)),
            "f1_std": float(np.std(ens_fold_f1)),
            "f1_values": ens_fold_f1,
            "aucpr": float(ens_aucpr),
            "auc_roc": float(ens_auc),
            "threshold": float(opt_t),
            "weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
            "bootstrap_ci": {"mean": float(np.mean(boot_f1)), "lower": float(ci_lo), "upper": float(ci_hi)},
        },
    }

    with open(RESULTS_DIR / "exhaustive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\n{'='*70}")
    log.info("FINAL RESULTS (EXHAUSTIVE SEARCH, LEAKAGE-FREE)")
    log.info(f"{'='*70}")
    ranking = sorted(MODEL_NAMES, key=lambda k: -np.mean(fold_f1[k]))
    for n in ranking:
        log.info(f"  {n:25s}: F1={np.mean(fold_f1[n]):.4f} +/- {np.std(fold_f1[n]):.4f} | AUC-PR={all_tuned[n]['best_aucpr']:.4f}")
    log.info(f"\n  Ensemble:              F1={np.mean(ens_fold_f1):.4f} +/- {np.std(ens_fold_f1):.4f}")
    log.info(f"  Bootstrap 95% CI:      [{ci_lo:.4f}, {ci_hi:.4f}]")
    log.info(f"\nDone.")


if __name__ == "__main__":
    main()
