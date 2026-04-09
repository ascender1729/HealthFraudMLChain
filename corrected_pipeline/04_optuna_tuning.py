"""
Phase 4: Optuna Hyperparameter Optimization
All methods pre-June 2024.

Tunes: XGBoost, LightGBM, GradientBoosting, CatBoost, RandomForest, LogisticRegression
Inner objective: AUC-PR via 5-fold stratified CV
Each model also decides whether SMOTE-ENN helps (boolean hyperparameter)
Outputs: results/best_params.json
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from scipy import stats
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("optuna_tuning.log"),
    ],
)
log = logging.getLogger(__name__)

np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: Optuna Hyperparameter Tuning")
    p.add_argument("--data-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    p.add_argument("--n-trials", type=int, default=60, help="Optuna trials per model")
    p.add_argument("--inner-folds", type=int, default=5, help="Inner CV folds for HPO")
    p.add_argument("--timeout", type=int, default=1800, help="Max seconds per model")
    return p.parse_args()


def build_xgboost_objective(X, y, pos_weight, inner_cv):
    def objective(trial):
        use_smote = trial.suggest_categorical("use_smote", [True, False])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, pos_weight * 2),
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        clf = XGBClassifier(**params)
        if use_smote:
            pipeline = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", clf)])
        else:
            pipeline = clf
        scores = cross_val_score(
            pipeline, X, y, cv=inner_cv,
            scoring="average_precision", n_jobs=-1,
        )
        return scores.mean()
    return objective


def build_lightgbm_objective(X, y, inner_cv):
    def objective(trial):
        use_smote = trial.suggest_categorical("use_smote", [True, False])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
            "random_state": 42,
            "verbose": -1,
        }
        clf = LGBMClassifier(**params)
        if use_smote:
            pipeline = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", clf)])
        else:
            pipeline = clf
        scores = cross_val_score(
            pipeline, X, y, cv=inner_cv,
            scoring="average_precision", n_jobs=-1,
        )
        return scores.mean()
    return objective


def build_gradientboosting_objective(X, y, inner_cv):
    def objective(trial):
        use_smote = trial.suggest_categorical("use_smote", [True, False])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
        }
        clf = GradientBoostingClassifier(**params)
        if use_smote:
            pipeline = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", clf)])
        else:
            pipeline = clf
        scores = cross_val_score(
            pipeline, X, y, cv=inner_cv,
            scoring="average_precision", n_jobs=-1,
        )
        return scores.mean()
    return objective


def build_catboost_objective(X, y, inner_cv):
    def objective(trial):
        use_smote = trial.suggest_categorical("use_smote", [True, False])
        params = {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "SqrtBalanced", "None"]
            ),
            "random_seed": 42,
            "verbose": 0,
        }
        # CatBoost auto_class_weights "None" means no weighting
        if params["auto_class_weights"] == "None":
            params["auto_class_weights"] = None
        clf = CatBoostClassifier(**params)
        if use_smote:
            pipeline = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", clf)])
        else:
            pipeline = clf
        scores = cross_val_score(
            pipeline, X, y, cv=inner_cv,
            scoring="average_precision", n_jobs=-1,
        )
        return scores.mean()
    return objective


def build_randomforest_objective(X, y, inner_cv):
    def objective(trial):
        use_smote = trial.suggest_categorical("use_smote", [True, False])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
            "random_state": 42,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        if use_smote:
            pipeline = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", clf)])
        else:
            pipeline = clf
        scores = cross_val_score(
            pipeline, X, y, cv=inner_cv,
            scoring="average_precision", n_jobs=-1,
        )
        return scores.mean()
    return objective


def build_logreg_objective(X, y, inner_cv):
    def objective(trial):
        use_smote = trial.suggest_categorical("use_smote", [True, False])
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        params = {
            "C": C,
            "penalty": penalty,
            "solver": "saga",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        clf = LogisticRegression(**params)
        if use_smote:
            pipeline = ImbPipeline([
                ("smoteenn", SMOTEENN(random_state=42)),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])
        else:
            pipeline = ImbPipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])
        scores = cross_val_score(
            pipeline, X, y, cv=inner_cv,
            scoring="average_precision", n_jobs=-1,
        )
        return scores.mean()
    return objective


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 60)
    log.info("PHASE 4: OPTUNA HYPERPARAMETER OPTIMIZATION")
    log.info(f"  Trials per model: {args.n_trials}")
    log.info(f"  Inner CV folds: {args.inner_folds}")
    log.info(f"  Timeout per model: {args.timeout}s")
    log.info("=" * 60)

    # Load data
    log.info("[1/3] Loading provider features...")
    data_path = f"{args.data_dir}/provider_features.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        log.error(f"File not found: {data_path}. Run 01_preprocess.py first.")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    X = df[feature_cols].values
    y = df["PotentialFraud"].values
    pos_weight = len(y[y == 0]) / len(y[y == 1])

    log.info(f"  {X.shape[0]} providers, {X.shape[1]} features")
    log.info(f"  Fraud rate: {y.mean()*100:.1f}%, imbalance ratio: {pos_weight:.1f}:1")

    inner_cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=42)

    # Define model objectives
    model_objectives = {
        "XGBoost": build_xgboost_objective(X, y, pos_weight, inner_cv),
        "LightGBM": build_lightgbm_objective(X, y, inner_cv),
        "GradientBoosting": build_gradientboosting_objective(X, y, inner_cv),
        "CatBoost": build_catboost_objective(X, y, inner_cv),
        "RandomForest": build_randomforest_objective(X, y, inner_cv),
        "LogisticRegression": build_logreg_objective(X, y, inner_cv),
    }

    # Run Optuna for each model
    log.info("[2/3] Running Optuna optimization...")
    best_params = {}
    study_results = {}

    for name, objective in model_objectives.items():
        log.info(f"\n{'='*50}")
        log.info(f"  Tuning {name}...")
        t0 = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=name,
        )

        try:
            study.optimize(
                objective,
                n_trials=args.n_trials,
                timeout=args.timeout,
                show_progress_bar=False,
            )
        except Exception as e:
            log.error(f"  Optuna failed for {name}: {e}")
            continue

        elapsed = time.time() - t0
        best = study.best_trial

        best_params[name] = {
            "params": best.params,
            "best_auc_pr": float(best.value),
            "n_trials_completed": len(study.trials),
            "time_seconds": elapsed,
        }

        study_results[name] = {
            "best_value": float(best.value),
            "best_trial_number": best.number,
            "n_trials": len(study.trials),
        }

        log.info(f"  Best AUC-PR: {best.value:.4f} (trial #{best.number})")
        log.info(f"  Completed {len(study.trials)} trials in {elapsed:.1f}s")
        log.info(f"  Best params: {json.dumps(best.params, indent=4, default=str)}")

    # Save results
    log.info("\n[3/3] Saving best parameters...")
    output = {
        "meta": {
            "n_trials_per_model": args.n_trials,
            "inner_cv_folds": args.inner_folds,
            "timeout_per_model": args.timeout,
            "objective": "average_precision (AUC-PR)",
            "sampler": "TPESampler",
            "dataset_size": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "fraud_rate": float(y.mean()),
        },
        "models": best_params,
    }

    out_path = RESULTS_DIR / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"  Saved to {out_path}")

    # Summary table
    log.info("\n" + "=" * 80)
    log.info(f"{'Model':25s} | {'Best AUC-PR':>12s} | {'Trials':>8s} | {'SMOTE':>6s} | {'Time':>8s}")
    log.info("-" * 80)
    for name, result in sorted(best_params.items(), key=lambda x: -x[1]["best_auc_pr"]):
        smote = result["params"].get("use_smote", "N/A")
        log.info(
            f"{name:25s} | {result['best_auc_pr']:12.4f} | "
            f"{result['n_trials_completed']:8d} | {str(smote):>6s} | "
            f"{result['time_seconds']:7.1f}s"
        )
    log.info("=" * 80)
    log.info("Phase 4 complete. Run 05_full_evaluation.py next.")


if __name__ == "__main__":
    main()
