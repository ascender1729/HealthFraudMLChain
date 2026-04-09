"""
Phase 10: Advanced Evaluation - Push F1 to 0.85+
All methods pre-June 2024.

Pipeline:
  1. Load v3 features (~180 features)
  2. Feature selection (correlation filter + mutual information)
  3. Train 8 diverse models with 10-fold CV
  4. Optuna-optimized weighted ensemble
  5. Stacked ensemble (XGBoost meta on OOF probs)
  6. Threshold optimization
  7. Statistical tests + bootstrap CIs
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("advanced_evaluation.log"),
    ],
)
log = logging.getLogger(__name__)

np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 10: Advanced Evaluation")
    p.add_argument("--data-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    return p.parse_args()


def holm_bonferroni(p_values):
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    corrected = {}
    for rank, (name, p) in enumerate(items):
        corrected[name] = min(p * (m - rank), 1.0)
    return corrected


def find_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def correlation_filter(X_df, threshold=0.95):
    """Remove features with correlation > threshold, keeping the first."""
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    log.info(f"  Correlation filter: removing {len(to_drop)} features (>{threshold} corr)")
    return [c for c in X_df.columns if c not in to_drop]


def run_cv(model, X, y, outer_cv, scoring, name=""):
    """Run 10-fold CV and return results dict."""
    t0 = time.time()
    try:
        cv_results = cross_validate(
            model, X, y, cv=outer_cv, scoring=scoring,
            return_train_score=False, n_jobs=1,
        )
    except Exception as e:
        log.error(f"  {name} failed: {e}")
        return None
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
        f"  {name:40s} | F1={result['f1']['mean']:.4f}+/-{result['f1']['std']:.4f} | "
        f"AUC-PR={result['average_precision']['mean']:.4f} | "
        f"MCC={result['mcc']['mean']:.4f} | {elapsed:.1f}s"
    )
    return result


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 70)
    log.info("PHASE 10: ADVANCED EVALUATION - TARGET F1 0.85+")
    log.info("=" * 70)

    # ---- Load data ----
    log.info("[1/8] Loading v3 features...")
    v3_path = f"{args.data_dir}/provider_features_v3.csv"
    params_path = RESULTS_DIR / "best_params.json"

    try:
        df = pd.read_csv(v3_path)
    except FileNotFoundError:
        log.error(f"File not found: {v3_path}. Run 09_advanced_preprocess.py first.")
        sys.exit(1)

    # Load Optuna tuned params if available
    optuna_params = {}
    try:
        with open(params_path) as f:
            optuna_params = json.load(f)
    except FileNotFoundError:
        log.warning("No Optuna params found, using defaults")

    feature_cols_all = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    log.info(f"  Total features loaded: {len(feature_cols_all)}")
    log.info(f"  Providers: {len(df)}, Fraud rate: {df['PotentialFraud'].mean()*100:.1f}%")

    # ---- Feature Selection ----
    log.info("\n[2/8] Feature selection...")

    X_all = df[feature_cols_all].copy()
    y = df["PotentialFraud"].values

    # Step 1: Correlation filter
    selected_cols = correlation_filter(X_all, threshold=0.95)
    log.info(f"  After correlation filter: {len(selected_cols)} features")

    # Step 2: Mutual information
    X_sel = X_all[selected_cols].fillna(0).replace([np.inf, -np.inf], 0)
    mi_scores = mutual_info_classif(X_sel.values, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": selected_cols, "mi_score": mi_scores}).sort_values("mi_score", ascending=False)

    # Keep features with MI > 0 (informative)
    informative = mi_df[mi_df["mi_score"] > 0.001]["feature"].tolist()
    log.info(f"  Features with MI > 0.001: {len(informative)}")

    # Save MI scores
    mi_df.to_csv(RESULTS_DIR / "mutual_info_scores_v3.csv", index=False)
    log.info(f"  Saved MI scores to mutual_info_scores_v3.csv")

    # Use all informative features
    feature_cols = informative
    X = X_sel[feature_cols].values
    log.info(f"  Final feature count: {X.shape[1]}")

    # ---- Setup ----
    N_FOLDS = 10
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    pos_weight = len(y[y == 0]) / max(len(y[y == 1]), 1)

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "mcc": make_scorer(matthews_corrcoef),
    }

    all_results = {}
    oof_predictions = {}  # Store OOF probabilities for ensemble

    # ---- Build Models ----
    log.info("\n[3/8] Building 8 diverse models...")

    models = {}

    # Model 1: XGBoost (tuned)
    if "models" in optuna_params and "XGBoost" in optuna_params["models"]:
        xgb_p = dict(optuna_params["models"]["XGBoost"]["params"])
        xgb_p.pop("use_smote", None)
        models["XGBoost"] = XGBClassifier(**xgb_p, random_state=42, eval_metric="logloss", verbosity=0)
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.01, subsample=0.85,
            colsample_bytree=0.65, scale_pos_weight=pos_weight,
            random_state=42, eval_metric="logloss", verbosity=0,
        )

    # Model 2: LightGBM (tuned)
    if "models" in optuna_params and "LightGBM" in optuna_params["models"]:
        lgb_p = dict(optuna_params["models"]["LightGBM"]["params"])
        lgb_p.pop("use_smote", None)
        models["LightGBM"] = LGBMClassifier(**lgb_p, random_state=42, verbose=-1)
    else:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.02, is_unbalance=True,
            random_state=42, verbose=-1,
        )

    # Model 3: CatBoost (tuned)
    if "models" in optuna_params and "CatBoost" in optuna_params["models"]:
        cb_p = dict(optuna_params["models"]["CatBoost"]["params"])
        cb_p.pop("use_smote", None)
        if cb_p.get("auto_class_weights") == "None":
            cb_p["auto_class_weights"] = None
        models["CatBoost"] = CatBoostClassifier(**cb_p, random_seed=42, verbose=0)
    else:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.01, random_seed=42, verbose=0,
        )

    # Model 4: GradientBoosting (tuned)
    if "models" in optuna_params and "GradientBoosting" in optuna_params["models"]:
        gb_p = dict(optuna_params["models"]["GradientBoosting"]["params"])
        gb_p.pop("use_smote", None)
        models["GradientBoosting"] = GradientBoostingClassifier(**gb_p, random_state=42)
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.02, random_state=42,
        )

    # Model 5: RandomForest (tuned)
    if "models" in optuna_params and "RandomForest" in optuna_params["models"]:
        rf_p = dict(optuna_params["models"]["RandomForest"]["params"])
        rf_p.pop("use_smote", None)
        models["RandomForest"] = RandomForestClassifier(**rf_p, random_state=42, n_jobs=-1)
    else:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )

    # Model 6: ExtraTreesClassifier (diversity via extreme randomness)
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500, max_depth=20, class_weight="balanced_subsample",
        max_features="sqrt", random_state=42, n_jobs=-1,
    )

    # Model 7: KNN (instance-based, different decision boundary)
    models["KNN"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance", metric="minkowski", p=2, n_jobs=-1)),
    ])

    # Model 8: SVM RBF (kernel method, different boundary shape)
    models["SVM_RBF"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                     probability=True, random_state=42)),
    ])

    # ---- Train all models + collect OOF predictions ----
    log.info("\n[4/8] Training 8 models with 10-fold CV + OOF predictions...")

    for name, model in models.items():
        log.info(f"\n  Training {name}...")

        # CV evaluation
        result = run_cv(model, X, y, outer_cv, scoring, name)
        if result is None:
            continue
        result["_label"] = name
        all_results[name] = result

        # OOF predictions for ensemble
        try:
            oof_proba = cross_val_predict(
                model, X, y, cv=outer_cv, method="predict_proba",
            )[:, 1]
            oof_predictions[name] = oof_proba
        except Exception as e:
            log.warning(f"  OOF predict_proba failed for {name}: {e}")
            try:
                oof_pred = cross_val_predict(model, X, y, cv=outer_cv).astype(float)
                oof_predictions[name] = oof_pred
            except Exception as e2:
                log.error(f"  OOF predict also failed for {name}: {e2}")

    # ---- Weighted Ensemble (Optuna-optimized) ----
    log.info("\n[5/8] Optimizing ensemble weights with Optuna...")

    if len(oof_predictions) >= 3:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            oof_matrix = np.column_stack([oof_predictions[k] for k in oof_predictions])
            model_names_oof = list(oof_predictions.keys())

            def objective(trial):
                weights = []
                for i, name in enumerate(model_names_oof):
                    w = trial.suggest_float(f"w_{name}", 0.0, 1.0)
                    weights.append(w)
                weights = np.array(weights)
                w_sum = weights.sum()
                if w_sum == 0:
                    return 0.0
                weights = weights / w_sum

                blended = oof_matrix @ weights
                # Try multiple thresholds
                best_f1 = 0
                for t in np.arange(0.2, 0.7, 0.02):
                    pred = (blended >= t).astype(int)
                    f1 = f1_score(y, pred)
                    if f1 > best_f1:
                        best_f1 = f1
                return best_f1

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=200, show_progress_bar=False)

            best_weights = []
            for name in model_names_oof:
                best_weights.append(study.best_params[f"w_{name}"])
            best_weights = np.array(best_weights)
            best_weights = best_weights / best_weights.sum()

            log.info(f"  Best Optuna ensemble F1: {study.best_value:.4f}")
            log.info(f"  Weights:")
            for name, w in zip(model_names_oof, best_weights):
                log.info(f"    {name:25s}: {w:.4f}")

            # Evaluate weighted ensemble
            blended_proba = oof_matrix @ best_weights
            opt_t_ens, opt_f1_ens = find_optimal_threshold(y, blended_proba)
            blended_pred = (blended_proba >= opt_t_ens).astype(int)

            ens_f1 = f1_score(y, blended_pred)
            ens_prec = precision_score(y, blended_pred)
            ens_rec = recall_score(y, blended_pred)
            ens_mcc = matthews_corrcoef(y, blended_pred)
            ens_auc = roc_auc_score(y, blended_proba)
            ens_aucpr = average_precision_score(y, blended_proba)

            log.info(f"\n  Weighted Ensemble Results (threshold={opt_t_ens:.3f}):")
            log.info(f"    F1={ens_f1:.4f}  Prec={ens_prec:.4f}  Rec={ens_rec:.4f}  MCC={ens_mcc:.4f}")
            log.info(f"    AUC-ROC={ens_auc:.4f}  AUC-PR={ens_aucpr:.4f}")

            # Also evaluate with cross-validation for fair comparison
            # Build per-fold weighted ensemble
            oof_ens_f1_values = []
            for train_idx, test_idx in outer_cv.split(X, y):
                fold_proba = oof_matrix[test_idx] @ best_weights
                fold_pred = (fold_proba >= opt_t_ens).astype(int)
                fold_f1 = f1_score(y[test_idx], fold_pred)
                oof_ens_f1_values.append(fold_f1)

            all_results["WeightedEnsemble"] = {
                "f1": {"mean": float(np.mean(oof_ens_f1_values)), "std": float(np.std(oof_ens_f1_values)), "values": oof_ens_f1_values},
                "precision": {"mean": float(ens_prec), "std": 0.0, "values": [float(ens_prec)]},
                "recall": {"mean": float(ens_rec), "std": 0.0, "values": [float(ens_rec)]},
                "mcc": {"mean": float(ens_mcc), "std": 0.0, "values": [float(ens_mcc)]},
                "roc_auc": {"mean": float(ens_auc), "std": 0.0, "values": [float(ens_auc)]},
                "average_precision": {"mean": float(ens_aucpr), "std": 0.0, "values": [float(ens_aucpr)]},
                "time_seconds": 0,
                "_label": "Optuna-Weighted Ensemble",
                "_weights": {name: float(w) for name, w in zip(model_names_oof, best_weights)},
                "_threshold": float(opt_t_ens),
            }
            log.info(f"  Per-fold F1: mean={np.mean(oof_ens_f1_values):.4f} std={np.std(oof_ens_f1_values):.4f}")

        except ImportError:
            log.warning("  Optuna not available, skipping weighted ensemble")
        except Exception as e:
            log.error(f"  Weighted ensemble optimization failed: {e}")

    # ---- Stacked Ensemble ----
    log.info("\n[5/8] Stacked ensemble (XGBoost meta on OOF probabilities)...")

    if len(oof_predictions) >= 3:
        oof_stack = np.column_stack([oof_predictions[k] for k in oof_predictions])

        meta_xgb = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=pos_weight,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        result_stack = run_cv(meta_xgb, oof_stack, y, outer_cv, scoring, "Stacked_XGBmeta")
        if result_stack:
            result_stack["_label"] = "Stacked Ensemble (XGB meta)"
            all_results["Stacked_XGBmeta"] = result_stack

    # ---- Threshold Optimization on Best Individual Model ----
    log.info("\n[6/8] Threshold optimization...")

    # Find best model by F1
    model_ranking = sorted(
        [(k, v["f1"]["mean"]) for k, v in all_results.items() if "f1" in v],
        key=lambda x: -x[1],
    )
    log.info(f"  Model ranking by F1:")
    for name, f1_val in model_ranking:
        log.info(f"    {name:40s}: {f1_val:.4f}")

    best_name = model_ranking[0][0]

    # Get OOF probabilities for best model
    if best_name == "WeightedEnsemble" and len(oof_predictions) >= 3:
        best_proba = oof_matrix @ best_weights
    elif best_name == "Stacked_XGBmeta" and len(oof_predictions) >= 3:
        best_proba = cross_val_predict(meta_xgb, oof_stack, y, cv=outer_cv, method="predict_proba")[:, 1]
    elif best_name in oof_predictions:
        best_proba = oof_predictions[best_name]
    else:
        # Fallback
        log.warning(f"  No OOF predictions for {best_name}, using XGBoost")
        best_name = "XGBoost"
        best_proba = oof_predictions.get("XGBoost", np.zeros(len(y)))

    opt_threshold, opt_f1 = find_optimal_threshold(y, best_proba)
    y_pred_opt = (best_proba >= opt_threshold).astype(int)
    y_pred_default = (best_proba >= 0.5).astype(int)

    log.info(f"\n  Best model: {best_name}")
    log.info(f"  Default (0.5):    F1={f1_score(y, y_pred_default):.4f}  P={precision_score(y, y_pred_default):.4f}  R={recall_score(y, y_pred_default):.4f}")
    log.info(f"  Optimal ({opt_threshold:.3f}): F1={f1_score(y, y_pred_opt):.4f}  P={precision_score(y, y_pred_opt):.4f}  R={recall_score(y, y_pred_opt):.4f}")

    all_results["_threshold_optimization"] = {
        "best_model": best_name,
        "optimal_threshold": float(opt_threshold),
        "default_f1": float(f1_score(y, y_pred_default)),
        "optimal_f1": float(f1_score(y, y_pred_opt)),
        "default_precision": float(precision_score(y, y_pred_default)),
        "optimal_precision": float(precision_score(y, y_pred_opt)),
        "default_recall": float(recall_score(y, y_pred_default)),
        "optimal_recall": float(recall_score(y, y_pred_opt)),
    }

    # ---- Statistical Tests ----
    log.info("\n[7/8] Statistical tests...")

    model_names_stat = [k for k in all_results if not k.startswith("_") and "f1" in all_results[k] and len(all_results[k]["f1"].get("values", [])) == N_FOLDS]
    f1_arrays = [all_results[k]["f1"]["values"] for k in model_names_stat]

    if len(f1_arrays) >= 3:
        stat, p_val = stats.friedmanchisquare(*f1_arrays)
        log.info(f"  Friedman test: chi2={stat:.4f}, p={p_val:.6f}")
        all_results["_friedman"] = {"statistic": float(stat), "p_value": float(p_val)}

    best_key = max(model_names_stat, key=lambda k: all_results[k]["f1"]["mean"])
    raw_pvalues = {}
    for name in model_names_stat:
        if name == best_key:
            continue
        best_scores = np.array(all_results[best_key]["f1"]["values"])
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
    log.info(f"\n  Wilcoxon tests vs {best_key}:")
    for name in sorted(corrected_pvalues.keys()):
        adj_p = corrected_pvalues[name]
        sig = "***" if adj_p < 0.01 else "**" if adj_p < 0.05 else "*" if adj_p < 0.1 else "ns"
        log.info(f"    vs {name:35s}: p={adj_p:.4f} [{sig}]")

    all_results["_wilcoxon"] = {
        "best_model": best_key,
        "corrected_pvalues": {k: float(v) for k, v in corrected_pvalues.items()},
    }

    # ---- Bootstrap CIs ----
    log.info("\n[7/8] Bootstrap 95% CIs...")

    n_bootstrap = 2000
    rng = np.random.RandomState(42)
    boot = {m: [] for m in ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc"]}

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        yb, ypb, yppb = y[idx], y_pred_opt[idx], best_proba[idx]
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
        "model": best_name,
        "n_iterations": n_bootstrap,
        "metrics": bootstrap_cis,
    }

    # ---- Save ----
    log.info("\n[8/8] Saving results...")

    out_path = RESULTS_DIR / "advanced_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"  Saved: {out_path}")

    # Save feature importance (MI scores)
    log.info(f"  Saved: mutual_info_scores_v3.csv")

    # Save best model
    log.info("  Training best model on full dataset...")
    try:
        if best_name in models:
            final_model = models[best_name]
        else:
            final_model = models["XGBoost"]
        final_model.fit(X, y)
        model_path = RESULTS_DIR / "advanced_best_model.pkl"
        joblib.dump(final_model, model_path)
        log.info(f"  Saved: {model_path}")
    except Exception as e:
        log.error(f"  Failed to save model: {e}")

    # Save selected feature list
    with open(RESULTS_DIR / "selected_features_v3.json", "w") as f:
        json.dump({"features": feature_cols, "n_features": len(feature_cols)}, f, indent=2)

    # ---- Summary ----
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"\n  {'Model':<40s} {'F1':>8s} {'AUC-PR':>8s} {'MCC':>8s}")
    log.info(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")

    for name in sorted(all_results.keys()):
        if name.startswith("_"):
            continue
        r = all_results[name]
        if "f1" not in r:
            continue
        label = r.get("_label", name)
        log.info(f"  {label:<40s} {r['f1']['mean']:>8.4f} {r.get('average_precision', {}).get('mean', 0):>8.4f} {r.get('mcc', {}).get('mean', 0):>8.4f}")

    log.info(f"\n  Best model: {best_name} (F1={all_results[best_name]['f1']['mean']:.4f})")
    log.info(f"  With threshold opt: F1={all_results['_threshold_optimization']['optimal_f1']:.4f}")
    log.info(f"  Features used: {len(feature_cols)}")
    log.info(f"\n  Target was F1 0.85+")
    best_achieved = max(
        all_results["_threshold_optimization"]["optimal_f1"],
        max(all_results[k]["f1"]["mean"] for k in all_results if not k.startswith("_") and "f1" in all_results[k]),
    )
    if best_achieved >= 0.85:
        log.info(f"  TARGET ACHIEVED: F1={best_achieved:.4f}")
    else:
        log.info(f"  Best achieved: F1={best_achieved:.4f} (gap: {0.85 - best_achieved:.4f})")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
