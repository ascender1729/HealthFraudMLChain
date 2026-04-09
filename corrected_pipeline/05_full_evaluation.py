"""
Phase 5: Comprehensive Evaluation with Tuned Hyperparameters
All methods pre-June 2024.

Includes:
- 10-fold stratified CV with tuned models
- Threshold optimization via PR curve
- SMOTE vs no-SMOTE ablation
- Feature selection via permutation importance
- Learning curves
- Calibration plots
- Bootstrap CIs + statistical tests
- SHAP + LIME with tuned best model
- All publication figures
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
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
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("full_evaluation.log"),
    ],
)
log = logging.getLogger(__name__)

np.random.seed(42)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: Full Evaluation")
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


def build_model(name, params, pos_weight):
    """Reconstruct a model from saved Optuna params."""
    use_smote = params.pop("use_smote", True)
    needs_scaler = False

    if name == "XGBoost":
        clf = XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0)
    elif name == "LightGBM":
        clf = LGBMClassifier(**params, random_state=42, verbose=-1)
    elif name == "GradientBoosting":
        clf = GradientBoostingClassifier(**params, random_state=42)
    elif name == "CatBoost":
        if params.get("auto_class_weights") == "None":
            params["auto_class_weights"] = None
        clf = CatBoostClassifier(**params, random_seed=42, verbose=0)
    elif name == "RandomForest":
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif name == "LogisticRegression":
        clf = LogisticRegression(**params, max_iter=2000, class_weight="balanced", random_state=42)
        needs_scaler = True
    else:
        raise ValueError(f"Unknown model: {name}")

    if use_smote:
        steps = [("smoteenn", SMOTEENN(random_state=42))]
        if needs_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", clf))
        return ImbPipeline(steps), use_smote
    else:
        if needs_scaler:
            return ImbPipeline([("scaler", StandardScaler()), ("clf", clf)]), use_smote
        return clf, use_smote


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], precisions, recalls, thresholds


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    FIG_DIR = RESULTS_DIR / "figures"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 60)
    log.info("PHASE 5: COMPREHENSIVE EVALUATION (TUNED MODELS)")
    log.info("=" * 60)

    # ---- Load data ----
    log.info("[1/12] Loading data and tuned parameters...")
    data_path = f"{args.data_dir}/provider_features.csv"
    params_path = RESULTS_DIR / "best_params.json"

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        log.error(f"File not found: {data_path}")
        sys.exit(1)

    try:
        with open(params_path) as f:
            optuna_results = json.load(f)
    except FileNotFoundError:
        log.error(f"File not found: {params_path}. Run 04_optuna_tuning.py first.")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    X = df[feature_cols].values
    y = df["PotentialFraud"].values
    pos_weight = len(y[y == 0]) / len(y[y == 1])

    log.info(f"  {X.shape[0]} providers, {X.shape[1]} features")
    log.info(f"  Fraud rate: {y.mean()*100:.1f}%")

    # ---- Build tuned models ----
    log.info("[2/12] Building tuned models...")
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

    tuned_models = {}
    smote_status = {}
    for model_name, model_info in optuna_results["models"].items():
        params = dict(model_info["params"])
        try:
            pipeline, used_smote = build_model(model_name, params, pos_weight)
            tuned_models[model_name] = pipeline
            smote_status[model_name] = used_smote
            log.info(f"  {model_name}: loaded (SMOTE={'yes' if used_smote else 'no'})")
        except Exception as e:
            log.error(f"  Failed to build {model_name}: {e}")

    # Add DummyBaseline
    tuned_models["DummyBaseline"] = DummyClassifier(strategy="most_frequent")
    smote_status["DummyBaseline"] = False

    # ---- 10-fold CV evaluation ----
    log.info("[3/12] Running 10-fold stratified CV with tuned models...")
    all_results = {}

    for name, model in tuned_models.items():
        t0 = time.time()
        try:
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
            result["use_smote"] = smote_status.get(name, False)
            all_results[name] = result

            log.info(
                f"  {name:25s} | F1={result['f1']['mean']:.4f}+/-{result['f1']['std']:.4f} | "
                f"AUC-PR={result['average_precision']['mean']:.4f} | "
                f"MCC={result['mcc']['mean']:.4f} | {elapsed:.1f}s"
            )
        except Exception as e:
            log.error(f"  {name} failed: {e}")

    # ---- Build tuned ensembles ----
    log.info("[4/12] Building tuned ensembles...")

    # Get top 4 individual tuned models (excluding Dummy)
    individual_models = {k: v for k, v in all_results.items()
                        if k not in ["DummyBaseline"] and not k.startswith("_")}
    top4 = sorted(individual_models.keys(), key=lambda k: -individual_models[k]["f1"]["mean"])[:4]
    log.info(f"  Top 4 for ensembles: {top4}")

    # Rebuild estimators for ensemble (need fresh instances)
    ensemble_estimators = []
    for name in top4:
        params = dict(optuna_results["models"][name]["params"])
        params.pop("use_smote", None)
        if name == "XGBoost":
            est = XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0)
        elif name == "LightGBM":
            est = LGBMClassifier(**params, random_state=42, verbose=-1)
        elif name == "GradientBoosting":
            est = GradientBoostingClassifier(**params, random_state=42)
        elif name == "CatBoost":
            if params.get("auto_class_weights") == "None":
                params["auto_class_weights"] = None
            est = CatBoostClassifier(**params, random_seed=42, verbose=0)
        elif name == "RandomForest":
            est = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif name == "LogisticRegression":
            est = LogisticRegression(**params, max_iter=2000, class_weight="balanced", random_state=42)
        else:
            continue
        ensemble_estimators.append((name.lower()[:3], est))

    # SoftVoting
    if len(ensemble_estimators) >= 2:
        soft_vote = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=1)
        soft_pipe = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", soft_vote)])

        t0 = time.time()
        cv_soft = cross_validate(soft_pipe, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
        elapsed = time.time() - t0
        result_soft = {"use_smote": True, "time_seconds": elapsed}
        for metric in scoring:
            scores = cv_soft[f"test_{metric}"]
            result_soft[metric] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "values": scores.tolist()}
        all_results["SoftVoting"] = result_soft
        log.info(f"  {'SoftVoting':25s} | F1={result_soft['f1']['mean']:.4f} | {elapsed:.1f}s")

    # Stacking
    if len(ensemble_estimators) >= 2:
        stack = StackingClassifier(
            estimators=ensemble_estimators,
            final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=False, n_jobs=1,
        )
        stack_pipe = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", stack)])

        t0 = time.time()
        cv_stack = cross_validate(stack_pipe, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
        elapsed = time.time() - t0
        result_stack = {"use_smote": True, "time_seconds": elapsed}
        for metric in scoring:
            scores = cv_stack[f"test_{metric}"]
            result_stack[metric] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "values": scores.tolist()}
        all_results["Stacking"] = result_stack
        log.info(f"  {'Stacking':25s} | F1={result_stack['f1']['mean']:.4f} | {elapsed:.1f}s")

    # ---- Statistical tests ----
    log.info("[5/12] Statistical significance tests...")
    model_names = [k for k in all_results if k != "DummyBaseline" and not k.startswith("_")]
    f1_arrays = [all_results[k]["f1"]["values"] for k in model_names]

    if len(f1_arrays) >= 3:
        stat, p_val = stats.friedmanchisquare(*f1_arrays)
        log.info(f"  Friedman test (F1): chi2={stat:.4f}, p={p_val:.6f}")
        all_results["_friedman"] = {"statistic": float(stat), "p_value": float(p_val)}

    best_model = max(model_names, key=lambda k: all_results[k]["f1"]["mean"])
    log.info(f"  Best model by F1: {best_model}")

    raw_pvalues = {}
    for name in model_names:
        if name == best_model:
            continue
        best_scores = np.array(all_results[best_model]["f1"]["values"])
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
    for name in sorted(corrected_pvalues.keys()):
        adj_p = corrected_pvalues[name]
        sig = "***" if adj_p < 0.01 else "**" if adj_p < 0.05 else "*" if adj_p < 0.1 else "ns"
        log.info(f"    vs {name:25s}: adjusted_p={adj_p:.4f} [{sig}]")

    all_results["_wilcoxon"] = {
        "best_model": best_model,
        "raw_pvalues": raw_pvalues,
        "corrected_pvalues": {k: float(v) for k, v in corrected_pvalues.items()},
        "correction": "holm-bonferroni",
    }

    # ---- Threshold optimization ----
    log.info("[6/12] Threshold optimization...")

    # Get cross-validated predictions for the best model
    if best_model in tuned_models:
        best_pipe = tuned_models[best_model]
    elif best_model == "SoftVoting":
        best_pipe = soft_pipe
    elif best_model == "Stacking":
        best_pipe = stack_pipe
    else:
        best_pipe = tuned_models[list(tuned_models.keys())[1]]

    y_pred_proba = cross_val_predict(best_pipe, X, y, cv=outer_cv, method="predict_proba")[:, 1]
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    opt_threshold, opt_f1, precisions, recalls, thresholds = find_optimal_threshold(y, y_pred_proba)
    y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)

    log.info(f"  Default threshold (0.5):  F1={f1_score(y, y_pred_default):.4f}, "
             f"Precision={precision_score(y, y_pred_default):.4f}, "
             f"Recall={recall_score(y, y_pred_default):.4f}")
    log.info(f"  Optimal threshold ({opt_threshold:.3f}): F1={f1_score(y, y_pred_opt):.4f}, "
             f"Precision={precision_score(y, y_pred_opt):.4f}, "
             f"Recall={recall_score(y, y_pred_opt):.4f}")

    all_results["_threshold_optimization"] = {
        "best_model": best_model,
        "default_threshold": 0.5,
        "optimal_threshold": float(opt_threshold),
        "default_f1": float(f1_score(y, y_pred_default)),
        "optimal_f1": float(f1_score(y, y_pred_opt)),
        "default_precision": float(precision_score(y, y_pred_default)),
        "optimal_precision": float(precision_score(y, y_pred_opt)),
        "default_recall": float(recall_score(y, y_pred_default)),
        "optimal_recall": float(recall_score(y, y_pred_opt)),
    }

    # Plot threshold sensitivity
    f1_at_thresholds = []
    prec_at_thresholds = []
    rec_at_thresholds = []
    threshold_range = np.arange(0.1, 0.9, 0.01)
    for t in threshold_range:
        yp = (y_pred_proba >= t).astype(int)
        if yp.sum() == 0:
            f1_at_thresholds.append(0)
            prec_at_thresholds.append(0)
            rec_at_thresholds.append(0)
        else:
            f1_at_thresholds.append(f1_score(y, yp))
            prec_at_thresholds.append(precision_score(y, yp, zero_division=0))
            rec_at_thresholds.append(recall_score(y, yp))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_range, f1_at_thresholds, label="F1", linewidth=2)
    ax.plot(threshold_range, prec_at_thresholds, label="Precision", linewidth=1.5, linestyle="--")
    ax.plot(threshold_range, rec_at_thresholds, label="Recall", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.5, color="gray", linestyle=":", label="Default (0.5)")
    ax.axvline(x=opt_threshold, color="red", linestyle=":", label=f"Optimal ({opt_threshold:.2f})")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sensitivity Analysis ({best_model})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_threshold_sensitivity.png")
    plt.close()
    log.info("  Saved threshold sensitivity plot")

    # ---- Bootstrap CIs ----
    log.info("[7/12] Bootstrap 95% CIs...")
    n_bootstrap = 2000
    rng = np.random.RandomState(42)
    boot = {m: [] for m in ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc"]}

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        yb, ypb, yppb = y[idx], y_pred_opt[idx], y_pred_proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot["f1"].append(f1_score(yb, ypb))
        boot["precision"].append(precision_score(yb, ypb, zero_division=0))
        boot["recall"].append(recall_score(yb, ypb))
        boot["mcc"].append(matthews_corrcoef(yb, ypb))
        boot["roc_auc"].append(roc_auc_score(yb, yppb))
        boot["pr_auc"].append(average_precision_score(yb, yppb))

    ci_results = {}
    for metric, values in boot.items():
        v = np.array(values)
        ci_results[metric] = {
            "mean": float(np.mean(v)),
            "ci_lower": float(np.percentile(v, 2.5)),
            "ci_upper": float(np.percentile(v, 97.5)),
        }
        log.info(f"  {metric:12s}: {ci_results[metric]['mean']:.4f} "
                 f"(95% CI: {ci_results[metric]['ci_lower']:.4f}-{ci_results[metric]['ci_upper']:.4f})")

    all_results["_bootstrap_ci"] = {"model": best_model, "threshold": float(opt_threshold), "n_iterations": n_bootstrap, "metrics": ci_results}

    # Confusion matrices
    for label, yp, thresh in [("default", y_pred_default, 0.5), ("optimal", y_pred_opt, opt_threshold)]:
        cm = confusion_matrix(y, yp)
        tn, fp, fn, tp = cm.ravel()
        log.info(f"\n  Confusion Matrix ({label} threshold={thresh:.3f}):")
        log.info(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
        log.info(f"\n{classification_report(y, yp, target_names=['Non-Fraud', 'Fraud'])}")

    # ---- SMOTE Ablation ----
    log.info("[8/12] SMOTE ablation study...")
    ablation_results = {}

    # Pick 3 key models for ablation
    ablation_models = ["XGBoost", "LightGBM", "GradientBoosting"]
    for model_name in ablation_models:
        if model_name not in optuna_results["models"]:
            continue
        params = dict(optuna_results["models"][model_name]["params"])
        params.pop("use_smote", None)

        for smote_setting in [True, False]:
            label = f"{model_name}_{'smote' if smote_setting else 'nosmote'}"
            try:
                _, _ = build_model(model_name, {**params, "use_smote": smote_setting}, pos_weight)
                # Rebuild since build_model pops use_smote
                params2 = dict(params)
                if model_name == "XGBoost":
                    clf = XGBClassifier(**params2, random_state=42, eval_metric="logloss", verbosity=0)
                elif model_name == "LightGBM":
                    clf = LGBMClassifier(**params2, random_state=42, verbose=-1)
                elif model_name == "GradientBoosting":
                    clf = GradientBoostingClassifier(**params2, random_state=42)

                if smote_setting:
                    pipe = ImbPipeline([("smoteenn", SMOTEENN(random_state=42)), ("clf", clf)])
                else:
                    pipe = clf

                cv_abl = cross_validate(pipe, X, y, cv=outer_cv, scoring={"f1": "f1", "average_precision": "average_precision"}, n_jobs=1)
                ablation_results[label] = {
                    "f1_mean": float(np.mean(cv_abl["test_f1"])),
                    "f1_std": float(np.std(cv_abl["test_f1"])),
                    "aucpr_mean": float(np.mean(cv_abl["test_average_precision"])),
                }
                log.info(f"  {label:35s} | F1={ablation_results[label]['f1_mean']:.4f} | AUC-PR={ablation_results[label]['aucpr_mean']:.4f}")
            except Exception as e:
                log.error(f"  Ablation {label} failed: {e}")

    all_results["_smote_ablation"] = ablation_results

    # ---- Feature importance + selection ----
    log.info("[9/12] Permutation importance and feature selection...")

    # Fit best model on full data for permutation importance
    best_pipe.fit(X, y)

    # Extract classifier for permutation importance
    if hasattr(best_pipe, "named_steps") and "clf" in best_pipe.named_steps:
        fitted_clf = best_pipe.named_steps["clf"]
    else:
        fitted_clf = best_pipe

    perm_result = permutation_importance(
        best_pipe, X, y, n_repeats=10, random_state=42,
        scoring="average_precision", n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_df.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    log.info(f"  Top 10 features by permutation importance:")
    for _, row in perm_df.head(10).iterrows():
        log.info(f"    {row['feature']:35s} {row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")

    # Feature selection ablation: top-K features
    top_features = perm_df["feature"].tolist()
    feature_selection_results = {}
    for k in [10, 20, 30, 40, 52]:
        selected = top_features[:k]
        selected_idx = [feature_cols.index(f) for f in selected]
        X_sel = X[:, selected_idx]

        # Rebuild a fresh model for this
        params = dict(optuna_results["models"].get(best_model, list(optuna_results["models"].values())[0])["params"])
        params.pop("use_smote", None)
        sel_pipe, _ = build_model(best_model, {**params, "use_smote": True}, pos_weight)

        cv_sel = cross_validate(sel_pipe, X_sel, y, cv=outer_cv, scoring={"f1": "f1"}, n_jobs=1)
        f1_mean = float(np.mean(cv_sel["test_f1"]))
        feature_selection_results[f"top_{k}"] = {"n_features": k, "f1_mean": f1_mean}
        log.info(f"  Top-{k:2d} features: F1={f1_mean:.4f}")

    all_results["_feature_selection"] = feature_selection_results

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = perm_df.head(20)
    ax.barh(range(len(top20)), top20["importance_mean"], xerr=top20["importance_std"], align="center")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Permutation Importance (AUC-PR)")
    ax.set_title(f"Top 20 Features by Permutation Importance ({best_model})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_permutation_importance.png")
    plt.close()
    log.info("  Saved permutation importance plot")

    # ---- Learning curves ----
    log.info("[10/12] Learning curves...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipe, X, y, cv=5,
        train_sizes=np.linspace(0.2, 1.0, 8),
        scoring="f1", n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes_abs, train_scores.mean(axis=1), "o-", label="Training F1")
    ax.fill_between(train_sizes_abs,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    ax.plot(train_sizes_abs, val_scores.mean(axis=1), "o-", label="Validation F1")
    ax.fill_between(train_sizes_abs,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Learning Curves ({best_model})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_learning_curves.png")
    plt.close()
    log.info("  Saved learning curves plot")

    all_results["_learning_curves"] = {
        "train_sizes": train_sizes_abs.tolist(),
        "train_f1_mean": train_scores.mean(axis=1).tolist(),
        "val_f1_mean": val_scores.mean(axis=1).tolist(),
    }

    # ---- Calibration ----
    log.info("[11/12] Calibration analysis...")
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(prob_pred, prob_true, "o-", label=best_model)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve ({best_model})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_calibration.png")
    plt.close()
    log.info("  Saved calibration plot")

    # ---- SHAP + LIME ----
    log.info("[12/12] SHAP + LIME explainability...")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Get the fitted classifier
    clf = best_pipe
    if hasattr(best_pipe, "named_steps"):
        clf = best_pipe.named_steps.get("clf", best_pipe)

    y_pred_test = best_pipe.predict(X_test)
    y_proba_test = best_pipe.predict_proba(X_test)[:, 1]

    try:
        import shap

        # Try TreeExplainer first, fall back to others
        model_type = type(clf).__name__
        log.info(f"  SHAP model type: {model_type}")

        try:
            if model_type in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
                             "RandomForestClassifier", "GradientBoostingClassifier"]:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer(pd.DataFrame(X_test, columns=feature_cols))
            else:
                raise ValueError("Not a tree model, using fallback")
        except Exception as e1:
            log.info(f"  TreeExplainer failed ({e1}), training fallback GradientBoosting...")
            from imblearn.combine import SMOTEENN as SMOTEENN2
            smoteenn = SMOTEENN2(random_state=42)
            X_res, y_res = smoteenn.fit_resample(X_train, y_train)
            gb_fallback = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            gb_fallback.fit(X_res, y_res)
            explainer = shap.TreeExplainer(gb_fallback)
            shap_values = explainer(pd.DataFrame(X_test, columns=feature_cols))
            y_pred_test = gb_fallback.predict(X_test)
            y_proba_test = gb_fallback.predict_proba(X_test)[:, 1]

        # SHAP Beeswarm
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(f"SHAP Feature Importance (Tuned {best_model})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_shap_beeswarm_tuned.png")
        plt.close()

        # SHAP Bar
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.title("Mean |SHAP| Feature Importance (Tuned)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_shap_bar_tuned.png")
        plt.close()

        # Dependence for top 4
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-4:][::-1]
        top_feats = [feature_cols[i] for i in top_idx]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, (fi, fn) in enumerate(zip(top_idx, top_feats)):
            ax = axes[i // 2][i % 2]
            ax.scatter(X_test[:, fi], shap_values.values[:, fi], c=y_test, cmap="RdBu", alpha=0.5, s=10)
            ax.set_xlabel(fn)
            ax.set_ylabel("SHAP value")
            ax.set_title(f"Dependence: {fn}")
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.suptitle("SHAP Dependence Plots (Tuned, Top 4 Features)", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_shap_dependence_tuned.png")
        plt.close()

        # Waterfall plots
        tp_mask = (y_test == 1) & (y_pred_test == 1)
        fp_mask = (y_test == 0) & (y_pred_test == 1)
        fn_mask = (y_test == 1) & (y_pred_test == 0)

        for label, mask, title in [
            ("tp", tp_mask, "True Positive: Correctly Detected Fraud (Tuned)"),
            ("fp", fp_mask, "False Positive: Wrongly Flagged Provider (Tuned)"),
            ("fn", fn_mask, "False Negative: Missed Fraud (Tuned)"),
        ]:
            if mask.sum() > 0:
                idx = np.where(mask)[0][np.argmax(y_proba_test[mask]) if label != "fn" else np.argmin(y_proba_test[mask])]
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(FIG_DIR / f"fig_shap_waterfall_{label}_tuned.png")
                plt.close()

        # SHAP importance CSV
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        importance_df.to_csv(RESULTS_DIR / "shap_importance_tuned.csv", index=False)

        log.info("  SHAP figures saved")

    except ImportError:
        log.warning("  SHAP not installed")
    except Exception as e:
        log.error(f"  SHAP failed: {e}")

    # LIME
    try:
        from lime.lime_tabular import LimeTabularExplainer

        lime_explainer = LimeTabularExplainer(
            X_train, feature_names=feature_cols,
            class_names=["Non-Fraud", "Fraud"], mode="classification", random_state=42,
        )

        tp_mask = (y_test == 1) & (y_pred_test == 1)
        fp_mask = (y_test == 0) & (y_pred_test == 1)
        fn_mask = (y_test == 1) & (y_pred_test == 0)
        lime_cases = {}

        for label, mask, title in [
            ("tp", tp_mask, "LIME: True Positive (Tuned)"),
            ("fp", fp_mask, "LIME: False Positive (Tuned)"),
            ("fn", fn_mask, "LIME: False Negative (Tuned)"),
        ]:
            if mask.sum() > 0:
                idx = np.where(mask)[0][np.argmax(y_proba_test[mask]) if label != "fn" else np.argmin(y_proba_test[mask])]
                exp = lime_explainer.explain_instance(X_test[idx], best_pipe.predict_proba, num_features=10)
                lime_cases[label] = exp.as_list()
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(10, 6)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(FIG_DIR / f"fig_lime_{label}_tuned.png")
                plt.close()

        with open(RESULTS_DIR / "lime_cases_tuned.json", "w") as f:
            json.dump(lime_cases, f, indent=2, default=str)

        log.info("  LIME figures saved")
    except ImportError:
        log.warning("  LIME not installed")
    except Exception as e:
        log.error(f"  LIME failed: {e}")

    # ---- Save final model and results ----
    log.info("Saving final model and results...")
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, RESULTS_DIR / "best_model_tuned.pkl")
    joblib.dump(feature_cols, RESULTS_DIR / "feature_cols.pkl")

    # Comparison with baseline
    try:
        with open(RESULTS_DIR / "cv_results.json") as f:
            baseline = json.load(f)
        baseline_best_f1 = max(
            v["f1"]["mean"] for k, v in baseline.items()
            if not k.startswith("_") and isinstance(v, dict) and "f1" in v
        )
        tuned_best_f1 = all_results[best_model]["f1"]["mean"]
        improvement = tuned_best_f1 - baseline_best_f1
        log.info(f"\n  Baseline best F1: {baseline_best_f1:.4f}")
        log.info(f"  Tuned best F1:    {tuned_best_f1:.4f}")
        log.info(f"  Improvement:      {improvement:+.4f} ({improvement/baseline_best_f1*100:+.1f}%)")
        all_results["_improvement"] = {
            "baseline_best_f1": baseline_best_f1,
            "tuned_best_f1": tuned_best_f1,
            "absolute_improvement": improvement,
            "relative_improvement_pct": improvement / baseline_best_f1 * 100,
        }
    except Exception:
        pass

    all_results["_limitations"] = [
        "Shared beneficiaries across providers create dependencies between CV folds.",
        "Optuna tuning used same data for HPO and evaluation (not fully nested CV). "
        "Best params found on 5-fold inner CV, then evaluated on 10-fold outer CV with fixed params.",
        "SMOTE-ENN inside Stacking wrapper means synthetic samples may span stacking inner folds.",
    ]

    with open(RESULTS_DIR / "tuned_cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    log.info("\n" + "=" * 115)
    log.info(f"{'Model':25s} | {'F1':^14s} | {'AUC-PR':^14s} | {'MCC':^14s} | {'Recall':^14s} | {'SMOTE':>5s} | {'Time':>8s}")
    log.info("-" * 115)
    for name in sorted(all_results.keys()):
        if name.startswith("_"):
            continue
        r = all_results[name]
        smote_flag = "yes" if r.get("use_smote", False) else "no"
        log.info(
            f"{name:25s} | "
            f"{r['f1']['mean']:.4f}+/-{r['f1']['std']:.4f} | "
            f"{r['average_precision']['mean']:.4f}+/-{r['average_precision']['std']:.4f} | "
            f"{r['mcc']['mean']:.4f}+/-{r['mcc']['std']:.4f} | "
            f"{r['recall']['mean']:.4f}+/-{r['recall']['std']:.4f} | "
            f"{smote_flag:>5s} | {r['time_seconds']:7.1f}s"
        )
    log.info("=" * 115)
    log.info(f"Best model: {best_model}")
    log.info(f"Optimal threshold: {opt_threshold:.3f}")
    log.info(f"Results saved to {RESULTS_DIR}")
    log.info("Phase 5 complete.")


if __name__ == "__main__":
    main()
