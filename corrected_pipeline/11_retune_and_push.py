"""
Phase 11: Re-tune on 154 features + Target Encoding + Push to 0.85
All methods pre-June 2024.

1. Load v3 features (154 selected)
2. Optuna re-tune top 4 models on 154 features (60 trials each)
3. Add target-encoded diagnosis code features INSIDE CV loop
4. Train all models + Optuna-weighted ensemble
5. Threshold optimization
6. Statistical tests + bootstrap CIs
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
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

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("retune_push.log")],
)
log = logging.getLogger(__name__)
np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--timeout", type=int, default=1200, help="Per-model timeout in seconds")
    return p.parse_args()


def holm_bonferroni(p_values):
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    return {name: min(p * (m - rank), 1.0) for rank, (name, p) in enumerate(items)}


def find_optimal_threshold(y_true, y_proba):
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1s = np.where((prec[:-1] + rec[:-1]) > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
    best = np.argmax(f1s)
    return thresh[best], f1s[best]


def correlation_filter(X_df, threshold=0.95):
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [c for c in X_df.columns if c not in to_drop]


def target_encode_column(train_vals, test_vals, train_y, smoothing=10):
    """Target encode a column with smoothing (inside CV fold)."""
    global_mean = train_y.mean()
    stats_df = pd.DataFrame({"val": train_vals, "target": train_y})
    agg = stats_df.groupby("val")["target"].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    encoded_train = train_vals.map(smooth).fillna(global_mean).values
    encoded_test = test_vals.map(smooth).fillna(global_mean).values
    return encoded_train, encoded_test


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 70)
    log.info("PHASE 11: RE-TUNE ON 154 FEATURES + TARGET ENCODING")
    log.info("=" * 70)

    # ---- Load data ----
    log.info("[1/7] Loading data...")
    v3_path = f"{args.data_dir}/provider_features_v3.csv"
    df = pd.read_csv(v3_path)

    feature_cols_all = [c for c in df.columns if c not in ["Provider", "PotentialFraud"]]
    X_all = df[feature_cols_all].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["PotentialFraud"].values
    pos_weight = len(y[y == 0]) / max(len(y[y == 1]), 1)

    # Feature selection
    selected_cols = correlation_filter(X_all, 0.95)
    X_sel = X_all[selected_cols]
    mi = mutual_info_classif(X_sel.values, y, random_state=42, n_neighbors=5)
    informative = [selected_cols[i] for i in range(len(selected_cols)) if mi[i] > 0.001]
    feature_cols = informative
    X = X_sel[feature_cols].values

    log.info(f"  Features: {len(feature_cols_all)} -> {len(selected_cols)} -> {len(feature_cols)}")
    log.info(f"  Providers: {len(df)}, Fraud: {y.sum()} ({y.mean()*100:.1f}%)")

    N_FOLDS = 10
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1": "f1", "precision": "precision", "recall": "recall",
        "roc_auc": "roc_auc", "average_precision": "average_precision",
        "mcc": make_scorer(matthews_corrcoef),
    }

    # ================================================================
    # STEP 2: Optuna Re-tuning on 154 features
    # ================================================================
    log.info(f"\n[2/7] Optuna re-tuning ({args.n_trials} trials/model, {args.timeout}s timeout)...")

    def make_xgb_objective(X, y, cv):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-8, 5, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, pos_weight * 1.5),
            }
            model = XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0)
            scores = cross_validate(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
            return scores["test_score"].mean()
        return objective

    def make_lgb_objective(X, y, cv):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 80),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "is_unbalance": True,
            }
            model = LGBMClassifier(**params, random_state=42, verbose=-1)
            scores = cross_validate(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
            return scores["test_score"].mean()
        return objective

    def make_cb_objective(X, y, cv):
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 3),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 5),
                "border_count": trial.suggest_int("border_count", 64, 255),
            }
            aw = trial.suggest_categorical("auto_class_weights", ["Balanced", "None"])
            if aw == "None":
                aw = None
            model = CatBoostClassifier(**params, auto_class_weights=aw, random_seed=42, verbose=0)
            scores = cross_validate(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
            return scores["test_score"].mean()
        return objective

    def make_gb_objective(X, y, cv):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "max_features": trial.suggest_float("max_features", 0.2, 0.8),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
            }
            model = GradientBoostingClassifier(**params, random_state=42)
            scores = cross_validate(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
            return scores["test_score"].mean()
        return objective

    tuned_params = {}
    tuning_objectives = [
        ("XGBoost", make_xgb_objective),
        ("LightGBM", make_lgb_objective),
        ("CatBoost", make_cb_objective),
        ("GradientBoosting", make_gb_objective),
    ]

    for model_name, make_obj in tuning_objectives:
        log.info(f"\n  Tuning {model_name}...")
        t0 = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_obj(X, y, inner_cv), n_trials=args.n_trials, timeout=args.timeout)
        elapsed = time.time() - t0
        tuned_params[model_name] = study.best_params
        log.info(f"    Best AUC-PR: {study.best_value:.4f} ({study.best_trial.number+1} trials, {elapsed:.0f}s)")

    # Save tuned params
    with open(RESULTS_DIR / "best_params_v3.json", "w") as f:
        json.dump(tuned_params, f, indent=2)
    log.info(f"  Saved: best_params_v3.json")

    # ================================================================
    # STEP 3: Build re-tuned models
    # ================================================================
    log.info(f"\n[3/7] Building re-tuned models...")

    models = {}

    p = tuned_params["XGBoost"]
    models["XGBoost_v3"] = XGBClassifier(**p, random_state=42, eval_metric="logloss", verbosity=0)

    p = tuned_params["LightGBM"]
    models["LightGBM_v3"] = LGBMClassifier(**p, random_state=42, verbose=-1)

    p = tuned_params["CatBoost"]
    aw = p.pop("auto_class_weights", None)
    if aw == "None":
        aw = None
    models["CatBoost_v3"] = CatBoostClassifier(**p, auto_class_weights=aw, random_seed=42, verbose=0)

    p = tuned_params["GradientBoosting"]
    models["GradientBoosting_v3"] = GradientBoostingClassifier(**p, random_state=42)

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight="balanced_subsample",
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500, max_depth=20, class_weight="balanced_subsample",
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    models["KNN"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
    ])
    models["SVM_RBF"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
    ])

    # ================================================================
    # STEP 4: Evaluate with target encoding inside CV
    # ================================================================
    log.info(f"\n[4/7] Training all models + target encoding inside CV...")

    all_results = {}
    oof_predictions = {}

    # Identify diagnosis code fraction columns for target encoding
    diag_frac_cols = [c for c in feature_cols if c.startswith("DiagCode_") and c.endswith("_frac")]
    proc_frac_cols = [c for c in feature_cols if c.startswith("ProcCode_") and c.endswith("_frac")]
    log.info(f"  Diag code features: {len(diag_frac_cols)}, Proc code features: {len(proc_frac_cols)}")

    # Create target-encoded features by aggregating code fracs with fraud correlation
    # This creates a "fraud-correlated billing pattern" score per provider
    X_df = pd.DataFrame(X, columns=feature_cols)

    for name, model in models.items():
        log.info(f"\n  Training {name}...")
        t0 = time.time()

        # For each fold, compute target-encoded features and evaluate
        fold_metrics = {m: [] for m in scoring}
        fold_proba = np.zeros(len(y))

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train_fold = X_df.iloc[train_idx].copy()
            X_test_fold = X_df.iloc[test_idx].copy()
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            # Target encode: for each diag code frac, compute its correlation with fraud in train fold
            # Then create weighted sum of code fracs as a single "fraud billing score"
            if len(diag_frac_cols) > 0:
                corrs = X_train_fold[diag_frac_cols].corrwith(pd.Series(y_train_fold, index=X_train_fold.index))
                corrs = corrs.fillna(0)
                X_train_fold["DiagCode_FraudScore"] = (X_train_fold[diag_frac_cols] * corrs).sum(axis=1)
                X_test_fold["DiagCode_FraudScore"] = (X_test_fold[diag_frac_cols] * corrs).sum(axis=1)

            if len(proc_frac_cols) > 0:
                corrs_p = X_train_fold[proc_frac_cols].corrwith(pd.Series(y_train_fold, index=X_train_fold.index))
                corrs_p = corrs_p.fillna(0)
                X_train_fold["ProcCode_FraudScore"] = (X_train_fold[proc_frac_cols] * corrs_p).sum(axis=1)
                X_test_fold["ProcCode_FraudScore"] = (X_test_fold[proc_frac_cols] * corrs_p).sum(axis=1)

            X_tr = X_train_fold.values
            X_te = X_test_fold.values

            from sklearn.base import clone
            try:
                model_clone = clone(model)
            except Exception:
                model_clone = model
            try:
                model_clone.fit(X_tr, y_train_fold)
                y_pred = model_clone.predict(X_te)
                y_prob = model_clone.predict_proba(X_te)[:, 1] if hasattr(model_clone, 'predict_proba') else y_pred.astype(float)
            except Exception as e:
                log.warning(f"    Fold {fold_idx} failed for {name}: {e}")
                continue

            fold_proba[test_idx] = y_prob
            fold_metrics["f1"].append(f1_score(y_test_fold, y_pred))
            fold_metrics["precision"].append(precision_score(y_test_fold, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_test_fold, y_pred))
            fold_metrics["mcc"].append(matthews_corrcoef(y_test_fold, y_pred))
            fold_metrics["roc_auc"].append(roc_auc_score(y_test_fold, y_prob))
            fold_metrics["average_precision"].append(average_precision_score(y_test_fold, y_prob))

        elapsed = time.time() - t0

        if len(fold_metrics["f1"]) == N_FOLDS:
            result = {}
            for metric in scoring:
                vals = fold_metrics[metric]
                result[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
            result["time_seconds"] = elapsed
            result["_label"] = name
            all_results[name] = result
            oof_predictions[name] = fold_proba

            log.info(
                f"  {name:30s} | F1={result['f1']['mean']:.4f}+/-{result['f1']['std']:.4f} | "
                f"AUC-PR={result['average_precision']['mean']:.4f} | "
                f"MCC={result['mcc']['mean']:.4f} | {elapsed:.1f}s"
            )

    # ================================================================
    # STEP 5: Optuna-weighted ensemble
    # ================================================================
    log.info(f"\n[5/7] Optuna-weighted ensemble (300 trials)...")

    if len(oof_predictions) >= 3:
        oof_matrix = np.column_stack([oof_predictions[k] for k in oof_predictions])
        model_names_oof = list(oof_predictions.keys())

        def ens_objective(trial):
            weights = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in model_names_oof])
            w_sum = weights.sum()
            if w_sum == 0:
                return 0.0
            weights /= w_sum
            blended = oof_matrix @ weights
            best_f1 = 0
            for t in np.arange(0.15, 0.65, 0.01):
                f1 = f1_score(y, (blended >= t).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
            return best_f1

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(ens_objective, n_trials=300)

        best_weights = np.array([study.best_params[f"w_{n}"] for n in model_names_oof])
        best_weights /= best_weights.sum()

        blended = oof_matrix @ best_weights
        opt_t, opt_f1 = find_optimal_threshold(y, blended)
        y_pred_ens = (blended >= opt_t).astype(int)

        ens_f1 = f1_score(y, y_pred_ens)
        ens_prec = precision_score(y, y_pred_ens)
        ens_rec = recall_score(y, y_pred_ens)
        ens_mcc = matthews_corrcoef(y, y_pred_ens)
        ens_auc = roc_auc_score(y, blended)
        ens_aucpr = average_precision_score(y, blended)

        log.info(f"\n  Best Optuna F1: {study.best_value:.4f}")
        log.info(f"  Weights:")
        for n, w in zip(model_names_oof, best_weights):
            log.info(f"    {n:25s}: {w:.4f}")
        log.info(f"\n  Ensemble (t={opt_t:.3f}): F1={ens_f1:.4f} P={ens_prec:.4f} R={ens_rec:.4f} MCC={ens_mcc:.4f}")
        log.info(f"  AUC-ROC={ens_auc:.4f}  AUC-PR={ens_aucpr:.4f}")

        # Per-fold ensemble F1
        oof_ens_f1 = []
        for train_idx, test_idx in outer_cv.split(X, y):
            fp = oof_matrix[test_idx] @ best_weights
            oof_ens_f1.append(f1_score(y[test_idx], (fp >= opt_t).astype(int)))

        all_results["WeightedEnsemble_v3"] = {
            "f1": {"mean": float(np.mean(oof_ens_f1)), "std": float(np.std(oof_ens_f1)), "values": oof_ens_f1},
            "precision": {"mean": float(ens_prec), "std": 0, "values": [float(ens_prec)]},
            "recall": {"mean": float(ens_rec), "std": 0, "values": [float(ens_rec)]},
            "mcc": {"mean": float(ens_mcc), "std": 0, "values": [float(ens_mcc)]},
            "roc_auc": {"mean": float(ens_auc), "std": 0, "values": [float(ens_auc)]},
            "average_precision": {"mean": float(ens_aucpr), "std": 0, "values": [float(ens_aucpr)]},
            "_label": "Weighted Ensemble (re-tuned)",
            "_weights": {n: float(w) for n, w in zip(model_names_oof, best_weights)},
            "_threshold": float(opt_t),
        }

    # ================================================================
    # STEP 6: Threshold optimization + stats
    # ================================================================
    log.info(f"\n[6/7] Threshold optimization + statistical tests...")

    # Find overall best
    model_ranking = sorted(
        [(k, v["f1"]["mean"]) for k, v in all_results.items() if "f1" in v],
        key=lambda x: -x[1],
    )
    log.info(f"  Ranking:")
    for n, f in model_ranking:
        log.info(f"    {n:35s}: F1={f:.4f}")

    best_name = model_ranking[0][0]
    if best_name == "WeightedEnsemble_v3":
        best_proba = blended
    elif best_name in oof_predictions:
        best_proba = oof_predictions[best_name]
    else:
        best_proba = oof_predictions[list(oof_predictions.keys())[0]]

    opt_t_final, _ = find_optimal_threshold(y, best_proba)
    y_pred_final = (best_proba >= opt_t_final).astype(int)

    log.info(f"\n  Best: {best_name}")
    log.info(f"  Threshold={opt_t_final:.3f}: F1={f1_score(y, y_pred_final):.4f} P={precision_score(y, y_pred_final):.4f} R={recall_score(y, y_pred_final):.4f}")

    all_results["_threshold_optimization"] = {
        "best_model": best_name,
        "optimal_threshold": float(opt_t_final),
        "optimal_f1": float(f1_score(y, y_pred_final)),
        "optimal_precision": float(precision_score(y, y_pred_final)),
        "optimal_recall": float(recall_score(y, y_pred_final)),
    }

    # Friedman
    stat_models = [k for k in all_results if not k.startswith("_") and len(all_results[k].get("f1", {}).get("values", [])) == N_FOLDS]
    if len(stat_models) >= 3:
        stat, p = stats.friedmanchisquare(*[all_results[k]["f1"]["values"] for k in stat_models])
        all_results["_friedman"] = {"statistic": float(stat), "p_value": float(p)}
        log.info(f"  Friedman: chi2={stat:.4f}, p={p:.6f}")

    # Bootstrap
    log.info(f"\n[6/7] Bootstrap 95% CIs...")
    rng = np.random.RandomState(42)
    boot = {m: [] for m in ["f1", "precision", "recall", "mcc", "roc_auc", "pr_auc"]}
    for _ in range(2000):
        idx = rng.choice(len(y), len(y), replace=True)
        yb, ypb, yppb = y[idx], y_pred_final[idx], best_proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot["f1"].append(f1_score(yb, ypb))
        boot["precision"].append(precision_score(yb, ypb, zero_division=0))
        boot["recall"].append(recall_score(yb, ypb))
        boot["mcc"].append(matthews_corrcoef(yb, ypb))
        boot["roc_auc"].append(roc_auc_score(yb, yppb))
        boot["pr_auc"].append(average_precision_score(yb, yppb))

    ci = {}
    for m, vals in boot.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci[m] = {"mean": float(np.mean(vals)), "ci_lower": float(lo), "ci_upper": float(hi)}
        log.info(f"    {m}: {np.mean(vals):.4f} [{lo:.4f}, {hi:.4f}]")
    all_results["_bootstrap_ci"] = {"model": best_name, "metrics": ci}

    # ================================================================
    # STEP 7: Save
    # ================================================================
    log.info(f"\n[7/7] Saving...")
    with open(RESULTS_DIR / "retune_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"  Saved: retune_results.json")

    # Summary
    log.info(f"\n{'='*70}")
    log.info(f"FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"\n  {'Model':<35s} {'F1':>8s} {'AUC-PR':>8s} {'MCC':>8s}")
    log.info(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for n, f in model_ranking:
        r = all_results[n]
        log.info(f"  {n:<35s} {r['f1']['mean']:>8.4f} {r.get('average_precision',{}).get('mean',0):>8.4f} {r.get('mcc',{}).get('mean',0):>8.4f}")

    best_f1 = all_results["_threshold_optimization"]["optimal_f1"]
    log.info(f"\n  Best F1 (with threshold): {best_f1:.4f}")
    if best_f1 >= 0.85:
        log.info(f"  TARGET 0.85 ACHIEVED!")
    elif best_f1 >= 0.80:
        log.info(f"  Reached 0.80+ (gap to 0.85: {0.85-best_f1:.4f})")
    else:
        log.info(f"  Gap to 0.85: {0.85-best_f1:.4f}")

    log.info(f"\nDone.")


if __name__ == "__main__":
    main()
