"""
Phase 3: SHAP + LIME Explainability
All methods pre-June 2024.

Fixes applied:
- Loads best model from Phase 2 (not always XGBoost)
- LIME uses original X_train (not SMOTE-resampled X_train_res)
- Proper main() guard
- Configurable paths
- Error handling
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    p = argparse.ArgumentParser(description="Phase 3: Explainability")
    p.add_argument("--data-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--results-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.results_dir)
    FIG_DIR = RESULTS_DIR / "figures"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    log.info("=" * 60)
    log.info("PHASE 3: EXPLAINABILITY (ALL FIXES APPLIED)")
    log.info("=" * 60)

    # ---- Load data ----
    log.info("[1/5] Loading data and best model...")
    try:
        df = pd.read_csv(f"{args.data_dir}/provider_features.csv")
        feature_cols = joblib.load(RESULTS_DIR / "feature_cols.pkl")
        best_pipeline = joblib.load(RESULTS_DIR / "best_model.pkl")
    except FileNotFoundError as e:
        log.error(f"Missing file: {e}. Run 01_preprocess.py and 02_train_evaluate.py first.")
        sys.exit(1)

    # Load CV results to know which model won
    try:
        with open(RESULTS_DIR / "cv_results.json") as f:
            cv_results = json.load(f)
        best_model_name = cv_results.get("_bootstrap_ci", {}).get("model", "unknown")
    except Exception:
        best_model_name = "unknown"

    X = df[feature_cols].values
    y = df["PotentialFraud"].values
    log.info(f"  {X.shape[0]} providers, {X.shape[1]} features")
    log.info(f"  Best model from Phase 2: {best_model_name}")

    # ---- Train/test split for SHAP ----
    log.info("[2/5] Preparing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    log.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # FIX: Extract the actual classifier from the pipeline for SHAP
    # The best_pipeline is an ImbPipeline with SMOTE-ENN + classifier
    # We need the fitted classifier for TreeExplainer
    clf = best_pipeline
    if hasattr(best_pipeline, "named_steps"):
        clf = best_pipeline.named_steps.get("clf", best_pipeline)

    # Get predictions from the full pipeline
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, y_pred)
    log.info(f"  Test F1: {test_f1:.4f}")

    # ---- SHAP Analysis ----
    log.info("[3/5] Computing SHAP values...")

    # Determine explainer type based on model
    model_type = type(clf).__name__
    log.info(f"  Model type for SHAP: {model_type}")

    try:
        if hasattr(clf, "estimators_") or model_type in [
            "XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
            "GradientBoostingClassifier", "AdaBoostClassifier",
        ]:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(pd.DataFrame(X_test, columns=feature_cols))
        else:
            # Fallback to KernelExplainer for non-tree models
            log.info("  Using KernelExplainer (slower)...")
            background = shap.sample(pd.DataFrame(X_train, columns=feature_cols), 100)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            raw_shap = explainer.shap_values(X_test)
            if isinstance(raw_shap, list):
                raw_shap = raw_shap[1]  # positive class
            shap_values = shap.Explanation(
                values=raw_shap,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_test,
                feature_names=feature_cols,
            )
    except Exception as e:
        log.error(f"  SHAP failed: {e}")
        log.info("  Falling back to training a fresh XGBoost for SHAP...")
        from xgboost import XGBClassifier
        from imblearn.combine import SMOTEENN

        smoteenn = SMOTEENN(random_state=42)
        X_res, y_res = smoteenn.fit_resample(X_train, y_train)
        pos_w = len(y[y == 0]) / len(y[y == 1])
        xgb_fallback = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=pos_w, random_state=42, eval_metric="logloss", verbosity=0,
        )
        xgb_fallback.fit(X_res, y_res)
        explainer = shap.TreeExplainer(xgb_fallback)
        shap_values = explainer(pd.DataFrame(X_test, columns=feature_cols))
        y_pred = xgb_fallback.predict(X_test)
        y_proba = xgb_fallback.predict_proba(X_test)[:, 1]
        log.info(f"  Fallback XGBoost Test F1: {f1_score(y_test, y_pred):.4f}")

    log.info(f"  SHAP values shape: {shap_values.values.shape}")

    # ---- Generate figures ----
    log.info("[4/5] Generating publication figures...")

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Feature Importance ({best_model_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_shap_beeswarm.png")
    plt.close()
    log.info("  Saved beeswarm plot")

    # Bar
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("Mean |SHAP| Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_shap_bar.png")
    plt.close()
    log.info("  Saved bar plot")

    # Dependence plots for top 4
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
    plt.suptitle("SHAP Dependence Plots (Top 4 Features)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_shap_dependence.png")
    plt.close()
    log.info("  Saved dependence plots")

    # Waterfall plots
    tp_mask = (y_test == 1) & (y_pred == 1)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    for label, mask, title in [
        ("tp", tp_mask, "True Positive: Correctly Detected Fraud"),
        ("fp", fp_mask, "False Positive: Wrongly Flagged Provider"),
        ("fn", fn_mask, "False Negative: Missed Fraud"),
    ]:
        if mask.sum() > 0:
            idx = np.where(mask)[0][np.argmax(y_proba[mask]) if label != "fn" else np.argmin(y_proba[mask])]
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"fig_shap_waterfall_{label}.png")
            plt.close()
            log.info(f"  Saved waterfall {label}")

    # SHAP importance CSV
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

    log.info("\n  Top 10 features by SHAP:")
    for _, row in importance_df.head(10).iterrows():
        log.info(f"    {row['feature']:35s} {row['mean_abs_shap']:.4f}")

    # ---- LIME ----
    log.info("[5/5] LIME case studies...")
    try:
        from lime.lime_tabular import LimeTabularExplainer

        # FIX: Use original X_train (not SMOTE-resampled) for LIME background
        lime_explainer = LimeTabularExplainer(
            X_train,  # Original data, not X_train_res
            feature_names=feature_cols,
            class_names=["Non-Fraud", "Fraud"],
            mode="classification",
            random_state=42,
        )

        cases = {}
        predict_fn = best_pipeline.predict_proba

        for label, mask, title in [
            ("tp", tp_mask, "LIME: True Positive"),
            ("fp", fp_mask, "LIME: False Positive"),
            ("fn", fn_mask, "LIME: False Negative"),
        ]:
            if mask.sum() > 0:
                idx = np.where(mask)[0][
                    np.argmax(y_proba[mask]) if label != "fn" else np.argmin(y_proba[mask])
                ]
                exp = lime_explainer.explain_instance(X_test[idx], predict_fn, num_features=10)
                cases[label] = exp.as_list()
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(10, 6)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(FIG_DIR / f"fig_lime_{label}.png")
                plt.close()
                log.info(f"  Saved LIME {label}")

        with open(RESULTS_DIR / "lime_cases.json", "w") as f:
            json.dump(cases, f, indent=2, default=str)

    except ImportError:
        log.warning("  LIME not installed, skipping. Install with: pip install lime")

    # Save SHAP artifacts
    np.save(RESULTS_DIR / "shap_values.npy", shap_values.values)
    joblib.dump(importance_df, RESULTS_DIR / "shap_importance_df.pkl")

    log.info(f"\nAll figures saved to {FIG_DIR}")
    figs = list(FIG_DIR.glob("*.png"))
    log.info(f"Generated {len(figs)} figures: {[f.name for f in figs]}")
    log.info("Explainability analysis complete.")


if __name__ == "__main__":
    main()
