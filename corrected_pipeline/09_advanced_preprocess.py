"""
Phase 9: Advanced Feature Engineering for F1 0.85+ Target
Dataset: Kaggle rohitrox Healthcare Provider Fraud Detection
All methods pre-June 2024.

Generates ~180 features from raw claim data:
  - 60 existing features (from 01_preprocess.py)
  - ~50 diagnosis code fraction features
  - ~20 procedure code fraction features
  - 6 physician behavior features
  - 4 geographic features
  - 5 beneficiary network features
  - 8 claim-type separated stats
  - 8 z-score features
  - 6 percentile rank features
  - 6 distributional tail features
  - 6 interaction features
  - 1 anomaly score (IsolationForest)
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, kurtosis as scipy_kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

np.random.seed(42)

TOP_N_DIAG = 50
TOP_N_PROC = 20


def parse_args():
    p = argparse.ArgumentParser(description="Phase 9: Advanced Preprocessing")
    p.add_argument("--data-dir", default="/home/ubuntu/data", help="Directory with Kaggle CSVs")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline", help="Output directory")
    p.add_argument("--out-filename", default="provider_features_v3.csv", help="Output CSV filename")
    return p.parse_args()


def load_csv(path, name):
    try:
        df = pd.read_csv(path)
        log.info(f"  Loaded {name}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        log.error(f"  File not found: {path}")
        sys.exit(1)


def main():
    args = parse_args()
    start = time.time()

    log.info("=" * 70)
    log.info("PHASE 9: ADVANCED FEATURE ENGINEERING (180 features)")
    log.info("=" * 70)

    # ---- Load datasets ----
    log.info("[1/11] Loading datasets...")
    D = args.data_dir
    train_labels = load_csv(f"{D}/Train-1542865627584.csv", "Provider labels")
    beneficiary = load_csv(f"{D}/Train_Beneficiarydata-1542865627584.csv", "Beneficiary")
    inpatient = load_csv(f"{D}/Train_Inpatientdata-1542865627584.csv", "Inpatient")
    outpatient = load_csv(f"{D}/Train_Outpatientdata-1542865627584.csv", "Outpatient")

    fraud_dist = train_labels["PotentialFraud"].value_counts().to_dict()
    log.info(f"  Fraud distribution: {fraud_dist}")

    # ---- Merge Inpatient + Outpatient ----
    log.info("[2/11] Merging inpatient + outpatient claims...")
    common_cols = [c for c in inpatient.columns if c in outpatient.columns]
    claims = outpatient.merge(inpatient, on=common_cols, how="outer", indicator="claim_type")
    claims["is_inpatient"] = (claims["claim_type"] == "right_only").astype(int)
    claims.drop("claim_type", axis=1, inplace=True)
    log.info(f"  Combined claims: {claims.shape[0]}")

    # ---- Merge with beneficiary ----
    log.info("[3/11] Joining with beneficiary data...")
    claims_benef = beneficiary.merge(claims, on="BeneID")
    log.info(f"  Claims+Beneficiary: {claims_benef.shape[0]}")

    # ---- Merge with provider labels ----
    full = train_labels.merge(claims_benef, on="Provider")
    full["PotentialFraud"] = full["PotentialFraud"].map({"Yes": 1, "No": 0})
    assert full["PotentialFraud"].isna().sum() == 0, "NaN in target"
    log.info(f"  Full dataset: {full.shape[0]} rows, {full.shape[1]} cols")

    # ---- Claim-level feature engineering (existing) ----
    log.info("[4/11] Engineering claim-level features...")

    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if col in full.columns:
            full[col] = pd.to_datetime(full[col], errors="coerce")
    full["DOB"] = pd.to_datetime(full["DOB"], errors="coerce")
    full["DOD"] = pd.to_datetime(full["DOD"], errors="coerce")

    full["Age"] = ((full["ClaimStartDt"] - full["DOB"]).dt.days / 365.25).fillna(0).astype(int)
    full["is_dead"] = full["DOD"].notna().astype(int)
    full["Claim_Duration"] = (full["ClaimEndDt"] - full["ClaimStartDt"]).dt.days.fillna(0).astype(int)
    full["Days_Admitted"] = (full["DischargeDt"] - full["AdmissionDt"]).dt.days.fillna(0).astype(int)
    full["Claim_GT_Admitted"] = (full["Claim_Duration"] > full["Days_Admitted"]).astype(int)

    chronic_cols = [c for c in full.columns if c.startswith("ChronicCond_")]
    for col in chronic_cols:
        full[col] = full[col].replace(2, 0)
    full["RenalDiseaseIndicator"] = full["RenalDiseaseIndicator"].replace({"0": 0, "Y": 1, 0: 0}).astype(int)
    full["Disease_Count"] = full[chronic_cols].sum(axis=1)

    phys_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    full["Physician_Count"] = full[phys_cols].notna().sum(axis=1)
    full["Unique_Physicians"] = full[phys_cols].nunique(axis=1)
    full["Same_Physician"] = ((full["Unique_Physicians"] == 1) & (full["Physician_Count"] > 1)).astype(int)

    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    diag_cols_present = [c for c in diag_cols if c in full.columns]
    proc_cols_present = [c for c in proc_cols if c in full.columns]

    full["Diag_Code_Count"] = full[diag_cols_present].notna().sum(axis=1)
    full["Proc_Code_Count"] = full[proc_cols_present].notna().sum(axis=1)
    full["Has_AdmitDiag"] = full["ClmAdmitDiagnosisCode"].notna().astype(int) if "ClmAdmitDiagnosisCode" in full.columns else 0
    full["Has_GroupCode"] = full["DiagnosisGroupCode"].notna().astype(int) if "DiagnosisGroupCode" in full.columns else 0

    full["DeductibleAmtPaid"] = full["DeductibleAmtPaid"].fillna(0)
    full["Has_Deductible"] = (full["DeductibleAmtPaid"] > 0).astype(int)
    full["TotalClaimAmt"] = full["InscClaimAmtReimbursed"].fillna(0) + full["DeductibleAmtPaid"]
    full["IP_TotalAmt"] = full["IPAnnualReimbursementAmt"].fillna(0) + full["IPAnnualDeductibleAmt"].fillna(0)
    full["OP_TotalAmt"] = full["OPAnnualReimbursementAmt"].fillna(0) + full["OPAnnualDeductibleAmt"].fillna(0)

    # New claim-level features for v3
    full["Charge_Per_Day"] = np.where(
        full["Days_Admitted"] > 0,
        full["TotalClaimAmt"] / full["Days_Admitted"],
        0,
    )
    full["_claim_doy"] = full["ClaimStartDt"].dt.dayofyear
    full["_claim_dow"] = full["ClaimStartDt"].dt.dayofweek  # 0=Mon, 6=Sun
    full["_is_weekend"] = (full["_claim_dow"] >= 5).astype(int)
    full["_claim_day_of_month"] = full["ClaimStartDt"].dt.day
    full["_is_monthend"] = (full["_claim_day_of_month"] >= 26).astype(int)

    # Gender numeric
    if "Gender" in full.columns:
        full["Gender_num"] = full["Gender"].map({1: 0, 2: 1}).fillna(0).astype(int)

    # Coverage months
    for cov_col in ["NoOfMonths_PartACov", "NoOfMonths_PartBCov"]:
        if cov_col in full.columns:
            full[cov_col] = full[cov_col].fillna(0)

    log.info(f"  Features engineered. Columns: {full.shape[1]}")

    # ---- Provider-level aggregation (existing features) ----
    log.info("[5/11] Aggregating existing features to provider level...")

    target = full[["Provider", "PotentialFraud"]].drop_duplicates("Provider")
    n_providers = train_labels["Provider"].nunique()

    agg_count = full.groupby("Provider").agg(
        Beneficiary_Count=("BeneID", "nunique"),
        Claim_Count=("ClaimID", "nunique"),
        Dead_Beneficiary_Count=("is_dead", lambda x: full.loc[x.index].drop_duplicates("BeneID")["is_dead"].sum()),
    ).reset_index()

    sum_cols = chronic_cols + [
        "RenalDiseaseIndicator", "is_inpatient",
        "Has_AdmitDiag", "Has_GroupCode", "Has_Deductible",
        "Claim_GT_Admitted", "Same_Physician",
    ]
    agg_sum = full.groupby("Provider")[sum_cols].sum().reset_index()

    mean_cols = [
        "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
        "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
        "InscClaimAmtReimbursed", "DeductibleAmtPaid",
        "Age", "Days_Admitted", "Disease_Count", "Physician_Count",
        "Unique_Physicians", "Diag_Code_Count", "Proc_Code_Count",
        "Claim_Duration", "TotalClaimAmt", "IP_TotalAmt", "OP_TotalAmt",
    ]
    agg_mean = full.groupby("Provider")[mean_cols].mean().reset_index()

    agg_adv = full.groupby("Provider").agg(
        Reimburse_std=("InscClaimAmtReimbursed", "std"),
        Reimburse_max=("InscClaimAmtReimbursed", "max"),
        Reimburse_min=("InscClaimAmtReimbursed", "min"),
        Deductible_std=("DeductibleAmtPaid", "std"),
        Deductible_max=("DeductibleAmtPaid", "max"),
        ClaimDur_std=("Claim_Duration", "std"),
        ClaimDur_max=("Claim_Duration", "max"),
        DaysAdm_std=("Days_Admitted", "std"),
        DaysAdm_max=("Days_Admitted", "max"),
    ).reset_index().fillna(0)

    provider_df = (
        agg_count
        .merge(agg_sum, on="Provider")
        .merge(agg_mean, on="Provider", suffixes=("_sum", "_mean"))
        .merge(agg_adv, on="Provider")
        .merge(target, on="Provider")
    )

    assert len(provider_df) == n_providers

    # Ratio features
    provider_df["Claims_Per_Bene"] = provider_df["Claim_Count"] / provider_df["Beneficiary_Count"].replace(0, np.nan)
    provider_df["Inpatient_Ratio"] = provider_df["is_inpatient"] / provider_df["Claim_Count"].replace(0, np.nan)
    provider_df["Dead_Patient_Ratio"] = provider_df["Dead_Beneficiary_Count"] / provider_df["Beneficiary_Count"].replace(0, np.nan)
    provider_df["Reimburse_CV"] = provider_df["Reimburse_std"] / provider_df["InscClaimAmtReimbursed"].replace(0, np.nan)
    provider_df["Reimburse_Deductible_Ratio"] = provider_df["InscClaimAmtReimbursed"] / provider_df["DeductibleAmtPaid"].replace(0, np.nan)

    # V2 features
    diag_melted = full[["Provider"] + diag_cols_present].melt(id_vars="Provider", value_name="code").dropna(subset=["code"])
    diag_counts = diag_melted.groupby("Provider")["code"].apply(lambda x: x.value_counts(normalize=True))
    diag_ent = diag_counts.groupby(level=0).apply(lambda x: scipy_entropy(x.values)).rename("Diag_Entropy")
    provider_df = provider_df.merge(diag_ent.reset_index(), on="Provider", how="left")

    proc_melted = full[["Provider"] + proc_cols_present].melt(id_vars="Provider", value_name="code").dropna(subset=["code"])
    if len(proc_melted) > 0:
        proc_counts = proc_melted.groupby("Provider")["code"].apply(lambda x: x.value_counts(normalize=True))
        proc_ent = proc_counts.groupby(level=0).apply(lambda x: scipy_entropy(x.values)).rename("Proc_Entropy")
        provider_df = provider_df.merge(proc_ent.reset_index(), on="Provider", how="left")
    else:
        provider_df["Proc_Entropy"] = 0.0

    phys_melted = full[["Provider", "AttendingPhysician"]].dropna(subset=["AttendingPhysician"])
    phys_shares = phys_melted.groupby("Provider")["AttendingPhysician"].apply(
        lambda x: ((x.value_counts(normalize=True) ** 2).sum())
    ).rename("Physician_HHI")
    provider_df = provider_df.merge(phys_shares.reset_index(), on="Provider", how="left")

    inp_std = full.groupby("Provider")["is_inpatient"].std().fillna(0).rename("InpatientRatio_std")
    provider_df = provider_df.merge(inp_std.reset_index(), on="Provider", how="left")

    reimb_skew = full.groupby("Provider")["InscClaimAmtReimbursed"].apply(
        lambda x: x.skew() if len(x) >= 3 else 0.0
    ).fillna(0).rename("Reimburse_skew")
    provider_df = provider_df.merge(reimb_skew.reset_index(), on="Provider", how="left")

    claim_date_std = full.groupby("Provider")["_claim_doy"].std().fillna(0).rename("ClaimDate_std")
    provider_df = provider_df.merge(claim_date_std.reset_index(), on="Provider", how="left")

    dead_claim_rate = full.groupby("Provider")["is_dead"].mean().rename("Dead_Claim_Rate")
    provider_df = provider_df.merge(dead_claim_rate.reset_index(), on="Provider", how="left")

    if len(proc_melted) > 0:
        unique_procs = proc_melted.groupby("Provider")["code"].nunique().rename("Unique_Proc_Codes")
        provider_df = provider_df.merge(unique_procs.reset_index(), on="Provider", how="left")
    else:
        provider_df["Unique_Proc_Codes"] = 0

    existing_feat_count = len([c for c in provider_df.columns if c not in ["Provider", "PotentialFraud"]])
    log.info(f"  Existing features: {existing_feat_count}")

    # ================================================================
    # NEW FEATURES - Category A: Diagnosis Code Fractions (top 50)
    # ================================================================
    log.info("[6/11] Computing diagnosis code fraction features (top 50)...")

    all_diag = diag_melted.copy()
    top_diag_codes = all_diag["code"].value_counts().head(TOP_N_DIAG).index.tolist()
    log.info(f"  Top {TOP_N_DIAG} diagnosis codes identified")

    # For each provider, fraction of claims containing each top code
    provider_claims = full.groupby("Provider")["ClaimID"].nunique()

    for code in top_diag_codes:
        code_claims = all_diag[all_diag["code"] == code].groupby("Provider")["code"].count().rename(f"DiagCode_{code}_count")
        provider_df = provider_df.merge(code_claims.reset_index(), on="Provider", how="left")
        provider_df[f"DiagCode_{code}_frac"] = provider_df[f"DiagCode_{code}_count"].fillna(0) / provider_df["Claim_Count"].replace(0, 1)
        provider_df.drop(f"DiagCode_{code}_count", axis=1, inplace=True)

    log.info(f"  Added {TOP_N_DIAG} diagnosis code fraction features")

    # ================================================================
    # NEW FEATURES - Category B: Procedure Code Fractions (top 20)
    # ================================================================
    log.info("[6/11] Computing procedure code fraction features (top 20)...")

    if len(proc_melted) > 0:
        top_proc_codes = proc_melted["code"].value_counts().head(TOP_N_PROC).index.tolist()
        for code in top_proc_codes:
            code_claims = proc_melted[proc_melted["code"] == code].groupby("Provider")["code"].count().rename(f"ProcCode_{code}_count")
            provider_df = provider_df.merge(code_claims.reset_index(), on="Provider", how="left")
            provider_df[f"ProcCode_{code}_frac"] = provider_df[f"ProcCode_{code}_count"].fillna(0) / provider_df["Claim_Count"].replace(0, 1)
            provider_df.drop(f"ProcCode_{code}_count", axis=1, inplace=True)
        log.info(f"  Added {TOP_N_PROC} procedure code fraction features")
    else:
        log.warning("  No procedure codes found, skipping")

    # ================================================================
    # NEW FEATURES - Category C: Physician Behavior (6 features)
    # ================================================================
    log.info("[7/11] Computing physician behavior features...")

    # Compute per-physician average reimbursement across ALL claims
    attending_stats = full.groupby("AttendingPhysician")["InscClaimAmtReimbursed"].agg(["mean", "count"]).rename(
        columns={"mean": "phys_avg_reimb", "count": "phys_claim_count"}
    )

    # Merge back to claims, then aggregate to provider
    full_with_phys = full.merge(attending_stats, left_on="AttendingPhysician", right_index=True, how="left")

    phys_prov = full_with_phys.groupby("Provider").agg(
        Phys_AvgReimb_mean=("phys_avg_reimb", "mean"),
        Phys_AvgReimb_std=("phys_avg_reimb", "std"),
        Phys_ClaimCount_mean=("phys_claim_count", "mean"),
    ).fillna(0).reset_index()
    provider_df = provider_df.merge(phys_prov, on="Provider", how="left")

    # Operating physician stats
    if "OperatingPhysician" in full.columns:
        op_phys_stats = full.groupby("OperatingPhysician")["InscClaimAmtReimbursed"].mean().rename("op_phys_avg_reimb")
        full_with_op = full.merge(op_phys_stats, left_on="OperatingPhysician", right_index=True, how="left")
        op_prov = full_with_op.groupby("Provider").agg(
            OpPhys_AvgReimb_mean=("op_phys_avg_reimb", "mean"),
            OpPhys_AvgReimb_std=("op_phys_avg_reimb", "std"),
        ).fillna(0).reset_index()
        provider_df = provider_df.merge(op_prov, on="Provider", how="left")

    # Physician role overlap: fraction of claims where attending = operating
    if "OperatingPhysician" in full.columns:
        full["_phys_overlap"] = (full["AttendingPhysician"] == full["OperatingPhysician"]).astype(int)
        phys_overlap = full.groupby("Provider")["_phys_overlap"].mean().rename("Phys_Role_Overlap")
        provider_df = provider_df.merge(phys_overlap.reset_index(), on="Provider", how="left")

    log.info(f"  Added physician behavior features")

    # ================================================================
    # NEW FEATURES - Category D: Geographic Features (4 features)
    # ================================================================
    log.info("[7/11] Computing geographic features...")

    if "State" in full.columns:
        state_agg = full.groupby("Provider").agg(
            Unique_States=("State", "nunique"),
        ).reset_index()

        # Dominant state fraction
        def dominant_state_frac(x):
            counts = x.value_counts(normalize=True)
            return counts.iloc[0] if len(counts) > 0 else 1.0

        dom_state = full.groupby("Provider")["State"].apply(dominant_state_frac).rename("Dominant_State_Frac")
        state_agg = state_agg.merge(dom_state.reset_index(), on="Provider")

        # State entropy
        state_ent = full.groupby("Provider")["State"].apply(
            lambda x: scipy_entropy(x.value_counts(normalize=True).values)
        ).rename("State_Entropy")
        state_agg = state_agg.merge(state_ent.reset_index(), on="Provider")

        provider_df = provider_df.merge(state_agg, on="Provider", how="left")

    if "County" in full.columns:
        county_count = full.groupby("Provider")["County"].nunique().rename("Unique_Counties")
        provider_df = provider_df.merge(county_count.reset_index(), on="Provider", how="left")

    log.info(f"  Added geographic features")

    # ================================================================
    # NEW FEATURES - Category E: Beneficiary Network (5 features)
    # ================================================================
    log.info("[8/11] Computing beneficiary network features...")

    bp = full[["Provider", "BeneID"]].drop_duplicates()

    # How many providers does each beneficiary visit?
    bene_provider_counts = bp.groupby("BeneID")["Provider"].nunique().rename("_n_providers")
    bp = bp.merge(bene_provider_counts.reset_index(), on="BeneID")

    # Per provider aggregations
    net_features = bp.groupby("Provider").agg(
        Shared_Bene_Count=("_n_providers", lambda x: (x > 1).sum()),
        Bene_Exclusivity=("_n_providers", lambda x: (x == 1).mean()),
        Avg_Providers_Per_Bene=("_n_providers", "mean"),
        Max_Providers_Per_Bene=("_n_providers", "max"),
    ).reset_index()

    # Provider network degree (how many other providers share at least 1 bene)
    shared_benes = bp[bp["_n_providers"] > 1][["Provider", "BeneID"]]
    if len(shared_benes) > 0:
        shared_pairs = shared_benes.merge(shared_benes, on="BeneID", suffixes=("_1", "_2"))
        shared_pairs = shared_pairs[shared_pairs["Provider_1"] != shared_pairs["Provider_2"]]
        net_degree = shared_pairs.groupby("Provider_1")["Provider_2"].nunique().rename("Provider_Network_Degree")
        net_features = net_features.merge(
            net_degree.reset_index().rename(columns={"Provider_1": "Provider"}),
            on="Provider", how="left"
        )
    else:
        net_features["Provider_Network_Degree"] = 0

    provider_df = provider_df.merge(net_features, on="Provider", how="left")
    log.info(f"  Added beneficiary network features")

    # ================================================================
    # NEW FEATURES - Category F: Claim-Type Separated Stats (8 features)
    # ================================================================
    log.info("[8/11] Computing claim-type separated stats...")

    ip_claims = full[full["is_inpatient"] == 1]
    op_claims = full[full["is_inpatient"] == 0]

    if len(ip_claims) > 0:
        ip_stats = ip_claims.groupby("Provider").agg(
            IP_Reimburse_mean=("InscClaimAmtReimbursed", "mean"),
            IP_Reimburse_std=("InscClaimAmtReimbursed", "std"),
            IP_ClaimDur_mean=("Claim_Duration", "mean"),
            IP_ChargePerDay_mean=("Charge_Per_Day", "mean"),
        ).fillna(0).reset_index()
        provider_df = provider_df.merge(ip_stats, on="Provider", how="left")

    if len(op_claims) > 0:
        op_stats = op_claims.groupby("Provider").agg(
            OP_Reimburse_mean=("InscClaimAmtReimbursed", "mean"),
            OP_Reimburse_std=("InscClaimAmtReimbursed", "std"),
            OP_ClaimDur_mean=("Claim_Duration", "mean"),
        ).fillna(0).reset_index()
        provider_df = provider_df.merge(op_stats, on="Provider", how="left")

    # IP/OP reimbursement ratio
    if "IP_Reimburse_mean" in provider_df.columns and "OP_Reimburse_mean" in provider_df.columns:
        provider_df["IP_OP_Reimburse_Ratio"] = (
            provider_df["IP_Reimburse_mean"].fillna(0) /
            provider_df["OP_Reimburse_mean"].fillna(0).replace(0, np.nan)
        )

    log.info(f"  Added claim-type separated stats")

    # ================================================================
    # NEW FEATURES - Category G: Z-Score Features (8 features)
    # ================================================================
    log.info("[9/11] Computing z-score features...")

    zscore_cols = [
        ("InscClaimAmtReimbursed", "Reimburse_mean_zscore"),
        ("Claim_Count", "ClaimCount_zscore"),
        ("Beneficiary_Count", "BeneCount_zscore"),
        ("Claim_Duration", "ClaimDur_mean_zscore"),
        ("Dead_Patient_Ratio", "DeadRatio_zscore"),
        ("Inpatient_Ratio", "InpatientRatio_zscore"),
        ("Claims_Per_Bene", "ClaimPerBene_zscore"),
        ("Reimburse_CV", "ReimburseCV_zscore"),
    ]

    for src_col, new_col in zscore_cols:
        if src_col in provider_df.columns:
            col_mean = provider_df[src_col].mean()
            col_std = provider_df[src_col].std()
            if col_std > 0:
                provider_df[new_col] = (provider_df[src_col] - col_mean) / col_std
            else:
                provider_df[new_col] = 0.0

    log.info(f"  Added z-score features")

    # ================================================================
    # NEW FEATURES - Category H: Percentile Rank Features (6 features)
    # ================================================================
    log.info("[9/11] Computing percentile rank features...")

    pctile_cols = [
        ("InscClaimAmtReimbursed", "Reimburse_mean_pctile"),
        ("Claim_Count", "ClaimCount_pctile"),
        ("Beneficiary_Count", "BeneCount_pctile"),
        ("Dead_Patient_Ratio", "DeadRatio_pctile"),
        ("Claim_Duration", "ClaimDur_mean_pctile"),
        ("Reimburse_max", "Reimburse_max_pctile"),
    ]

    for src_col, new_col in pctile_cols:
        if src_col in provider_df.columns:
            provider_df[new_col] = provider_df[src_col].rank(pct=True)

    log.info(f"  Added percentile rank features")

    # ================================================================
    # NEW FEATURES - Category I: Distributional Tail Features (6 features)
    # ================================================================
    log.info("[10/11] Computing distributional tail features...")

    reimb_tails = full.groupby("Provider")["InscClaimAmtReimbursed"].agg(
        Reimburse_kurtosis=lambda x: scipy_kurtosis(x, nan_policy="omit") if len(x) >= 4 else 0.0,
        Reimburse_p95=lambda x: np.nanpercentile(x, 95),
        Reimburse_p99=lambda x: np.nanpercentile(x, 99),
        Reimburse_iqr=lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25),
    ).reset_index()
    provider_df = provider_df.merge(reimb_tails, on="Provider", how="left")

    claimdur_kurt = full.groupby("Provider")["Claim_Duration"].apply(
        lambda x: scipy_kurtosis(x, nan_policy="omit") if len(x) >= 4 else 0.0
    ).rename("ClaimDur_kurtosis")
    provider_df = provider_df.merge(claimdur_kurt.reset_index(), on="Provider", how="left")

    daysadm_kurt = full.groupby("Provider")["Days_Admitted"].apply(
        lambda x: scipy_kurtosis(x, nan_policy="omit") if len(x) >= 4 else 0.0
    ).rename("DaysAdm_kurtosis")
    provider_df = provider_df.merge(daysadm_kurt.reset_index(), on="Provider", how="left")

    log.info(f"  Added distributional tail features")

    # ================================================================
    # NEW FEATURES - Category J: Interaction Features (6 features)
    # ================================================================
    log.info("[10/11] Computing interaction features...")

    # Top SHAP features interactions
    if "Has_Deductible" in provider_df.columns and "Reimburse_max" in provider_df.columns:
        provider_df["HasDeductible_x_ReimburseMax"] = provider_df["Has_Deductible"] * provider_df["Reimburse_max"]
    if "Dead_Patient_Ratio" in provider_df.columns and "Claims_Per_Bene" in provider_df.columns:
        provider_df["DeadRatio_x_ClaimPerBene"] = provider_df["Dead_Patient_Ratio"] * provider_df["Claims_Per_Bene"]
    if "Reimburse_CV" in provider_df.columns and "ClaimDur_std" in provider_df.columns:
        provider_df["ReimburseCV_x_ClaimDurStd"] = provider_df["Reimburse_CV"] * provider_df["ClaimDur_std"]
    if "Inpatient_Ratio" in provider_df.columns and "Reimburse_max" in provider_df.columns:
        provider_df["InpatientRatio_x_ReimburseMax"] = provider_df["Inpatient_Ratio"] * provider_df["Reimburse_max"]
    if "Claims_Per_Bene" in provider_df.columns and "Reimburse_std" in provider_df.columns:
        provider_df["ClaimPerBene_x_ReimburseStd"] = provider_df["Claims_Per_Bene"] * provider_df["Reimburse_std"]
    if "Physician_HHI" in provider_df.columns and "Diag_Entropy" in provider_df.columns:
        provider_df["PhysHHI_x_DiagEntropy"] = provider_df["Physician_HHI"] * provider_df["Diag_Entropy"]

    log.info(f"  Added interaction features")

    # ================================================================
    # NEW FEATURES - Category K: Anomaly Score (1 feature)
    # ================================================================
    log.info("[11/11] Computing IsolationForest anomaly score...")

    feature_cols = [c for c in provider_df.columns if c not in ["Provider", "PotentialFraud"]]
    X_temp = provider_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    provider_df["IsoForest_Score"] = iso.fit_predict(X_temp)
    provider_df["IsoForest_Anomaly"] = iso.decision_function(X_temp)

    log.info(f"  Added IsolationForest anomaly features")

    # ================================================================
    # Temporal pattern features (5 features)
    # ================================================================
    log.info("  Computing temporal pattern features...")

    temporal = full.groupby("Provider").agg(
        Weekend_Claim_Ratio=("_is_weekend", "mean"),
        MonthEnd_Claim_Ratio=("_is_monthend", "mean"),
        Claim_DOY_mean=("_claim_doy", "mean"),
    ).fillna(0).reset_index()
    provider_df = provider_df.merge(temporal, on="Provider", how="left")

    # Claim gap features (time between consecutive claims per provider)
    claim_dates = full.sort_values(["Provider", "ClaimStartDt"])[["Provider", "ClaimStartDt"]].dropna()
    claim_dates["_gap"] = claim_dates.groupby("Provider")["ClaimStartDt"].diff().dt.days
    gap_stats = claim_dates.groupby("Provider")["_gap"].agg(
        Claim_Gap_Mean="mean",
        Claim_Gap_Std="std",
    ).fillna(0).reset_index()
    provider_df = provider_df.merge(gap_stats, on="Provider", how="left")

    # Beneficiary demographics features
    log.info("  Computing beneficiary demographics features...")
    bene_demo = full.drop_duplicates(["Provider", "BeneID"])
    bene_agg = bene_demo.groupby("Provider").agg(
        Bene_Age_Std=("Age", "std"),
        Bene_Chronic_Mean=("Disease_Count", "mean"),
        Bene_Chronic_Std=("Disease_Count", "std"),
    ).fillna(0).reset_index()
    provider_df = provider_df.merge(bene_agg, on="Provider", how="left")

    if "Gender_num" in full.columns:
        gender_ratio = bene_demo.groupby("Provider")["Gender_num"].mean().rename("Bene_Gender_Ratio")
        provider_df = provider_df.merge(gender_ratio.reset_index(), on="Provider", how="left")

    if "NoOfMonths_PartACov" in full.columns:
        cov_mean = bene_demo.groupby("Provider")["NoOfMonths_PartACov"].mean().rename("Bene_CoverageA_Mean")
        provider_df = provider_df.merge(cov_mean.reset_index(), on="Provider", how="left")

    # ================================================================
    # Final cleanup
    # ================================================================
    log.info("  Final cleanup...")

    # Handle inf/nan
    nan_count = provider_df.isna().sum().sum()
    inf_count = np.isinf(provider_df.select_dtypes(include=[np.number])).sum().sum()
    log.info(f"  Filling {nan_count} NaN and {inf_count} inf values with 0")
    provider_df.fillna(0, inplace=True)
    provider_df.replace([np.inf, -np.inf], 0, inplace=True)

    # ---- Save ----
    out_path = f"{args.out_dir}/{args.out_filename}"
    provider_df.to_csv(out_path, index=False)

    feature_cols = [c for c in provider_df.columns if c not in ["Provider", "PotentialFraud"]]
    fraud_dist = provider_df["PotentialFraud"].value_counts()

    log.info(f"\n  === SUMMARY ===")
    log.info(f"  Output: {out_path}")
    log.info(f"  Providers: {len(provider_df)}")
    log.info(f"  Features: {len(feature_cols)}")
    log.info(f"  Fraud: {fraud_dist.get(1, 0)} ({fraud_dist.get(1, 0)/len(provider_df)*100:.1f}%)")
    log.info(f"  Non-fraud: {fraud_dist.get(0, 0)} ({fraud_dist.get(0, 0)/len(provider_df)*100:.1f}%)")
    log.info(f"  Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
