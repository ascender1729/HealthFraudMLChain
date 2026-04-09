"""
Phase 1: Data Loading, Merging, Feature Engineering, Provider-Level Aggregation
Dataset: Kaggle rohitrox Healthcare Provider Fraud Detection
All methods available before June 2024.

Fixes applied:
- Dead_Patient_Ratio uses Beneficiary_Count denominator (not Claim_Count)
- Configurable paths via argparse
- Error handling on all file I/O
- Assertions for data validation
- Age calculated per-claim using ClaimStartDt
- Proper inf/nan handling with documentation
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

np.random.seed(42)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Preprocessing")
    p.add_argument("--data-dir", default="/home/ubuntu/data", help="Directory with Kaggle CSVs")
    p.add_argument("--out-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline", help="Output directory")
    p.add_argument("--out-filename", default="provider_features.csv",
                   help="Output CSV filename (e.g. provider_features_v2.csv for 60-feature version)")
    return p.parse_args()


def load_csv(path, name):
    try:
        df = pd.read_csv(path)
        log.info(f"  Loaded {name}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        log.error(f"  File not found: {path}")
        log.error(f"  Download dataset: kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis")
        sys.exit(1)


def main():
    args = parse_args()
    start = time.time()

    log.info("=" * 60)
    log.info("PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING")
    log.info("=" * 60)

    # ---- Load datasets ----
    log.info("[1/6] Loading datasets...")
    D = args.data_dir
    train_labels = load_csv(f"{D}/Train-1542865627584.csv", "Provider labels")
    beneficiary = load_csv(f"{D}/Train_Beneficiarydata-1542865627584.csv", "Beneficiary")
    inpatient = load_csv(f"{D}/Train_Inpatientdata-1542865627584.csv", "Inpatient")
    outpatient = load_csv(f"{D}/Train_Outpatientdata-1542865627584.csv", "Outpatient")

    fraud_dist = train_labels["PotentialFraud"].value_counts().to_dict()
    log.info(f"  Fraud distribution: {fraud_dist}")

    # ---- Merge Inpatient + Outpatient ----
    log.info("[2/6] Merging inpatient + outpatient claims...")
    common_cols = [c for c in inpatient.columns if c in outpatient.columns]
    claims = outpatient.merge(inpatient, on=common_cols, how="outer", indicator="claim_type")
    claims["is_inpatient"] = (claims["claim_type"] == "right_only").astype(int)
    claims.drop("claim_type", axis=1, inplace=True)
    log.info(f"  Combined claims: {claims.shape[0]}")

    # ---- Merge with beneficiary ----
    log.info("[3/6] Joining with beneficiary data...")
    claims_benef = beneficiary.merge(claims, on="BeneID")
    log.info(f"  Claims+Beneficiary: {claims_benef.shape[0]}")

    # ---- Merge with provider labels ----
    full = train_labels.merge(claims_benef, on="Provider")
    full["PotentialFraud"] = full["PotentialFraud"].map({"Yes": 1, "No": 0})
    assert full["PotentialFraud"].isna().sum() == 0, "NaN in target after mapping"
    log.info(f"  Full dataset: {full.shape[0]} rows, {full.shape[1]} cols")

    # ---- Feature engineering (claim-level) ----
    log.info("[4/6] Engineering claim-level features...")

    # Dates
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if col in full.columns:
            full[col] = pd.to_datetime(full[col], errors="coerce")
    full["DOB"] = pd.to_datetime(full["DOB"], errors="coerce")
    full["DOD"] = pd.to_datetime(full["DOD"], errors="coerce")

    # Age at time of claim (more accurate than hardcoded 2009)
    full["Age"] = ((full["ClaimStartDt"] - full["DOB"]).dt.days / 365.25).fillna(0).astype(int)

    # Deceased
    full["is_dead"] = full["DOD"].notna().astype(int)

    # Claim duration
    full["Claim_Duration"] = (full["ClaimEndDt"] - full["ClaimStartDt"]).dt.days.fillna(0).astype(int)

    # Days admitted
    full["Days_Admitted"] = (full["DischargeDt"] - full["AdmissionDt"]).dt.days.fillna(0).astype(int)

    # Claim longer than stay
    full["Claim_GT_Admitted"] = (full["Claim_Duration"] > full["Days_Admitted"]).astype(int)

    # Chronic conditions (recode 2 to 0)
    chronic_cols = [c for c in full.columns if c.startswith("ChronicCond_")]
    for col in chronic_cols:
        full[col] = full[col].replace(2, 0)
    full["RenalDiseaseIndicator"] = full["RenalDiseaseIndicator"].replace({"0": 0, "Y": 1, 0: 0}).astype(int)
    full["Disease_Count"] = full[chronic_cols].sum(axis=1)

    # Physician features (vectorized, no slow apply)
    phys_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    full["Physician_Count"] = full[phys_cols].notna().sum(axis=1)
    full["Unique_Physicians"] = full[phys_cols].nunique(axis=1)
    full["Same_Physician"] = ((full["Unique_Physicians"] == 1) & (full["Physician_Count"] > 1)).astype(int)

    # Diagnosis/procedure code counts
    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    full["Diag_Code_Count"] = full[[c for c in diag_cols if c in full.columns]].notna().sum(axis=1)
    full["Proc_Code_Count"] = full[[c for c in proc_cols if c in full.columns]].notna().sum(axis=1)
    full["Has_AdmitDiag"] = full["ClmAdmitDiagnosisCode"].notna().astype(int) if "ClmAdmitDiagnosisCode" in full.columns else 0
    full["Has_GroupCode"] = full["DiagnosisGroupCode"].notna().astype(int) if "DiagnosisGroupCode" in full.columns else 0

    # Financial features
    full["DeductibleAmtPaid"] = full["DeductibleAmtPaid"].fillna(0)
    full["Has_Deductible"] = (full["DeductibleAmtPaid"] > 0).astype(int)
    full["TotalClaimAmt"] = full["InscClaimAmtReimbursed"].fillna(0) + full["DeductibleAmtPaid"]
    full["IP_TotalAmt"] = full["IPAnnualReimbursementAmt"].fillna(0) + full["IPAnnualDeductibleAmt"].fillna(0)
    full["OP_TotalAmt"] = full["OPAnnualReimbursementAmt"].fillna(0) + full["OPAnnualDeductibleAmt"].fillna(0)

    log.info(f"  Features engineered. Columns: {full.shape[1]}")

    # ---- Provider-level aggregation ----
    log.info("[5/6] Aggregating to provider level...")

    target = full[["Provider", "PotentialFraud"]].drop_duplicates("Provider")
    n_providers = train_labels["Provider"].nunique()

    # Count features
    agg_count = full.groupby("Provider").agg(
        Beneficiary_Count=("BeneID", "nunique"),
        Claim_Count=("ClaimID", "nunique"),
        Dead_Beneficiary_Count=("is_dead", lambda x: full.loc[x.index].drop_duplicates("BeneID")["is_dead"].sum()),
    ).reset_index()

    # Sum features
    sum_cols = chronic_cols + [
        "RenalDiseaseIndicator", "is_inpatient",
        "Has_AdmitDiag", "Has_GroupCode", "Has_Deductible",
        "Claim_GT_Admitted", "Same_Physician",
    ]
    agg_sum = full.groupby("Provider")[sum_cols].sum().reset_index()

    # Mean features
    mean_cols = [
        "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
        "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
        "InscClaimAmtReimbursed", "DeductibleAmtPaid",
        "Age", "Days_Admitted", "Disease_Count", "Physician_Count",
        "Unique_Physicians", "Diag_Code_Count", "Proc_Code_Count",
        "Claim_Duration", "TotalClaimAmt", "IP_TotalAmt", "OP_TotalAmt",
    ]
    agg_mean = full.groupby("Provider")[mean_cols].mean().reset_index()

    # Std/Max features
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

    # Merge all
    provider_df = (
        agg_count
        .merge(agg_sum, on="Provider")
        .merge(agg_mean, on="Provider", suffixes=("_sum", "_mean"))
        .merge(agg_adv, on="Provider")
        .merge(target, on="Provider")
    )

    assert len(provider_df) == n_providers, f"Provider count mismatch: {len(provider_df)} vs {n_providers}"
    assert provider_df["PotentialFraud"].isna().sum() == 0, "NaN in target after aggregation"

    # Ratio features (FIX: Dead_Patient_Ratio uses Beneficiary_Count)
    provider_df["Claims_Per_Bene"] = (
        provider_df["Claim_Count"] / provider_df["Beneficiary_Count"].replace(0, np.nan)
    )
    provider_df["Inpatient_Ratio"] = (
        provider_df["is_inpatient"] / provider_df["Claim_Count"].replace(0, np.nan)
    )
    # FIX: Use Dead_Beneficiary_Count / Beneficiary_Count (not claim-level)
    provider_df["Dead_Patient_Ratio"] = (
        provider_df["Dead_Beneficiary_Count"] / provider_df["Beneficiary_Count"].replace(0, np.nan)
    )
    provider_df["Reimburse_CV"] = (
        provider_df["Reimburse_std"] / provider_df["InscClaimAmtReimbursed"].replace(0, np.nan)
    )
    provider_df["Reimburse_Deductible_Ratio"] = (
        provider_df["InscClaimAmtReimbursed"] / provider_df["DeductibleAmtPaid"].replace(0, np.nan)
    )

    # ---- 8 New Features (v2) ----
    log.info("  Computing 8 additional provider-level features...")

    # 1. Diag_Entropy: Shannon entropy of diagnosis code distribution per provider
    diag_cols_present = [c for c in diag_cols if c in full.columns]
    diag_melted = full[["Provider"] + diag_cols_present].melt(id_vars="Provider", value_name="code").dropna(subset=["code"])
    diag_counts = diag_melted.groupby("Provider")["code"].apply(lambda x: x.value_counts(normalize=True))
    diag_ent = diag_counts.groupby(level=0).apply(lambda x: scipy_entropy(x.values)).rename("Diag_Entropy")
    provider_df = provider_df.merge(diag_ent.reset_index().rename(columns={"Provider": "Provider"}),
                                     on="Provider", how="left")

    # 2. Proc_Entropy: Shannon entropy of procedure code distribution per provider
    proc_cols_present = [c for c in proc_cols if c in full.columns]
    proc_melted = full[["Provider"] + proc_cols_present].melt(id_vars="Provider", value_name="code").dropna(subset=["code"])
    if len(proc_melted) > 0:
        proc_counts = proc_melted.groupby("Provider")["code"].apply(lambda x: x.value_counts(normalize=True))
        proc_ent = proc_counts.groupby(level=0).apply(lambda x: scipy_entropy(x.values)).rename("Proc_Entropy")
        provider_df = provider_df.merge(proc_ent.reset_index().rename(columns={"Provider": "Provider"}),
                                         on="Provider", how="left")
    else:
        provider_df["Proc_Entropy"] = 0.0

    # 3. Physician_HHI: Herfindahl-Hirschman Index of physician concentration per provider
    phys_melted = full[["Provider", "AttendingPhysician"]].dropna(subset=["AttendingPhysician"])
    phys_shares = phys_melted.groupby("Provider")["AttendingPhysician"].apply(
        lambda x: ((x.value_counts(normalize=True) ** 2).sum())
    ).rename("Physician_HHI")
    provider_df = provider_df.merge(phys_shares.reset_index(), on="Provider", how="left")

    # 4. InpatientRatio_std: Std of is_inpatient per provider (rigidity indicator)
    inp_std = full.groupby("Provider")["is_inpatient"].std().fillna(0).rename("InpatientRatio_std")
    provider_df = provider_df.merge(inp_std.reset_index(), on="Provider", how="left")

    # 5. Reimburse_skew: Skewness of reimbursement distribution per provider
    reimb_skew = full.groupby("Provider")["InscClaimAmtReimbursed"].apply(
        lambda x: x.skew() if len(x) >= 3 else 0.0
    ).fillna(0).rename("Reimburse_skew")
    provider_df = provider_df.merge(reimb_skew.reset_index(), on="Provider", how="left")

    # 6. ClaimDate_std: Std of claim submission day-of-year per provider
    full["_claim_doy"] = full["ClaimStartDt"].dt.dayofyear
    claim_date_std = full.groupby("Provider")["_claim_doy"].std().fillna(0).rename("ClaimDate_std")
    provider_df = provider_df.merge(claim_date_std.reset_index(), on="Provider", how="left")

    # 7. Dead_Claim_Rate: Fraction of claims involving a deceased beneficiary
    dead_claim_rate = full.groupby("Provider")["is_dead"].mean().rename("Dead_Claim_Rate")
    provider_df = provider_df.merge(dead_claim_rate.reset_index(), on="Provider", how="left")

    # 8. Unique_Proc_Codes: Count of distinct procedure codes per provider
    if len(proc_melted) > 0:
        unique_procs = proc_melted.groupby("Provider")["code"].nunique().rename("Unique_Proc_Codes")
        provider_df = provider_df.merge(unique_procs.reset_index(), on="Provider", how="left")
    else:
        provider_df["Unique_Proc_Codes"] = 0

    log.info(f"  Added 8 new features. Total columns: {provider_df.shape[1]}")

    # Handle inf/nan from division (fill with 0, document this choice)
    nan_count = provider_df.isna().sum().sum()
    inf_count = np.isinf(provider_df.select_dtypes(include=[np.number])).sum().sum()
    log.info(f"  Filling {nan_count} NaN and {inf_count} inf values with 0")
    provider_df.fillna(0, inplace=True)
    provider_df.replace([np.inf, -np.inf], 0, inplace=True)

    # ---- Save ----
    log.info("[6/6] Saving...")
    out_path = f"{args.out_dir}/{args.out_filename}"
    provider_df.to_csv(out_path, index=False)

    feature_cols = [c for c in provider_df.columns if c not in ["Provider", "PotentialFraud"]]
    fraud_dist = provider_df["PotentialFraud"].value_counts()

    log.info(f"  Providers: {len(provider_df)}")
    log.info(f"  Features: {len(feature_cols)}")
    log.info(f"  Fraud: {fraud_dist.get(1, 0)} ({fraud_dist.get(1, 0)/len(provider_df)*100:.1f}%)")
    log.info(f"  Non-fraud: {fraud_dist.get(0, 0)} ({fraud_dist.get(0, 0)/len(provider_df)*100:.1f}%)")
    log.info(f"  Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
