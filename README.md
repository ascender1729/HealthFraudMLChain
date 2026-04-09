# HealthFraudMLChain

**Enhancing Healthcare Insurance Fraud Detection and Prevention with a Machine Learning and Blockchain-Based Approach**

M.Sc. Dissertation, Department of Mathematics, National Institute of Technology Patna, 2024

**Live Demo:** [healthcare-fraud-detection.pages.dev](https://healthcare-fraud-detection.pages.dev/)

## Overview

This repository contains the complete research pipeline for healthcare insurance fraud detection using an ensemble of Optuna-tuned machine learning classifiers with blockchain-secured audit trails and ECIES encryption. The system aggregates 558,211 Medicare claims into 5,410 provider-level records with 190 engineered features, achieving F1 = 0.7345 with zero data leakage under 10-fold stratified cross-validation.

## Key Results

| Metric | Value | 95% Bootstrap CI |
|--------|-------|-------------------|
| F1 Score | 0.7345 | [0.7118, 0.7715] |
| Precision | 73.7% | [69.8%, 77.5%] |
| Recall | 74.7% | [70.9%, 78.4%] |
| MCC | 0.7151 | [0.6818, 0.7474] |
| ROC-AUC | 0.9587 | [0.9477, 0.9686] |
| PR-AUC | 0.8114 | [0.7801, 0.8406] |

Weighted ensemble: LightGBM (39.6%), CatBoost (32.0%), XGBoost (28.2%) at fixed threshold t = 0.444.

**Statistical significance:** Friedman test p = 0.00089; all pairwise Wilcoxon comparisons significant after Holm correction.

## Features

- **ML Pipeline:** Five gradient-boosted classifiers (XGBoost, LightGBM, CatBoost, GradientBoosting, RandomForest) with Optuna TPE hyperparameter optimization (60 trials per model)
- **Leakage-Free Evaluation:** All feature engineering (z-scores, percentiles, IsolationForest, physician statistics) computed strictly within each CV fold
- **Explainability:** Dual SHAP (global) + LIME (local) interpretation identifying deductible patterns, reimbursement anomalies, and claim duration signals as top fraud indicators
- **Blockchain Audit:** Custom SHA-256 blockchain with Merkle tree integrity verification, PoW (difficulty 2), and SQLite persistence (110 blocks, 5,411 records)
- **ECIES Encryption:** secp256k1 + HKDF-SHA256 + AES-256-GCM field-level encryption for provider PII protection
- **Statistical Rigor:** Friedman + Wilcoxon (Holm-corrected) non-parametric tests, Cohen's d effect sizes, 2000-iteration bootstrap confidence intervals
- **Web Prototype:** Interactive Next.js static site deployed on Cloudflare Pages with browser-side ECIES demo

## Repository Structure

```
corrected_pipeline/
  01_preprocess.py              # Data preprocessing and feature engineering
  02_train_evaluate.py          # Initial model training
  04_optuna_tuning.py           # Optuna hyperparameter optimization
  15_definitive_final.py        # Definitive leakage-free evaluation (main script)
  16_regenerate_all_figures.py  # Figure generation
  17_multi_node_demo.py         # Multi-node blockchain consensus demo
  export_static_data.py         # Export data for web frontend
  blockchain/
    chain.py                    # SHA-256 blockchain with Merkle trees
    ecies_cipher.py             # ECIES encryption (secp256k1 + AES-256-GCM)
  results/
    definitive_final_results.json  # All model metrics and statistical tests
    figures/                       # 15 research figures (PNG + PDF)
  thesis_latex/
    main.tex                    # Complete thesis LaTeX source
    chapters/                   # Ch1-Ch5 chapter files
Dataset/                        # Kaggle rohitrox Medicare claims data
thesis.pdf                      # Compiled thesis PDF
```

## Quick Start

```bash
# Clone
git clone https://github.com/ascender1729/HealthFraudMLChain.git
cd HealthFraudMLChain/corrected_pipeline

# Install dependencies
pip install -r requirements.txt

# Run the definitive evaluation (requires Dataset/ with Medicare CSVs)
python 15_definitive_final.py

# Run blockchain demo
python 06_blockchain_demo.py

# Multi-node consensus validation
python 17_multi_node_demo.py
```

## Web Prototype

The interactive demo is deployed at [healthcare-fraud-detection.pages.dev](https://healthcare-fraud-detection.pages.dev/) with:

- **Provider Analysis** - Look up any provider's fraud risk with SHAP/LIME explanations
- **Research Results** - All 15 figures, statistical tests, bootstrap CIs
- **Blockchain Explorer** - Inspect the 110-block audit chain
- **ECIES Demo** - Live browser-side encrypt/decrypt using Web Crypto API

## Tools and Technologies

| Category | Technologies |
|----------|-------------|
| ML Pipeline | Python, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, LIME |
| Blockchain | SHA-256, Merkle Trees, Proof-of-Work, SQLite |
| Encryption | ECIES (secp256k1 + HKDF-SHA256 + AES-256-GCM) |
| Statistics | Friedman test, Wilcoxon signed-rank, Holm correction, Bootstrap CI |
| Web | Next.js 14, TypeScript, Tailwind CSS, Magic UI, Cloudflare Pages |
| Data | Kaggle rohitrox Medicare claims (558K claims, 5,410 providers) |

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](LICENSE.md).

## Acknowledgments

- Dr. Rajesh Kumar Sinha for guidance and supervision
- National Institute of Technology Patna for research facilities

---

## Citation

If you use this project in your research, please cite:

```bibtex
@thesis{dubasi2024_healthfraudmlchain,
  author  = {Dubasi, Pavan Kumar},
  title   = {Enhancing Healthcare Insurance Fraud Detection and Prevention
             with a Machine Learning and Blockchain-Based Approach},
  school  = {National Institute of Technology Patna},
  year    = {2024},
  type    = {Integrated M.Sc. Mathematics Dissertation},
  advisor = {Dr. Rajesh Kumar Sinha},
  url     = {https://github.com/ascender1729/HealthFraudMLChain}
}
```

## Author

**Dubasi Pavan Kumar**
- Website: [dubasipavankumar.com](https://dubasipavankumar.com)
- LinkedIn: [in/im-pavankumar](https://linkedin.com/in/im-pavankumar)
- ORCID: [0009-0006-1060-4598](https://orcid.org/0009-0006-1060-4598)
