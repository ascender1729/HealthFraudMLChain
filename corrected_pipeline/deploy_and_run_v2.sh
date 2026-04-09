#!/bin/bash
# Deploy improved pipeline to Lambda Cloud and run enhanced evaluation
# Fixes SMOTE ensemble bug, adds new models, adds 8 new features
# Usage: bash deploy_and_run_v2.sh <lambda-ip>

set -e

IP="${1:-$(cat /tmp/lambda_ip.txt 2>/dev/null)}"
if [ -z "$IP" ]; then
    echo "Usage: bash deploy_and_run_v2.sh <lambda-cloud-ip>"
    exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud"

REMOTE_DIR="~/HealthFraudMLChain/corrected_pipeline"
LOCAL_DIR="corrected_pipeline"

echo "=========================================="
echo "  Improved Pipeline Deployment (v2)"
echo "  Target: $IP"
echo "=========================================="

# 1. Upload all pipeline files
echo ""
echo "[1/7] Uploading pipeline files..."
$SSH "mkdir -p $REMOTE_DIR/blockchain $REMOTE_DIR/results/figures $REMOTE_DIR/app/templates"

# Core scripts
$SCP $LOCAL_DIR/01_preprocess.py ubuntu@$IP:$REMOTE_DIR/
$SCP $LOCAL_DIR/08_improved_evaluation.py ubuntu@$IP:$REMOTE_DIR/
$SCP $LOCAL_DIR/regenerate_figures.py ubuntu@$IP:$REMOTE_DIR/
$SCP $LOCAL_DIR/requirements.txt ubuntu@$IP:$REMOTE_DIR/

# Also upload existing scripts needed for reference
$SCP $LOCAL_DIR/04_optuna_tuning.py ubuntu@$IP:$REMOTE_DIR/ 2>/dev/null || true
$SCP $LOCAL_DIR/05_full_evaluation.py ubuntu@$IP:$REMOTE_DIR/ 2>/dev/null || true

# Blockchain module
$SCP $LOCAL_DIR/blockchain/__init__.py ubuntu@$IP:$REMOTE_DIR/blockchain/ 2>/dev/null || true
$SCP $LOCAL_DIR/blockchain/chain.py ubuntu@$IP:$REMOTE_DIR/blockchain/ 2>/dev/null || true
$SCP $LOCAL_DIR/blockchain/ecies_cipher.py ubuntu@$IP:$REMOTE_DIR/blockchain/ 2>/dev/null || true

# Upload existing results (best_params.json, tuned_cv_results.json are required)
if [ -f $LOCAL_DIR/results/best_params.json ]; then
    echo "  Uploading tuned parameters..."
    $SCP $LOCAL_DIR/results/best_params.json ubuntu@$IP:$REMOTE_DIR/results/
fi
if [ -f $LOCAL_DIR/results/tuned_cv_results.json ]; then
    echo "  Uploading baseline CV results..."
    $SCP $LOCAL_DIR/results/tuned_cv_results.json ubuntu@$IP:$REMOTE_DIR/results/
fi

# Upload existing preprocessed data if available
if [ -f $LOCAL_DIR/provider_features.csv ]; then
    echo "  Uploading 52-feature preprocessed data..."
    $SCP $LOCAL_DIR/provider_features.csv ubuntu@$IP:$REMOTE_DIR/
fi

# Upload other result files that might be needed
for f in $LOCAL_DIR/results/permutation_importance.csv $LOCAL_DIR/results/shap_importance.csv; do
    if [ -f "$f" ]; then
        $SCP "$f" ubuntu@$IP:$REMOTE_DIR/results/ 2>/dev/null || true
    fi
done

echo "  Upload complete."

# 2. Install dependencies
echo ""
echo "[2/7] Installing dependencies..."
$SSH "pip install --quiet optuna catboost xgboost lightgbm imbalanced-learn shap lime cryptography flask scikit-learn scipy joblib matplotlib pandas numpy 2>&1 | tail -5"

# 3. Download Kaggle data if needed
echo ""
echo "[3/7] Checking raw data..."
HAS_DATA=$($SSH "ls $REMOTE_DIR/provider_features.csv 2>/dev/null && echo 'yes' || echo 'no'")
if [ "$HAS_DATA" = "no" ]; then
    echo "  Downloading Kaggle dataset..."
    $SCP ~/.kaggle/kaggle.json ubuntu@$IP:~/.kaggle/kaggle.json 2>/dev/null || true
    $SSH "mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json && pip install --quiet kaggle && mkdir -p ~/data && cd ~/data && kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis && unzip -o *.zip"
    echo "  Running preprocessing (52 features)..."
    $SSH "cd $REMOTE_DIR && python3 01_preprocess.py --out-filename provider_features.csv"
fi

# 4. Run enhanced preprocessing (60 features)
echo ""
echo "[4/7] Running enhanced preprocessing (60 features)..."
HAS_V2=$($SSH "ls $REMOTE_DIR/provider_features_v2.csv 2>/dev/null && echo 'yes' || echo 'no'")
if [ "$HAS_V2" = "no" ]; then
    $SSH "cd $REMOTE_DIR && python3 01_preprocess.py --out-filename provider_features_v2.csv 2>&1 | tee preprocess_v2.log"
else
    echo "  provider_features_v2.csv already exists, skipping."
fi

# 5. Verify required inputs exist
echo ""
echo "[5/7] Verifying required inputs..."
$SSH "ls -la $REMOTE_DIR/provider_features.csv $REMOTE_DIR/provider_features_v2.csv $REMOTE_DIR/results/best_params.json $REMOTE_DIR/results/tuned_cv_results.json"

# 6. Run improved evaluation
echo ""
echo "[6/7] Running improved evaluation (Parts A-G)..."
echo "  This will take approximately 2-3 hours."
echo "  Parts: A=Fix SMOTE bug, B=New models, C=Calibration, D=OOF Stacking,"
echo "         E=60-feature evaluation, F=Threshold opt, G=Statistical tests"
$SSH "cd $REMOTE_DIR && python3 08_improved_evaluation.py 2>&1 | tee improved_evaluation.log" &
EVAL_PID=$!
echo "  PID: $EVAL_PID (waiting...)"
wait $EVAL_PID

# 7. Regenerate all figures
echo ""
echo "[7/7] Regenerating publication-quality figures..."
$SSH "cd $REMOTE_DIR && python3 regenerate_figures.py 2>&1 | tee regenerate_figures.log"

# Download all results
echo ""
echo "=========================================="
echo "  Downloading results..."
echo "=========================================="
mkdir -p $LOCAL_DIR/results/figures

# Results JSON files
$SCP "ubuntu@$IP:$REMOTE_DIR/results/improved_results.json" $LOCAL_DIR/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE_DIR/results/improved_best_model.pkl" $LOCAL_DIR/results/ 2>/dev/null || true

# All figures
$SCP "ubuntu@$IP:$REMOTE_DIR/results/figures/*" $LOCAL_DIR/results/figures/ 2>/dev/null || true

# Logs
$SCP "ubuntu@$IP:$REMOTE_DIR/*.log" $LOCAL_DIR/ 2>/dev/null || true

# Download v2 preprocessed data
$SCP "ubuntu@$IP:$REMOTE_DIR/provider_features_v2.csv" $LOCAL_DIR/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "  DONE"
echo "=========================================="
echo "Results: $LOCAL_DIR/results/improved_results.json"
echo "Figures: $LOCAL_DIR/results/figures/"
echo "Logs:    $LOCAL_DIR/improved_evaluation.log"
echo ""
echo "IMPORTANT: Remember to terminate the Lambda instance to stop billing!"
echo "  lambda-cloud instance terminate <instance-id>"
