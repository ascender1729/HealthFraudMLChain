#!/bin/bash
# Deploy advanced pipeline (v3) to Lambda Cloud - target F1 0.85+
# Usage: bash deploy_and_run_v3.sh <lambda-ip>
set -e

IP="${1:-$(cat /tmp/lambda_ip.txt 2>/dev/null)}"
if [ -z "$IP" ]; then
    echo "Usage: bash deploy_and_run_v3.sh <lambda-cloud-ip>"
    exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud"
REMOTE="~/HealthFraudMLChain/corrected_pipeline"
LOCAL="corrected_pipeline"

echo "============================================"
echo "  Advanced Pipeline v3 - Target F1 0.85+"
echo "  Target: $IP"
echo "============================================"

# 1. Upload
echo -e "\n[1/6] Uploading files..."
$SSH "mkdir -p $REMOTE/results/figures"
$SCP $LOCAL/09_advanced_preprocess.py ubuntu@$IP:$REMOTE/
$SCP $LOCAL/10_advanced_evaluation.py ubuntu@$IP:$REMOTE/
$SCP $LOCAL/requirements.txt ubuntu@$IP:$REMOTE/
$SCP $LOCAL/results/best_params.json ubuntu@$IP:$REMOTE/results/ 2>/dev/null || true

# 2. Install deps
echo -e "\n[2/6] Installing dependencies..."
$SSH "pip install --upgrade numpy pandas scipy scikit-learn matplotlib 2>&1 | tail -3"
$SSH "pip install --quiet optuna catboost xgboost lightgbm imbalanced-learn joblib 2>&1 | tail -3"
echo "  Verifying imports..."
$SSH "python3 -c 'import sklearn, xgboost, lightgbm, catboost, optuna, imblearn; print(\"All imports OK\")'"

# 3. Data
echo -e "\n[3/6] Checking data..."
HAS_DATA=$($SSH "ls ~/data/Train-1542865627584.csv 2>/dev/null && echo 'yes' || echo 'no'")
if [ "$HAS_DATA" = "no" ]; then
    echo "  Downloading Kaggle dataset..."
    $SCP ~/.kaggle/kaggle.json ubuntu@$IP:~/
    $SSH "mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json"
    $SSH "pip install --quiet kaggle 2>&1 | tail -1"
    $SSH "mkdir -p ~/data && cd ~/data && ~/.local/bin/kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis && unzip -o *.zip 2>&1 | tail -3"
fi
echo "  Data ready."

# 4. Run advanced preprocessing (180 features)
echo -e "\n[4/6] Running advanced preprocessing (~180 features)..."
$SSH "cd $REMOTE && python3 09_advanced_preprocess.py 2>&1 | tee advanced_preprocess.log"

# 5. Run advanced evaluation (8 models + ensemble)
echo -e "\n[5/6] Running advanced evaluation (8 models + Optuna ensemble)..."
echo "  Estimated runtime: 60-90 minutes"
$SSH "cd $REMOTE && python3 10_advanced_evaluation.py 2>&1 | tee advanced_evaluation.log"

# 6. Download results
echo -e "\n[6/6] Downloading results..."
mkdir -p $LOCAL/results
$SCP "ubuntu@$IP:$REMOTE/results/advanced_results.json" $LOCAL/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE/results/advanced_best_model.pkl" $LOCAL/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE/results/mutual_info_scores_v3.csv" $LOCAL/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE/results/selected_features_v3.json" $LOCAL/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE/provider_features_v3.csv" $LOCAL/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE/*.log" $LOCAL/ 2>/dev/null || true

echo -e "\n============================================"
echo "  DONE"
echo "============================================"
echo "Results: $LOCAL/results/advanced_results.json"
echo "IMPORTANT: Terminate the Lambda instance!"
