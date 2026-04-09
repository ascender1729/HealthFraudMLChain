#!/bin/bash
# Deploy corrected pipeline to Lambda Cloud and run Optuna + full evaluation
# Usage: bash deploy_and_run.sh <lambda-ip>

set -e

IP="${1:-$(cat /tmp/lambda_ip.txt 2>/dev/null)}"
if [ -z "$IP" ]; then
    echo "Usage: bash deploy_and_run.sh <lambda-cloud-ip>"
    exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud"

echo "=== Deploying to $IP ==="

# Update SSH config
sed -i "s/HostName .*/HostName $IP/" ~/.ssh/config 2>/dev/null || true

# 1. Upload all pipeline files
echo "[1/6] Uploading pipeline files..."
$SSH "mkdir -p ~/HealthFraudMLChain/corrected_pipeline/blockchain ~/HealthFraudMLChain/corrected_pipeline/results ~/HealthFraudMLChain/corrected_pipeline/app/templates"

$SCP corrected_pipeline/01_preprocess.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/
$SCP corrected_pipeline/04_optuna_tuning.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/
$SCP corrected_pipeline/05_full_evaluation.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/
$SCP corrected_pipeline/06_blockchain_demo.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/
$SCP corrected_pipeline/requirements.txt ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/
$SCP corrected_pipeline/blockchain/__init__.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/blockchain/
$SCP corrected_pipeline/blockchain/chain.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/blockchain/
$SCP corrected_pipeline/blockchain/ecies_cipher.py ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/blockchain/

# Upload existing results if provider_features.csv exists locally
if [ -f corrected_pipeline/provider_features.csv ]; then
    echo "  Uploading preprocessed data..."
    $SCP corrected_pipeline/provider_features.csv ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/
fi

# 2. Install dependencies
echo "[2/6] Installing dependencies..."
$SSH "pip install --quiet optuna catboost xgboost lightgbm imbalanced-learn shap lime cryptography flask scikit-learn scipy joblib matplotlib pandas numpy 2>&1 | tail -5"

# 3. Setup Kaggle data (if not already preprocessed)
echo "[3/6] Checking data..."
HAS_DATA=$($SSH "ls ~/HealthFraudMLChain/corrected_pipeline/provider_features.csv 2>/dev/null && echo 'yes' || echo 'no'")
if [ "$HAS_DATA" = "no" ]; then
    echo "  Downloading Kaggle dataset..."
    $SCP ~/.kaggle/kaggle.json ubuntu@$IP:~/.kaggle/kaggle.json 2>/dev/null || true
    $SSH "mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json && pip install --quiet kaggle && mkdir -p ~/data && cd ~/data && kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis && unzip -o *.zip"
    echo "  Running preprocessing..."
    $SSH "cd ~/HealthFraudMLChain/corrected_pipeline && python3 01_preprocess.py"
fi

# 4. Run Optuna tuning
echo "[4/6] Running Optuna hyperparameter tuning (60 trials/model)..."
$SSH "cd ~/HealthFraudMLChain/corrected_pipeline && python3 04_optuna_tuning.py --n-trials 60 --timeout 1800 2>&1 | tee optuna_tuning.log" &
OPTUNA_PID=$!
echo "  PID: $OPTUNA_PID (waiting...)"
wait $OPTUNA_PID

# 5. Run full evaluation with tuned params
echo "[5/6] Running full evaluation with tuned models..."
$SSH "cd ~/HealthFraudMLChain/corrected_pipeline && python3 05_full_evaluation.py 2>&1 | tee full_evaluation.log" &
EVAL_PID=$!
echo "  PID: $EVAL_PID (waiting...)"
wait $EVAL_PID

# 6. Run blockchain demo
echo "[6/6] Running blockchain + ECIES demo..."
$SSH "cd ~/HealthFraudMLChain/corrected_pipeline && python3 06_blockchain_demo.py 2>&1 | tee blockchain_demo.log"

# Download results
echo "=== Downloading results ==="
mkdir -p corrected_pipeline/results/figures
$SCP "ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/results/*" corrected_pipeline/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/results/figures/*" corrected_pipeline/results/figures/ 2>/dev/null || true
$SCP "ubuntu@$IP:~/HealthFraudMLChain/corrected_pipeline/*.log" corrected_pipeline/ 2>/dev/null || true

echo "=== DONE ==="
echo "Results in corrected_pipeline/results/"
echo "Remember to terminate the instance!"
