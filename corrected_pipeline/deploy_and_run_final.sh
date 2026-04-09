#!/bin/bash
# Deploy FINAL evaluation to Lambda Cloud
# Usage: bash deploy_and_run_final.sh <lambda-ip>
set -e

IP="${1:-$(cat /tmp/lambda_ip.txt 2>/dev/null)}"
if [ -z "$IP" ]; then echo "Usage: bash deploy_and_run_final.sh <lambda-ip>"; exit 1; fi

SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i ~/.ssh/lambda_cloud"
REMOTE="~/HealthFraudMLChain/corrected_pipeline"
LOCAL="corrected_pipeline"

echo "============================================"
echo "  FINAL Evaluation (leakage-free + claim model)"
echo "  Target: $IP"
echo "============================================"

echo -e "\n[1/6] Upload..."
$SSH "mkdir -p $REMOTE/results"
$SCP $LOCAL/09_advanced_preprocess.py ubuntu@$IP:$REMOTE/
$SCP $LOCAL/14_final_evaluation.py ubuntu@$IP:$REMOTE/
$SCP $LOCAL/results/best_params_v3.json ubuntu@$IP:$REMOTE/results/ 2>/dev/null || true
$SCP $LOCAL/results/exhaustive_params.json ubuntu@$IP:$REMOTE/results/ 2>/dev/null || true

echo -e "\n[2/6] Install deps..."
$SSH "pip install --upgrade numpy pandas scipy scikit-learn 2>&1 | tail -2"
$SSH "pip install --quiet optuna catboost xgboost lightgbm imbalanced-learn joblib category_encoders matplotlib 2>&1 | tail -2"
$SSH "python3 -c 'import sklearn,xgboost,lightgbm,catboost,optuna,category_encoders; print(\"OK\")'"

echo -e "\n[3/6] Data..."
HAS=$($SSH "ls ~/data/Train-1542865627584.csv 2>/dev/null && echo yes || echo no")
if [ "$HAS" = "no" ]; then
    $SCP ~/.kaggle/kaggle.json ubuntu@$IP:~/
    $SSH "mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json && pip install --quiet kaggle 2>&1 | tail -1"
    $SSH "mkdir -p ~/data && cd ~/data && ~/.local/bin/kaggle datasets download -d rohitrox/healthcare-provider-fraud-detection-analysis && unzip -o *.zip 2>&1 | tail -3"
fi

echo -e "\n[4/6] Preprocess v3 (if needed)..."
HAS_V3=$($SSH "ls $REMOTE/provider_features_v3.csv 2>/dev/null && echo yes || echo no")
if [ "$HAS_V3" = "no" ]; then
    $SSH "cd $REMOTE && python3 09_advanced_preprocess.py 2>&1 | tail -5"
fi

echo -e "\n[5/6] Run FINAL evaluation..."
$SSH "cd $REMOTE && python3 14_final_evaluation.py 2>&1 | tee final_evaluation.log"

echo -e "\n[6/6] Download results..."
mkdir -p $LOCAL/results
$SCP "ubuntu@$IP:$REMOTE/results/final_results.json" $LOCAL/results/ 2>/dev/null || true
$SCP "ubuntu@$IP:$REMOTE/final_evaluation.log" $LOCAL/ 2>/dev/null || true

echo -e "\n============================================"
echo "  DONE - Terminate the instance!"
echo "============================================"
