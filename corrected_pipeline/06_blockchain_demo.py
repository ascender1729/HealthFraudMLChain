"""
Phase 6: Blockchain + ECIES Integration Demo
Demonstrates the full pipeline: ML prediction -> ECIES encryption -> blockchain audit trail.

This script:
1. Loads the trained model and provider data
2. Generates fraud predictions for all providers
3. Encrypts sensitive PII with ECIES (secp256k1 + AES-256-GCM)
4. Stores prediction records on a SHA-256 blockchain with Merkle trees
5. Validates chain integrity
6. Demonstrates audit trail queries and decryption
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from blockchain import Blockchain, MerkleTree, ECIESCipher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 6: Blockchain + ECIES Demo")
    p.add_argument("--data-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline")
    p.add_argument("--results-dir", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results")
    p.add_argument("--db-path", default="/home/ubuntu/HealthFraudMLChain/corrected_pipeline/results/audit_chain.db")
    p.add_argument("--difficulty", type=int, default=2, help="PoW difficulty (leading zeros)")
    p.add_argument("--batch-size", type=int, default=50, help="Records per block")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR = Path(args.results_dir)

    log.info("=" * 60)
    log.info("PHASE 6: BLOCKCHAIN + ECIES INTEGRATION")
    log.info("=" * 60)

    # ---- Load data and model ----
    log.info("[1/7] Loading model and data...")

    # Try tuned model first, fall back to baseline
    model_path = RESULTS_DIR / "best_model_tuned.pkl"
    if not model_path.exists():
        model_path = RESULTS_DIR / "best_model.pkl"
    if not model_path.exists():
        log.error("No trained model found. Run 02 or 05 first.")
        sys.exit(1)

    model = joblib.load(model_path)
    feature_cols = joblib.load(RESULTS_DIR / "feature_cols.pkl")
    df = pd.read_csv(f"{args.data_dir}/provider_features.csv")

    X = df[feature_cols].values
    providers = df["Provider"].values
    y_true = df["PotentialFraud"].values

    log.info(f"  Loaded model from {model_path.name}")
    log.info(f"  {len(providers)} providers, {len(feature_cols)} features")

    # ---- Generate predictions ----
    log.info("[2/7] Generating fraud predictions...")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    n_flagged = y_pred.sum()
    log.info(f"  Flagged {n_flagged} providers as potential fraud ({n_flagged/len(providers)*100:.1f}%)")

    # ---- Generate ECIES keys ----
    log.info("[3/7] Generating ECIES key pair (secp256k1)...")
    private_key, public_key = ECIESCipher.generate_keypair()

    # Demonstrate key serialization
    pub_bytes = ECIESCipher.serialize_public_key(public_key)
    priv_bytes = ECIESCipher.serialize_private_key(private_key)
    log.info(f"  Public key: {pub_bytes.hex()[:40]}... ({len(pub_bytes)} bytes)")
    log.info(f"  Private key: {len(priv_bytes)} bytes (PEM)")

    # Save keys (in real deployment, private key would be HSM-protected)
    (RESULTS_DIR / "ecies_public_key.bin").write_bytes(pub_bytes)
    (RESULTS_DIR / "ecies_private_key.pem").write_bytes(priv_bytes)
    log.warning("  Private key saved to disk - in production, use HSM or KMS")
    log.info("  Keys generated and saved")

    # ---- Encrypt and store on blockchain ----
    log.info("[4/7] Building blockchain with encrypted records...")
    chain = Blockchain(difficulty=args.difficulty, db_path=args.db_path)

    t0 = time.time()
    records_processed = 0

    for i in range(len(providers)):
        # Build fraud detection record
        record = {
            "provider_id": str(providers[i]),
            "fraud_probability": float(round(y_proba[i], 6)),
            "prediction": "fraud" if y_pred[i] == 1 else "non-fraud",
            "ground_truth": "fraud" if y_true[i] == 1 else "non-fraud",
            "risk_level": (
                "critical" if y_proba[i] >= 0.8 else
                "high" if y_proba[i] >= 0.6 else
                "medium" if y_proba[i] >= 0.4 else
                "low"
            ),
            "model_version": model_path.name,
            "timestamp": time.time(),
        }

        # H2 fix: Encrypt PII fields AND ground truth (both are sensitive)
        encrypted_record = ECIESCipher.encrypt_record(
            record, public_key,
            sensitive_fields=["provider_id", "ground_truth"],
        )

        chain.add_record(encrypted_record)
        records_processed += 1

        # Mine block when batch is full
        if len(chain.pending_records) >= args.batch_size:
            block = chain.mine_block()
            if block:
                log.info(
                    f"  Block #{block.index}: {len(block.records)} records, "
                    f"hash={block.hash[:16]}..., nonce={block.nonce}"
                )

    # Mine remaining records
    if chain.pending_records:
        block = chain.mine_block()
        if block:
            log.info(
                f"  Block #{block.index}: {len(block.records)} records (final), "
                f"hash={block.hash[:16]}..., nonce={block.nonce}"
            )

    elapsed = time.time() - t0
    stats = chain.get_stats()
    log.info(f"  Blockchain built in {elapsed:.1f}s")
    log.info(f"  Chain: {stats['chain_length']} blocks, {stats['total_records']} records")

    # ---- Validate chain ----
    log.info("[5/7] Validating chain integrity...")
    is_valid, errors = chain.validate_chain()
    if is_valid:
        log.info("  Chain validation: PASSED (all blocks verified)")
    else:
        log.error(f"  Chain validation: FAILED ({len(errors)} errors)")
        for err in errors:
            log.error(f"    {err}")

    # ---- Demonstrate Merkle proof ----
    log.info("[6/7] Merkle proof verification demo...")
    if len(chain.chain) > 1:
        demo_block = chain.chain[1]  # First data block
        if demo_block.records:
            # Get proof for first record
            proof = MerkleTree.get_proof(demo_block.records, 0)
            verified = MerkleTree.verify_record(demo_block.records[0], proof, demo_block.merkle_root)
            log.info(f"  Record verification (block #1, record #0): {'PASSED' if verified else 'FAILED'}")
            log.info(f"  Merkle root: {demo_block.merkle_root[:32]}...")
            log.info(f"  Proof depth: {len(proof)} levels")

    # ---- Demonstrate decryption ----
    log.info("[7/7] ECIES decryption demo...")
    if len(chain.chain) > 1:
        sample_record = chain.chain[1].records[0]
        log.info(f"  Encrypted record keys: {list(sample_record.keys())}")
        log.info(f"  PII encrypted: {sample_record.get('pii_encrypted', False)}")

        # Decrypt
        decrypted = ECIESCipher.decrypt_record(sample_record, private_key)
        log.info(f"  Decrypted provider_id: {decrypted.get('provider_id', 'N/A')}")
        log.info(f"  Fraud probability: {decrypted.get('fraud_probability', 'N/A')}")
        log.info(f"  Prediction: {decrypted.get('prediction', 'N/A')}")

    # ---- Save summary ----
    summary = {
        "blockchain": {
            "chain_length": stats["chain_length"],
            "total_records": stats["total_records"],
            "difficulty": args.difficulty,
            "batch_size": args.batch_size,
            "build_time_seconds": elapsed,
            "validation": "PASSED" if is_valid else "FAILED",
            "hash_algorithm": "SHA-256",
            "merkle_tree": True,
            "proof_of_work": True,
            "persistence": "SQLite",
        },
        "ecies": {
            "curve": "secp256k1",
            "kdf": "HKDF-SHA256",
            "cipher": "AES-256-GCM",
            "key_size_bits": 256,
            "fields_encrypted": ["provider_id", "ground_truth"],
        },
        "predictions": {
            "total_providers": int(len(providers)),
            "flagged_fraud": int(n_flagged),
            "fraud_rate": float(n_flagged / len(providers)),
            "risk_distribution": {
                "critical": int((y_proba >= 0.8).sum()),
                "high": int(((y_proba >= 0.6) & (y_proba < 0.8)).sum()),
                "medium": int(((y_proba >= 0.4) & (y_proba < 0.6)).sum()),
                "low": int((y_proba < 0.4).sum()),
            },
        },
    }

    with open(RESULTS_DIR / "blockchain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nResults saved to {RESULTS_DIR}")
    log.info("Phase 6 complete.")


if __name__ == "__main__":
    main()
