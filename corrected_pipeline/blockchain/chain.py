"""
Blockchain with SHA-256 hashing, Merkle trees, proof-of-work, and SQLite persistence.
Replaces the original toy MD5 implementation (code/block.py).

Design:
- Each block stores a batch of fraud detection records
- Merkle root provides tamper-evident integrity for all records in the block
- Proof-of-work with adjustable difficulty prevents trivial chain manipulation
- SQLite backend for persistent storage
- Chain validation verifies hash linkage, Merkle roots, and PoW
"""
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class MerkleTree:
    """SHA-256 Merkle tree for tamper-evident record batching."""

    @staticmethod
    def hash_leaf(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        combined = left + right
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @classmethod
    def compute_root(cls, records: list[dict]) -> str:
        """Compute Merkle root from a list of record dicts."""
        if not records:
            return hashlib.sha256(b"empty").hexdigest()

        # Leaf hashes
        leaves = [cls.hash_leaf(json.dumps(r, sort_keys=True, default=str)) for r in records]

        # Build tree bottom-up
        level = leaves
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                next_level.append(cls.hash_pair(left, right))
            level = next_level

        return level[0]

    @classmethod
    def verify_record(cls, record: dict, proof: list[tuple[str, str]], root: str) -> bool:
        """Verify a single record against a Merkle proof."""
        current = cls.hash_leaf(json.dumps(record, sort_keys=True, default=str))
        for sibling, direction in proof:
            if direction == "left":
                current = cls.hash_pair(sibling, current)
            else:
                current = cls.hash_pair(current, sibling)
        return current == root

    @classmethod
    def get_proof(cls, records: list[dict], index: int) -> list[tuple[str, str]]:
        """Get Merkle proof for a record at given index."""
        if not records or index >= len(records):
            return []

        leaves = [cls.hash_leaf(json.dumps(r, sort_keys=True, default=str)) for r in records]
        proof = []
        level = leaves

        idx = index
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                next_level.append(cls.hash_pair(left, right))

                if i == idx or i + 1 == idx:
                    if i == idx:
                        sibling = right
                        direction = "right"
                    else:
                        sibling = left
                        direction = "left"
                    proof.append((sibling, direction))

            idx = idx // 2
            level = next_level

        return proof


@dataclass
class Block:
    """A single block in the chain."""
    index: int
    timestamp: float
    records: list[dict]
    merkle_root: str
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        block_data = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(block_data.encode("utf-8")).hexdigest()

    def mine(self, difficulty: int = 2) -> str:
        """Proof-of-work: find nonce such that hash starts with `difficulty` zeros."""
        target = "0" * difficulty
        while True:
            self.hash = self.compute_hash()
            if self.hash.startswith(target):
                return self.hash
            self.nonce += 1

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "records": self.records,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
            "n_records": len(self.records),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        block = cls(
            index=data["index"],
            timestamp=data["timestamp"],
            records=data["records"],
            merkle_root=data["merkle_root"],
            previous_hash=data["previous_hash"],
            nonce=data["nonce"],
            hash=data["hash"],
        )
        return block


class Blockchain:
    """
    Immutable audit trail for fraud detection results.

    Each block contains a batch of fraud prediction records:
    - provider_id, fraud_probability, prediction, timestamp
    - Optionally encrypted PII via ECIES
    """

    def __init__(self, difficulty: int = 2, db_path: Optional[str] = None):
        self.difficulty = difficulty
        self.chain: list[Block] = []
        self.pending_records: list[dict] = []
        self.db_path = db_path

        if db_path:
            self._init_db()
            self._load_from_db()

        if not self.chain:
            self._create_genesis()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                idx INTEGER PRIMARY KEY,
                block_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT block_json FROM blocks ORDER BY idx").fetchall()
        conn.close()
        for row in rows:
            block_data = json.loads(row[0])
            self.chain.append(Block.from_dict(block_data))

    def _save_block_to_db(self, block: Block):
        if not self.db_path:
            return
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO blocks (idx, block_json) VALUES (?, ?)",
            (block.index, json.dumps(block.to_dict(), default=str)),
        )
        conn.commit()
        conn.close()

    def _create_genesis(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            records=[{"type": "genesis", "message": "Healthcare Fraud Detection Audit Chain"}],
            merkle_root=MerkleTree.compute_root([{"type": "genesis", "message": "Healthcare Fraud Detection Audit Chain"}]),
            previous_hash="0" * 64,
        )
        genesis.mine(self.difficulty)
        self.chain.append(genesis)
        self._save_block_to_db(genesis)

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def add_record(self, record: dict):
        """Add a fraud detection record to pending batch."""
        record["added_at"] = time.time()
        self.pending_records.append(record)

    def mine_block(self) -> Optional[Block]:
        """Mine pending records into a new block."""
        if not self.pending_records:
            return None

        merkle_root = MerkleTree.compute_root(self.pending_records)

        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            records=list(self.pending_records),
            merkle_root=merkle_root,
            previous_hash=self.last_block.hash,
        )
        block.mine(self.difficulty)

        self.chain.append(block)
        self._save_block_to_db(block)
        self.pending_records.clear()

        return block

    def validate_chain(self) -> tuple[bool, list[str]]:
        """Validate entire chain integrity."""
        errors = []

        for i, block in enumerate(self.chain):
            # Verify hash
            computed = block.compute_hash()
            if computed != block.hash:
                errors.append(f"Block {i}: hash mismatch (stored={block.hash[:16]}..., computed={computed[:16]}...)")

            # Verify proof-of-work
            if not block.hash.startswith("0" * self.difficulty):
                errors.append(f"Block {i}: invalid proof-of-work")

            # Verify Merkle root
            expected_root = MerkleTree.compute_root(block.records)
            if expected_root != block.merkle_root:
                errors.append(f"Block {i}: Merkle root mismatch")

            # Verify chain linkage
            if i > 0 and block.previous_hash != self.chain[i - 1].hash:
                errors.append(f"Block {i}: previous_hash doesn't match block {i-1}")

        return len(errors) == 0, errors

    def get_audit_trail(self, provider_id: str) -> list[dict]:
        """Get all records for a specific provider."""
        trail = []
        for block in self.chain:
            for record in block.records:
                if record.get("provider_id") == provider_id:
                    trail.append({
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "timestamp": block.timestamp,
                        **record,
                    })
        return trail

    def get_stats(self) -> dict:
        total_records = sum(len(b.records) for b in self.chain)
        return {
            "chain_length": len(self.chain),
            "total_records": total_records,
            "difficulty": self.difficulty,
            "last_block_hash": self.last_block.hash,
            "pending_records": len(self.pending_records),
        }

    def validate_against_peer(self, peer: "Blockchain") -> tuple[bool, list[str]]:
        """Cross-validate this chain against an independent peer's chain.

        Compares block hashes, Merkle roots, and chain length to detect
        any divergence between two independently maintained copies.
        """
        errors = []
        if len(self.chain) != len(peer.chain):
            errors.append(
                f"Chain length mismatch: {len(self.chain)} vs {len(peer.chain)}"
            )
        for i, (a, b) in enumerate(zip(self.chain, peer.chain)):
            if a.hash != b.hash:
                errors.append(f"Block {i}: hash mismatch ({a.hash[:12]}... vs {b.hash[:12]}...)")
            if a.merkle_root != b.merkle_root:
                errors.append(f"Block {i}: Merkle root mismatch")
            if len(a.records) != len(b.records):
                errors.append(f"Block {i}: record count mismatch ({len(a.records)} vs {len(b.records)})")
        return len(errors) == 0, errors

    def get_chain_fingerprint(self) -> str:
        """Return a compact fingerprint of the entire chain for quick comparison."""
        import hashlib
        h = hashlib.sha256()
        for block in self.chain:
            h.update(block.hash.encode())
        return h.hexdigest()

    def to_json(self) -> str:
        return json.dumps(
            [b.to_dict() for b in self.chain],
            indent=2,
            default=str,
        )
