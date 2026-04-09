"""
Blockchain module for healthcare fraud detection audit trail.
Implements SHA-256 hashing, Merkle trees, proof-of-work, and ECIES encryption.
All components use the `cryptography` library (pre-June 2024).
"""
from .chain import Block, Blockchain, MerkleTree
from .ecies_cipher import ECIESCipher

__all__ = ["Block", "Blockchain", "MerkleTree", "ECIESCipher"]
