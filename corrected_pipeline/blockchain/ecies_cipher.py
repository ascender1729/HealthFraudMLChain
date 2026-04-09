"""
ECIES (Elliptic Curve Integrated Encryption Scheme) for PII protection.
Uses secp256k1 + HKDF-SHA256 + AES-256-GCM.

All primitives from the `cryptography` library (available pre-June 2024).

Purpose in thesis context:
- Encrypt sensitive patient/provider data before storing on the blockchain
- Only authorized parties with the private key can decrypt PII
- Fraud scores and hashes remain visible for auditing

ECIES flow:
1. Sender generates ephemeral EC key pair
2. ECDH: shared_secret = ephemeral_private * recipient_public
3. KDF: derive AES key from shared_secret using HKDF-SHA256
4. Encrypt plaintext with AES-256-GCM
5. Send: ephemeral_public_key || nonce || ciphertext || tag
"""
import json
import os
from base64 import b64decode, b64encode

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class ECIESCipher:
    """ECIES encryption/decryption using secp256k1."""

    CURVE = ec.SECP256K1()
    KEY_SIZE = 32  # AES-256
    NONCE_SIZE = 12  # AES-GCM standard

    @classmethod
    def generate_keypair(cls) -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        """Generate a new secp256k1 key pair."""
        private_key = ec.generate_private_key(cls.CURVE)
        return private_key, private_key.public_key()

    @classmethod
    def serialize_public_key(cls, public_key: ec.EllipticCurvePublicKey) -> bytes:
        """Serialize public key to compressed format."""
        return public_key.public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.CompressedPoint,
        )

    @classmethod
    def deserialize_public_key(cls, data: bytes) -> ec.EllipticCurvePublicKey:
        """Deserialize public key from compressed format."""
        return ec.EllipticCurvePublicKey.from_encoded_point(cls.CURVE, data)

    @classmethod
    def serialize_private_key(cls, private_key: ec.EllipticCurvePrivateKey) -> bytes:
        """Serialize private key (PEM format, no encryption)."""
        return private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    @classmethod
    def deserialize_private_key(cls, data: bytes) -> ec.EllipticCurvePrivateKey:
        """Deserialize private key from PEM."""
        return serialization.load_pem_private_key(data, password=None)

    @classmethod
    def _derive_key(cls, shared_secret: bytes) -> bytes:
        """Derive AES-256 key from ECDH shared secret using HKDF-SHA256."""
        return HKDF(
            algorithm=hashes.SHA256(),
            length=cls.KEY_SIZE,
            salt=None,
            info=b"healthcare-fraud-ecies",
        ).derive(shared_secret)

    @classmethod
    def encrypt(cls, plaintext: str, recipient_public_key: ec.EllipticCurvePublicKey) -> dict:
        """
        Encrypt plaintext for a recipient.

        Returns dict with:
        - ephemeral_pubkey: base64-encoded ephemeral public key
        - nonce: base64-encoded AES-GCM nonce
        - ciphertext: base64-encoded encrypted data (includes GCM tag)
        """
        # Generate ephemeral key pair
        ephemeral_private = ec.generate_private_key(cls.CURVE)
        ephemeral_public = ephemeral_private.public_key()

        # ECDH to get shared secret
        shared_secret = ephemeral_private.exchange(ec.ECDH(), recipient_public_key)

        # Derive AES key
        aes_key = cls._derive_key(shared_secret)

        # Encrypt with AES-256-GCM
        nonce = os.urandom(cls.NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        return {
            "ephemeral_pubkey": b64encode(cls.serialize_public_key(ephemeral_public)).decode("ascii"),
            "nonce": b64encode(nonce).decode("ascii"),
            "ciphertext": b64encode(ciphertext).decode("ascii"),
        }

    @classmethod
    def decrypt(cls, encrypted: dict, recipient_private_key: ec.EllipticCurvePrivateKey) -> str:
        """
        Decrypt ciphertext using recipient's private key.

        Args:
            encrypted: dict from encrypt() with ephemeral_pubkey, nonce, ciphertext
            recipient_private_key: recipient's secp256k1 private key

        Returns:
            Decrypted plaintext string
        """
        # Reconstruct ephemeral public key
        ephemeral_public = cls.deserialize_public_key(b64decode(encrypted["ephemeral_pubkey"]))

        # ECDH to recover shared secret
        shared_secret = recipient_private_key.exchange(ec.ECDH(), ephemeral_public)

        # Derive AES key
        aes_key = cls._derive_key(shared_secret)

        # Decrypt
        nonce = b64decode(encrypted["nonce"])
        ciphertext = b64decode(encrypted["ciphertext"])
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return plaintext.decode("utf-8")

    @classmethod
    def encrypt_record(cls, record: dict, recipient_public_key: ec.EllipticCurvePublicKey,
                       sensitive_fields: list[str] = None) -> dict:
        """
        Encrypt sensitive fields in a fraud detection record.
        Non-sensitive fields (fraud_score, prediction) remain in plaintext.

        Args:
            record: fraud detection record dict
            recipient_public_key: key for encryption
            sensitive_fields: list of field names to encrypt (default: PII fields)

        Returns:
            Record with sensitive fields encrypted, others untouched
        """
        if sensitive_fields is None:
            sensitive_fields = ["provider_id", "patient_ids", "claim_ids", "pii_data"]

        result = {}
        encrypted_data = {}

        for key, value in record.items():
            if key in sensitive_fields and value is not None:
                encrypted_data[key] = value
            else:
                result[key] = value

        if encrypted_data:
            result["encrypted_pii"] = cls.encrypt(
                json.dumps(encrypted_data, default=str),
                recipient_public_key,
            )
            result["pii_encrypted"] = True
        else:
            result["pii_encrypted"] = False

        return result

    @classmethod
    def decrypt_record(cls, record: dict, recipient_private_key: ec.EllipticCurvePrivateKey) -> dict:
        """Decrypt sensitive fields in a record."""
        if not record.get("pii_encrypted"):
            return record

        result = dict(record)
        encrypted_pii = result.pop("encrypted_pii", None)
        result.pop("pii_encrypted", None)

        if encrypted_pii:
            decrypted = json.loads(cls.decrypt(encrypted_pii, recipient_private_key))
            result.update(decrypted)

        return result
