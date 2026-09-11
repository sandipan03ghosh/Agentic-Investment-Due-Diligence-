from __future__ import annotations

import hashlib


def new_sha256() -> "hashlib._Hash":
    return hashlib.sha256()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
