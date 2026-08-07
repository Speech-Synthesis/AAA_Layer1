# =============================================================
# layer1/db_utils.py
# AAA Engine — Layer 1: Speaker Verification
# PostgreSQL database utilities for voiceprint storage & auth logging
# =============================================================

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("[DB WARNING] psycopg2 not installed. Database operations will be disabled.")
    print("[DB WARNING] Install with: pip install psycopg2-binary")

import numpy as np
import hashlib
import json
from typing import Optional

# ── Database Configuration ──────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "aaa_engine",
    "user": "postgres",
    "password": "your_password"
}


def get_connection():
    """Return a new PostgreSQL connection."""
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 is not installed. Install with: pip install psycopg2-binary")
    return psycopg2.connect(**DB_CONFIG)


def store_voiceprint(user_id: str, embedding: np.ndarray) -> bool:
    """
    Store a 192-dim embedding for a new user.
    Returns True on success, False on failure.

    Uses ON CONFLICT DO UPDATE to support re-enrollment:
    if the user_id already exists, the embedding and hash
    are overwritten and last_updated is refreshed.
    """
    emb_list = embedding.tolist()
    emb_hash = hashlib.sha256(
        json.dumps(emb_list).encode()
    ).hexdigest()

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            '''
            INSERT INTO voiceprints (user_id, embedding, emb_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    emb_hash = EXCLUDED.emb_hash,
                    last_updated = NOW()
            ''',
            (user_id, emb_list, emb_hash)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB ERROR] store_voiceprint: {e}")
        return False


def fetch_voiceprint(user_id: str) -> Optional[np.ndarray]:
    """
    Retrieve stored embedding for a user_id.
    Returns numpy array of shape (192,) or None if not found.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            'SELECT embedding FROM voiceprints WHERE user_id = %s',
            (user_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return np.array(row[0], dtype=np.float32)
        return None
    except Exception as e:
        print(f"[DB ERROR] fetch_voiceprint: {e}")
        return None


def log_auth_result(
    session_id: str,
    user_id: str,
    result: str,
    l1_score: float,
    layer_blocked: Optional[int]
) -> None:
    """
    Log every verification attempt to auth_log table.

    Args:
        session_id:    Unique session identifier (UUID)
        user_id:       The user being authenticated
        result:        'ACCEPT' or 'REJECT'
        l1_score:      Cosine similarity score (0.0 - 1.0)
        layer_blocked: 1 if Layer 1 rejected, None if accepted
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            '''
            INSERT INTO auth_log
            (session_id, user_id, result, layer_blocked, l1_score)
            VALUES (%s, %s, %s, %s, %s)
            ''',
            (session_id, user_id, result, layer_blocked, l1_score)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] log_auth_result: {e}")
