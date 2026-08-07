-- ============================================================
-- AAA Engine — Layer 1: Speaker Verification
-- Database Schema Setup (PostgreSQL)
-- ============================================================

-- voiceprints table: stores enrolled 192-dim ECAPA-TDNN embeddings
CREATE TABLE voiceprints (
    user_id      UUID PRIMARY KEY,
    embedding    FLOAT8[192]  NOT NULL,  -- 192-dim ECAPA-TDNN voiceprint
    emb_hash     TEXT         NOT NULL,  -- SHA256 of embedding (integrity check)
    update_count INTEGER      DEFAULT 0, -- number of Layer 3 adaptive updates
    enrolled_at  TIMESTAMP    DEFAULT NOW(),
    last_updated TIMESTAMP    DEFAULT NOW()
);

-- auth_log table: records every verification attempt
CREATE TABLE auth_log (
    session_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID         REFERENCES voiceprints(user_id),
    result        TEXT         NOT NULL,  -- 'ACCEPT' or 'REJECT'
    layer_blocked INTEGER,               -- 1 if Layer 1 rejected, NULL if accepted
    l1_score      FLOAT8,               -- cosine similarity score (0.0 - 1.0)
    threshold     FLOAT8       DEFAULT 0.75,
    timestamp     TIMESTAMP    DEFAULT NOW()
);
