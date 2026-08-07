# AAA Engine — Adaptive Acoustic Auth Engine

## Layer 1: Speaker Verification

ECAPA-TDNN Speaker Embedding & Cosine Verification Pipeline.

### Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up PostgreSQL (optional — needed for enrollment/verification)
psql -U postgres -f setup_db.sql

# 4. Run unit tests (no DB or model needed)
cd aaa_engine
python -m pytest ../tests/test_layer1.py -v

# 5. Start the API server
cd aaa_engine
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Project Structure

```
aaa_engine/
├── layer1/
│   ├── __init__.py              ← Package init & re-exports
│   ├── speaker_verifier.py      ← CORE: ECAPA-TDNN embedding + cosine similarity
│   ├── enrollment.py            ← Enrollment / re-enrollment flow
│   └── db_utils.py              ← PostgreSQL read/write for voiceprints
├── pretrained_models/           ← Auto-created by SpeechBrain on first run
└── main.py                      ← FastAPI app with /auth/enroll & /auth/verify
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/auth/enroll` | POST | Enroll a new user (user_id + audio file) |
| `/auth/verify` | POST | Verify speaker (user_id + session_token + audio) |

### Configuration

- **Cosine threshold**: 0.75 (in `speaker_verifier.py`)
- **Audio duration**: 1.5s–10.0s for verification, 10s–15s for enrollment
- **Sample rate**: 16kHz mono
- **Embedding size**: 192 dimensions
- **DB config**: Edit `DB_CONFIG` in `db_utils.py`
