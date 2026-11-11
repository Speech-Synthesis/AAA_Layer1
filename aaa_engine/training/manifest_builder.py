# =============================================================
# training/manifest_builder.py
# AAA Engine — JSONL Manifest Builder
# =============================================================
# Builds standardised JSONL manifests for speaker verification
# training, calibration, and evaluation.
#
# Supported source datasets:
#   • VoxCeleb1   (folder structure: wav_root/spkID/videoID/*.wav)
#   • Common Voice (validated.tsv)
#   • In-house recordings (directory + metadata CSV)
#
# Output schema (one JSON object per line):
#   {
#     "id":              <str>           unique within manifest,
#     "split":           <str>           train|dev|test|cal,
#     "audio_path":      <str>           absolute path,
#     "sha256":          <str>           hex digest of file bytes,
#     "duration_s":      <float>         seconds,
#     "sample_rate":     <int>           Hz,
#     "channels":        <int>           1 or 2,
#     "speaker_id":      <str>           dataset-specific ID,
#     "source_dataset":  <str>           voxceleb1|commonvoice|inhouse,
#     "licence":         <str>           SPDX or free-form note,
#     "consent_id":      <str|null>      null if not applicable,
#     "session_id":      <str|null>,
#     "device":          <str|null>,
#     "is_augmented":    <bool>,
#     "augmentation":    <str|null>      description if is_augmented
#   }
#
# Usage:
#   python -m aaa_engine.training.manifest_builder voxceleb \
#       --wav-root /data/voxceleb1 --split train --output manifest.jsonl
#
#   python -m aaa_engine.training.manifest_builder commonvoice \
#       --cv-root /data/cv-corpus-13.0 --split train \
#       --client-ids-file ids.txt --output manifest.jsonl
#
#   python -m aaa_engine.training.manifest_builder inhouse \
#       --recordings-dir /data/inhouse --metadata-csv metadata.csv \
#       --partition-map partition.json --output manifest.jsonl
#
#   python -m aaa_engine.training.manifest_builder hash \
#       --manifest manifest.jsonl
# =============================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DURATION_MIN_S = 1.0
DURATION_MAX_S = 10.0

# JSONL record schema — all fields; None values written as null
RECORD_KEYS = [
    "id", "split", "audio_path", "sha256", "duration_s", "sample_rate",
    "channels", "speaker_id", "source_dataset", "licence", "consent_id",
    "session_id", "device", "is_augmented", "augmentation",
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    """Return hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _audio_info(path: str) -> Optional[Tuple[float, int, int]]:
    """
    Return (duration_s, sample_rate, channels) using torchaudio.info.

    Returns None on error (logs a warning).
    """
    try:
        info = torchaudio.info(path)
        duration_s = info.num_frames / info.sample_rate
        return duration_s, info.sample_rate, info.num_channels
    except Exception as exc:
        logger.warning("torchaudio.info failed for %s: %s", path, exc)
        return None


def _make_record(**kwargs) -> Dict:
    """Build a manifest record dict with all required keys present."""
    rec = {k: None for k in RECORD_KEYS}
    rec.update(kwargs)
    # Ensure booleans, not None
    if rec["is_augmented"] is None:
        rec["is_augmented"] = False
    return rec


def _write_jsonl(records: List[Dict], output_path: str) -> None:
    """Write list of dicts to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(records), output_path)


def _write_exclusion_report(
    excluded: List[Dict], output_path: str, reason_key: str = "reason"
) -> None:
    """Write exclusion report as JSONL alongside the manifest."""
    if not excluded:
        return
    report_path = str(output_path).replace(".jsonl", "_excluded.jsonl")
    with open(report_path, "w", encoding="utf-8") as f:
        for item in excluded:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(
        "Exclusion report: %d items → %s", len(excluded), report_path
    )


# ---------------------------------------------------------------------------
# VoxCeleb1 manifest builder
# ---------------------------------------------------------------------------
def build_voxceleb_manifest(
    wav_root: str,
    split: str,
    output_path: str,
    seed: int = 42,
) -> None:
    """
    Build a JSONL manifest from VoxCeleb1 folder structure.

    Expected layout::

        wav_root/
            id07xxx/        ← speaker ID
                videoID/    ← YouTube video ID
                    *.wav   ← utterance files

    Processing:
      • Computes sha256 of each file.
      • Measures duration via torchaudio.info (no decode).
      • Excludes files with duration outside [1.0, 10.0] s.
      • Excludes sha256 duplicates (keeps first encountered).
      • Writes an exclusion report alongside the output JSONL.

    Licence note:
      "CC BY 4.0 [VERIFY]" is a placeholder.  Confirm the actual
      licence of your VoxCeleb1 copy before use.

    Args:
        wav_root:    Root directory of VoxCeleb1 wav files.
        split:       Split label to assign (e.g. 'train', 'dev', 'test').
        output_path: Destination JSONL file.
        seed:        Not used for VoxCeleb (deterministic walk), kept for
                     API consistency.
    """
    wav_root_p = Path(wav_root)
    if not wav_root_p.is_dir():
        raise NotADirectoryError(f"wav_root not found: {wav_root}")

    records: List[Dict] = []
    excluded: List[Dict] = []
    seen_hashes: Set[str] = set()

    # Walk: spkID / videoID / *.wav
    spk_dirs = sorted(p for p in wav_root_p.iterdir() if p.is_dir())
    logger.info("VoxCeleb1: found %d speaker dirs under %s", len(spk_dirs), wav_root_p)

    for spk_dir in spk_dirs:
        spk_id = spk_dir.name
        for video_dir in sorted(p for p in spk_dir.iterdir() if p.is_dir()):
            for wav_file in sorted(video_dir.glob("*.wav")):
                fpath = str(wav_file.resolve())

                # Audio info (no decode)
                info = _audio_info(fpath)
                if info is None:
                    excluded.append({"path": fpath, "reason": "torchaudio.info_error"})
                    continue
                duration_s, sample_rate, channels = info

                # Duration filter
                if not (DURATION_MIN_S <= duration_s <= DURATION_MAX_S):
                    excluded.append({
                        "path": fpath,
                        "reason": f"duration_oob:{duration_s:.3f}s",
                    })
                    continue

                # SHA256 (also deduplication key)
                sha = _sha256_file(fpath)
                if sha in seen_hashes:
                    excluded.append({
                        "path": fpath,
                        "reason": f"sha256_duplicate:{sha[:12]}",
                    })
                    continue
                seen_hashes.add(sha)

                rec = _make_record(
                    id=str(uuid.uuid4()),
                    split=split,
                    audio_path=fpath,
                    sha256=sha,
                    duration_s=round(duration_s, 6),
                    sample_rate=sample_rate,
                    channels=channels,
                    speaker_id=spk_id,
                    source_dataset="voxceleb1",
                    licence="CC BY 4.0 [VERIFY]",
                    consent_id=None,
                    session_id=video_dir.name,
                    device=None,
                    is_augmented=False,
                    augmentation=None,
                )
                records.append(rec)

    logger.info(
        "VoxCeleb1 split=%s: %d included, %d excluded.",
        split, len(records), len(excluded),
    )
    _write_jsonl(records, output_path)
    _write_exclusion_report(excluded, output_path)


# ---------------------------------------------------------------------------
# Common Voice manifest builder
# ---------------------------------------------------------------------------
def build_commonvoice_manifest(
    cv_root: str,
    split: str,
    client_ids: Set[str],
    output_path: str,
) -> None:
    """
    Build a JSONL manifest from Mozilla Common Voice (validated.tsv).

    Filtering:
      • Only client_ids in the provided set.
      • Clients with fewer than 6 clips are excluded.
      • Duration filter [1.0, 10.0] s applied.

    Licence note:
      "CC0+CC BY 4.0 [VERIFY biometric consent]" — verify the actual
      licence for your CV version. Biometric consent is PENDING legal
      review; consent_id is set to null and a note is added.

    Args:
        cv_root:     Root directory of the Common Voice corpus
                     (must contain ``validated.tsv`` and ``clips/``).
        split:       Split label (e.g. 'train', 'dev').
        client_ids:  Set of client_id strings to include.
        output_path: Destination JSONL file.
    """
    cv_root_p = Path(cv_root)
    tsv_path = cv_root_p / "validated.tsv"
    clips_dir = cv_root_p / "clips"

    if not tsv_path.is_file():
        raise FileNotFoundError(f"validated.tsv not found: {tsv_path}")
    if not clips_dir.is_dir():
        raise NotADirectoryError(f"clips/ not found: {clips_dir}")

    # Read TSV
    rows_by_client: Dict[str, List[Dict]] = {}
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row.get("client_id", "").strip()
            if cid not in client_ids:
                continue
            rows_by_client.setdefault(cid, []).append(row)

    # Apply ≥6 clips filter
    excluded_clients = {
        cid for cid, rows in rows_by_client.items() if len(rows) < 6
    }
    if excluded_clients:
        logger.info(
            "CommonVoice: excluding %d clients with <6 clips.", len(excluded_clients)
        )
    rows_by_client = {
        cid: rows for cid, rows in rows_by_client.items()
        if cid not in excluded_clients
    }

    records: List[Dict] = []
    excluded: List[Dict] = []
    seen_hashes: Set[str] = set()

    for cid, rows in rows_by_client.items():
        for row in rows:
            fname = row.get("path", "").strip()
            if not fname:
                excluded.append({"client_id": cid, "reason": "missing_path"})
                continue
            fpath = str((clips_dir / fname).resolve())
            if not os.path.isfile(fpath):
                excluded.append({"path": fpath, "reason": "file_not_found"})
                continue

            info = _audio_info(fpath)
            if info is None:
                excluded.append({"path": fpath, "reason": "torchaudio.info_error"})
                continue
            duration_s, sample_rate, channels = info

            if not (DURATION_MIN_S <= duration_s <= DURATION_MAX_S):
                excluded.append({
                    "path": fpath,
                    "reason": f"duration_oob:{duration_s:.3f}s",
                })
                continue

            sha = _sha256_file(fpath)
            if sha in seen_hashes:
                excluded.append({"path": fpath, "reason": f"sha256_duplicate:{sha[:12]}"})
                continue
            seen_hashes.add(sha)

            rec = _make_record(
                id=str(uuid.uuid4()),
                split=split,
                audio_path=fpath,
                sha256=sha,
                duration_s=round(duration_s, 6),
                sample_rate=sample_rate,
                channels=channels,
                speaker_id=cid,
                source_dataset="commonvoice",
                # Biometric consent is pending legal review for CV audio used in
                # speaker verification. Update this field once legal sign-off received.
                licence="CC0+CC BY 4.0 [VERIFY biometric consent]",
                consent_id=None,
                session_id=None,
                device=None,
                is_augmented=False,
                augmentation=None,
            )
            # Add consent note as extra field
            rec["_note"] = "consent_pending_legal_review"
            records.append(rec)

    logger.info(
        "CommonVoice split=%s: %d included, %d excluded.",
        split, len(records), len(excluded),
    )
    _write_jsonl(records, output_path)
    _write_exclusion_report(excluded, output_path)


# ---------------------------------------------------------------------------
# In-house manifest builder
# ---------------------------------------------------------------------------
def build_inhouse_manifest(
    recordings_dir: str,
    metadata_csv: str,
    partition_map: Dict[str, str],
    output_path: str,
) -> None:
    """
    Build a JSONL manifest from in-house recordings.

    The metadata_csv must conform to the RECORDING_PROTOCOL.md schema.
    Required columns: speaker_id, file_name, consent_id.
    Optional columns: session_id, device, notes.

    Processing:
      • Rows with null/empty consent_id are REJECTED — counted and logged.
      • Split is assigned from partition_map {speaker_id → split}.
      • Speakers not in partition_map are assigned split='unpartitioned'.
      • Duration filter [1.0, 10.0] s applied.
      • SHA256 deduplication applied.

    Args:
        recordings_dir:  Directory containing recording files.
        metadata_csv:    Path to metadata CSV (see RECORDING_PROTOCOL.md).
        partition_map:   Dict mapping speaker_id → split string.
        output_path:     Destination JSONL file.
    """
    recordings_dir_p = Path(recordings_dir)
    if not recordings_dir_p.is_dir():
        raise NotADirectoryError(f"recordings_dir not found: {recordings_dir}")

    meta_path = Path(metadata_csv)
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")

    records: List[Dict] = []
    excluded: List[Dict] = []
    seen_hashes: Set[str] = set()
    n_no_consent = 0

    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spk_id = row.get("speaker_id", "").strip()
            file_name = row.get("file_name", "").strip()
            consent_id = row.get("consent_id", "").strip() or None

            # Hard reject: no consent
            if consent_id is None:
                n_no_consent += 1
                excluded.append({
                    "speaker_id": spk_id,
                    "file_name": file_name,
                    "reason": "missing_consent_id",
                })
                continue

            if not file_name:
                excluded.append({"speaker_id": spk_id, "reason": "missing_file_name"})
                continue

            fpath = str((recordings_dir_p / file_name).resolve())
            if not os.path.isfile(fpath):
                excluded.append({"path": fpath, "reason": "file_not_found"})
                continue

            info = _audio_info(fpath)
            if info is None:
                excluded.append({"path": fpath, "reason": "torchaudio.info_error"})
                continue
            duration_s, sample_rate, channels = info

            if not (DURATION_MIN_S <= duration_s <= DURATION_MAX_S):
                excluded.append({
                    "path": fpath,
                    "reason": f"duration_oob:{duration_s:.3f}s",
                })
                continue

            sha = _sha256_file(fpath)
            if sha in seen_hashes:
                excluded.append({"path": fpath, "reason": f"sha256_duplicate:{sha[:12]}"})
                continue
            seen_hashes.add(sha)

            split = partition_map.get(spk_id, "unpartitioned")

            rec = _make_record(
                id=str(uuid.uuid4()),
                split=split,
                audio_path=fpath,
                sha256=sha,
                duration_s=round(duration_s, 6),
                sample_rate=sample_rate,
                channels=channels,
                speaker_id=spk_id,
                source_dataset="inhouse",
                licence="proprietary — internal use only",
                consent_id=consent_id,
                session_id=row.get("session_id", "").strip() or None,
                device=row.get("device", "").strip() or None,
                is_augmented=False,
                augmentation=None,
            )
            records.append(rec)

    logger.info(
        "In-house: %d included, %d excluded (%d rejected for missing consent_id).",
        len(records), len(excluded), n_no_consent,
    )
    if n_no_consent:
        logger.warning(
            "[CONSENT] %d rows rejected because consent_id is null. "
            "These recordings MUST NOT be used without documented consent.",
            n_no_consent,
        )
    _write_jsonl(records, output_path)
    _write_exclusion_report(excluded, output_path)


# ---------------------------------------------------------------------------
# Manifest hash
# ---------------------------------------------------------------------------
def compute_manifest_hash(manifest_path: str) -> str:
    """
    Compute sha256 of the full JSONL manifest file content.

    This hash is used to lock the manifest in eval_config.yaml and is
    verified before evaluation runs (see sv_eval.run_evaluation).

    Args:
        manifest_path: Path to the JSONL manifest file.

    Returns:
        Lowercase hex sha256 string.
    """
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    logger.info("Manifest hash: %s  (%s)", digest, manifest_path)
    return digest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_voxceleb(args):
    build_voxceleb_manifest(
        wav_root=args.wav_root,
        split=args.split,
        output_path=args.output,
        seed=args.seed,
    )


def _cli_commonvoice(args):
    # Load client IDs
    client_ids: Set[str] = set()
    if args.client_ids_file:
        with open(args.client_ids_file, "r", encoding="utf-8") as f:
            client_ids = {line.strip() for line in f if line.strip()}
    elif args.client_ids:
        client_ids = set(args.client_ids.split(","))
    else:
        logger.error("Provide --client-ids-file or --client-ids.")
        sys.exit(1)

    build_commonvoice_manifest(
        cv_root=args.cv_root,
        split=args.split,
        client_ids=client_ids,
        output_path=args.output,
    )


def _cli_inhouse(args):
    if args.partition_map_file:
        with open(args.partition_map_file, "r", encoding="utf-8") as f:
            partition_map = json.load(f)
    else:
        partition_map = {}

    build_inhouse_manifest(
        recordings_dir=args.recordings_dir,
        metadata_csv=args.metadata_csv,
        partition_map=partition_map,
        output_path=args.output,
    )


def _cli_hash(args):
    digest = compute_manifest_hash(args.manifest)
    print(digest)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AAA Engine — JSONL Manifest Builder"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # voxceleb subcommand
    p_vc = sub.add_parser("voxceleb", help="Build manifest from VoxCeleb1.")
    p_vc.add_argument("--wav-root", required=True, help="VoxCeleb1 wav root directory.")
    p_vc.add_argument("--split", required=True, help="Split label (train/dev/test).")
    p_vc.add_argument("--output", required=True, help="Output JSONL path.")
    p_vc.add_argument("--seed", type=int, default=42)
    p_vc.set_defaults(func=_cli_voxceleb)

    # commonvoice subcommand
    p_cv = sub.add_parser("commonvoice", help="Build manifest from Common Voice.")
    p_cv.add_argument("--cv-root", required=True, help="Common Voice corpus root.")
    p_cv.add_argument("--split", required=True, help="Split label.")
    p_cv.add_argument("--client-ids-file", default=None, help="File with one client_id per line.")
    p_cv.add_argument("--client-ids", default=None, help="Comma-separated client IDs.")
    p_cv.add_argument("--output", required=True, help="Output JSONL path.")
    p_cv.set_defaults(func=_cli_commonvoice)

    # inhouse subcommand
    p_ih = sub.add_parser("inhouse", help="Build manifest from in-house recordings.")
    p_ih.add_argument("--recordings-dir", required=True)
    p_ih.add_argument("--metadata-csv", required=True)
    p_ih.add_argument(
        "--partition-map-file",
        default=None,
        help="JSON file mapping speaker_id → split.",
    )
    p_ih.add_argument("--output", required=True)
    p_ih.set_defaults(func=_cli_inhouse)

    # hash subcommand
    p_hash = sub.add_parser("hash", help="Compute sha256 of a manifest JSONL file.")
    p_hash.add_argument("--manifest", required=True)
    p_hash.set_defaults(func=_cli_hash)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
