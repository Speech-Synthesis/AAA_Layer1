# =============================================================
# training/check_leakage.py
# AAA Engine — Data Leakage Checker
# =============================================================
# Verifies that speaker identities and audio content hashes do
# NOT cross split boundaries (train / dev / test / cal / etc.).
#
# Three checks:
#   1. Speaker overlap  — same speaker_id in multiple splits.
#   2. Hash overlap     — same sha256 (identical audio) in multiple splits.
#   3. VoxCeleb isolation — VoxCeleb dev speakers must not appear
#      in calibration or in-house test sets.
#
# A FAIL in any check must be investigated before training.
# The script writes a JSON report and exits with code 1 on FAIL.
#
# Usage:
#   python -m aaa_engine.training.check_leakage \
#       --manifest path/to/manifest.jsonl \
#       --voxceleb-ids path/to/voxceleb_dev_ids.txt \
#       --output leakage_report.json
# =============================================================

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest_speakers(path: str) -> Dict[str, Set[str]]:
    """
    Return {split: set_of_speaker_ids} from a JSONL manifest.

    Each JSONL record must have 'split' and 'speaker_id' fields.
    Records missing either field are logged and skipped.

    Args:
        path: Path to the JSONL manifest file.

    Returns:
        Dict mapping split name to the set of unique speaker IDs in that split.
    """
    result: Dict[str, Set[str]] = defaultdict(set)
    n_skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d JSON error: %s", lineno, exc)
                n_skipped += 1
                continue

            split = rec.get("split")
            spk = rec.get("speaker_id")
            if not split or not spk:
                logger.debug("Line %d: missing split or speaker_id — skipped.", lineno)
                n_skipped += 1
                continue
            result[split].add(str(spk))

    if n_skipped:
        logger.warning("Skipped %d malformed/incomplete records.", n_skipped)

    for split, spks in result.items():
        logger.info("Split %-16s : %d unique speakers", f"'{split}'", len(spks))

    return dict(result)


def load_manifest_hashes(path: str) -> Dict[str, Set[str]]:
    """
    Return {split: set_of_sha256s} from a JSONL manifest.

    Args:
        path: Path to the JSONL manifest file.

    Returns:
        Dict mapping split name to the set of sha256 hashes in that split.
    """
    result: Dict[str, Set[str]] = defaultdict(set)
    n_skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d JSON error: %s", lineno, exc)
                n_skipped += 1
                continue

            split = rec.get("split")
            sha = rec.get("sha256")
            if not split or not sha:
                n_skipped += 1
                continue
            result[split].add(str(sha))

    if n_skipped:
        logger.warning("Skipped %d records (missing split or sha256).", n_skipped)

    for split, hashes in result.items():
        logger.info("Split %-16s : %d unique hashes", f"'{split}'", len(hashes))

    return dict(result)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_speaker_overlap(speaker_sets: Dict[str, Set[str]]) -> Dict:
    """
    For every pair of splits, compute speaker ID intersection.

    Args:
        speaker_sets: {split: set_of_speaker_ids}

    Returns:
        Dict keyed by "{split_a}_vs_{split_b}" with:
          overlap_count: int
          result:        "PASS" or "FAIL"
          examples:      list of up to 5 overlapping speaker IDs
    """
    splits = sorted(speaker_sets.keys())
    results = {}

    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            overlap = speaker_sets[split_a] & speaker_sets[split_b]
            key = f"{split_a}_vs_{split_b}"
            status = "PASS" if len(overlap) == 0 else "FAIL"
            if status == "FAIL":
                logger.warning(
                    "[LEAKAGE] Speaker overlap between '%s' and '%s': %d speakers.",
                    split_a, split_b, len(overlap),
                )
            results[key] = {
                "overlap_count": len(overlap),
                "result": status,
                "examples": sorted(overlap)[:5],
            }

    return results


def check_hash_overlap(hash_sets: Dict[str, Set[str]]) -> Dict:
    """
    For every pair of splits, compute sha256 hash intersection.

    Identical hashes mean the same audio file appears in both splits —
    a hard data leakage violation.

    Args:
        hash_sets: {split: set_of_sha256s}

    Returns:
        Dict keyed by "{split_a}_vs_{split_b}" with:
          overlap_count: int
          result:        "PASS" or "FAIL"
          examples:      list of up to 5 overlapping hashes (truncated to 16 chars)
    """
    splits = sorted(hash_sets.keys())
    results = {}

    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            overlap = hash_sets[split_a] & hash_sets[split_b]
            key = f"{split_a}_vs_{split_b}"
            status = "PASS" if len(overlap) == 0 else "FAIL"
            if status == "FAIL":
                logger.warning(
                    "[LEAKAGE] Hash overlap between '%s' and '%s': %d files.",
                    split_a, split_b, len(overlap),
                )
            results[key] = {
                "overlap_count": len(overlap),
                "result": status,
                "examples": [h[:16] + "…" for h in sorted(overlap)[:5]],
            }

    return results


def check_voxceleb_isolation(
    speaker_sets: Dict[str, Set[str]],
    voxceleb_speaker_ids: Set[str],
) -> Dict:
    """
    Verify that VoxCeleb dev/test speaker IDs do not appear in
    calibration ('cal') or in-house test ('test-AAA') splits.

    If either of those splits is absent from speaker_sets, the check
    is skipped and reported as SKIP.

    Args:
        speaker_sets:         {split: set_of_speaker_ids}.
        voxceleb_speaker_ids: Set of speaker IDs from VoxCeleb dev/test
                              that must remain isolated.

    Returns:
        Dict keyed by split name with:
          voxceleb_speakers_found: int
          result:                  "PASS", "FAIL", or "SKIP"
          examples:                list of up to 5 leaking speaker IDs
    """
    results = {}
    protected_splits = ["cal", "test-AAA", "test_aaa", "test"]

    for split in protected_splits:
        if split not in speaker_sets:
            results[split] = {
                "voxceleb_speakers_found": 0,
                "result": "SKIP",
                "examples": [],
                "note": f"Split '{split}' not present in manifest.",
            }
            continue

        overlap = speaker_sets[split] & voxceleb_speaker_ids
        status = "PASS" if len(overlap) == 0 else "FAIL"
        if status == "FAIL":
            logger.warning(
                "[LEAKAGE] VoxCeleb speakers found in protected split '%s': %d.",
                split, len(overlap),
            )
        results[split] = {
            "voxceleb_speakers_found": len(overlap),
            "result": status,
            "examples": sorted(overlap)[:5],
        }

    return results


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_checks(
    manifest_path: str,
    voxceleb_id_file: Optional[str] = None,
) -> Dict:
    """
    Run all leakage checks and return a consolidated report dict.

    Args:
        manifest_path:    Path to JSONL manifest.
        voxceleb_id_file: Optional path to a file with one VoxCeleb
                          dev/test speaker ID per line.

    Returns:
        Dict with:
          overall:          "PASS" or "FAIL"
          timestamp:        ISO 8601
          manifest_path:    str
          checks.speakers:  output of check_speaker_overlap()
          checks.hashes:    output of check_hash_overlap()
          checks.voxceleb:  output of check_voxceleb_isolation() or {"result": "SKIP"}
    """
    logger.info("Loading manifest speakers from: %s", manifest_path)
    speaker_sets = load_manifest_speakers(manifest_path)

    logger.info("Loading manifest hashes from: %s", manifest_path)
    hash_sets = load_manifest_hashes(manifest_path)

    logger.info("Running speaker overlap check …")
    spk_check = check_speaker_overlap(speaker_sets)

    logger.info("Running hash overlap check …")
    hash_check = check_hash_overlap(hash_sets)

    # VoxCeleb isolation
    if voxceleb_id_file:
        logger.info("Loading VoxCeleb speaker IDs from: %s", voxceleb_id_file)
        with open(voxceleb_id_file, "r", encoding="utf-8") as f:
            vc_ids: Set[str] = {line.strip() for line in f if line.strip()}
        logger.info("Loaded %d VoxCeleb speaker IDs.", len(vc_ids))
        logger.info("Running VoxCeleb isolation check …")
        vc_check = check_voxceleb_isolation(speaker_sets, vc_ids)
    else:
        logger.info("No --voxceleb-ids provided — skipping VoxCeleb isolation check.")
        vc_check = {"result": "SKIP", "note": "No voxceleb_id_file provided."}

    # Determine overall pass/fail
    all_results: List[str] = []
    for v in spk_check.values():
        all_results.append(v["result"])
    for v in hash_check.values():
        all_results.append(v["result"])
    if isinstance(vc_check, dict) and "result" in vc_check:
        # Flat dict for single-split case
        if vc_check["result"] not in ("SKIP",):
            all_results.append(vc_check["result"])
    elif isinstance(vc_check, dict):
        for v in vc_check.values():
            if isinstance(v, dict) and v.get("result") not in ("SKIP",):
                all_results.append(v["result"])

    overall = "FAIL" if "FAIL" in all_results else "PASS"
    if overall == "FAIL":
        logger.error("[LEAKAGE] Overall result: FAIL — investigate before training!")
    else:
        logger.info("[OK] Overall leakage check: PASS")

    return {
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "checks": {
            "speakers": spk_check,
            "hashes": hash_check,
            "voxceleb_isolation": vc_check,
        },
    }


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def save_report(results: Dict, output_path: str) -> None:
    """
    Write the leakage check report as a JSON file.

    Args:
        results:     Dict returned by run_checks().
        output_path: Destination JSON file path.
    """
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Report saved: %s  (overall=%s)", output_p, results.get("overall"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "AAA Engine — Data Leakage Checker. "
            "Checks speaker and hash overlap across manifest splits. "
            "Exits with code 1 if any check FAILs."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the JSONL manifest file.",
    )
    parser.add_argument(
        "--voxceleb-ids",
        default=None,
        dest="voxceleb_id_file",
        help=(
            "Path to a file with one VoxCeleb dev/test speaker ID per line. "
            "If omitted, VoxCeleb isolation check is skipped."
        ),
    )
    parser.add_argument(
        "--output",
        default="leakage_report.json",
        help="Destination for the JSON report (default: leakage_report.json).",
    )
    args = parser.parse_args(argv)

    results = run_checks(
        manifest_path=args.manifest,
        voxceleb_id_file=args.voxceleb_id_file,
    )
    save_report(results, args.output)

    if results["overall"] == "FAIL":
        print(
            "\n[LEAKAGE] FAIL — data leakage detected. "
            "See report for details. Do NOT proceed with training.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"\n[OK] Leakage check PASS — report saved to {args.output}")


if __name__ == "__main__":
    main()
