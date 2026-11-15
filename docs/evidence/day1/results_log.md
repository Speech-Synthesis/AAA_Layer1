# Results Log — AAA-SV B1 Fine-Tune

> [!IMPORTANT]
> No metric is ever written here until a run log, config hash, manifest hash,
> and checkpoint hash all exist. All entries are TBD until verified.

## Column Definitions

| Column | Description |
|---|---|
| `run_id` | `YYYYMMDD-HHMM_<gitsha7>_<cfghash6>` |
| `config_hash` | First 6 hex chars of SHA-256 of YAML config |
| `manifest_hash` | SHA-256 of `train.csv` used in this run |
| `checkpoint_hash` | SHA-256 of `checkpoint_best.pt` |
| `data` | Dataset description |
| `aug` | Augmentation summary |
| `schedule` | LR schedule summary |
| `seed` | Master RNG seed |
| `val_eer` | Validation EER (%) — only from validation split |
| `val_frr_at_far1` | FRR@FAR1% — only from validation split |
| `verdict` | PASS / FAIL / TBD / SUPERSEDED |

## Runs

| run_id | config_hash | manifest_hash | checkpoint_hash | data | aug | schedule | seed | val_eer | val_frr_at_far1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261111-1918_9eef8dc_6596ea | 6596ea | 27189a59bc12659472c9ce9ac3fbcbc612991608269c1ee9724ff43de00f63e5 | f479378f16472e1a7a6589ac6537eb2a617d5954c5a52f0f2c8d00a17831ee6c | synthetic 20spk/420utt (smoke) | resample+bandlimit (no MUSAN/RIR/codec, ffmpeg not found) | AdamW cosine 50wu | 1986 | n/a* | n/a* | SUPERSEDED BY 20261112-0758_ca0744b_6596ea |
| 20261112-0758_ca0744b_6596ea | 6596ea | 27189a59bc12659472c9ce9ac3fbcbc612991608269c1ee9724ff43de00f63e5 | de1e6523e4f1a4c923f9c0ee7774780d3183db5248133c4da923310b79d77cb0 | synthetic 20spk/420utt (clean smoke) | resample+bandlimit (no MUSAN/RIR/codec, ffmpeg not found) | AdamW cosine 50wu | 1986 | n/a* | n/a* | SMOKE OK (clean tree) |
| B0_vox1o_zero_shot | TBD | TBD | [BASELINE] stock ECAPA | VoxCeleb1-O cleaned | None | Zero-shot | None | not yet | not yet | NOT YET RUN |
| 93476.amritahpc (B1) | TBD | 1a3c010b39da65d63701dfa3ef8307d64690f09cf6e3b256eb5b4b968942bc3f | in-progress (PBS queued) | VoxCeleb1 dev (1,211 spk / 112,470 train / 2,422 val) | resample+bandlimit | AdamW cosine 500wu | 1986 | in-progress | in-progress | PBS QUEUED (93476.amritahpc) |

## Notes

1. **Run 20261111-1918_9eef8dc_6596ea Status (SUPERSEDED):**  
   Executed on commit `9eef8dc` with uncommitted working-tree changes recorded in `git_diff.txt` (ECAPA-TDNN 3D output squeeze fix in `train_b1.py` and `test_pairs.csv` absolute paths). The squeeze fix was subsequently committed in `4d9b239`. Superseded by clean run `20261112-0758_ca0744b_6596ea`.
2. **Run 20261112-0758_ca0744b_6596ea Status (SMOKE OK):**  
   Executed on clean commit `ca0744b` (HEAD). `git_diff.txt` is 0 bytes (empty). Checkpoint hashes verified:  
   - `checkpoint_best.pt` SHA-256: `de1e6523e4f1a4c923f9c0ee7774780d3183db5248133c4da923310b79d77cb0`  
   - `checkpoint_last.pt` SHA-256: `d03262b4af13d7ec6b422344515a9a57e0c8507ac6818cb097756a72af0e37cc`  
   - Manifest SHA-256: `27189a59bc12659472c9ce9ac3fbcbc612991608269c1ee9724ff43de00f63e5`  
3. Smoke run on synthetic dataset: metrics are not meaningful for real SV performance.
   Result exists only to confirm the pipeline runs end-to-end.
4. *n/a = val EER not computable: dev set too small for trial-pair generation
   ("No trial pairs could be built" — expected on 4 clips/speaker synthetic data).
   val_EER fallback=1.0 is NOT a real measurement. See `baseline_smoke_result.md`.
5. Full VoxCeleb1-dev run will be executed on HPC V100. Metrics from that run will be
   recorded here with full provenance.
6. Calibration data is never used for model selection.
7. Test data (VoxCeleb1-O cleaned) is locked until Day 3 freeze.
8. VoxCeleb dev is never used in cal/test-AAA splits.
9. Failed/negative runs are kept in this log — never deleted.
