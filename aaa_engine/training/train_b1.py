# =============================================================
# training/train_b1.py
# AAA Engine – B1 Fine-Tune Script
# [FINE-TUNED] ECAPA-TDNN fine-tune with AAM-Softmax head
# =============================================================
#
# Usage:
#   python -m training.train_b1 --config configs/train_local_smoke.yaml
#   python -m training.train_b1 --config configs/train_hpc.yaml \
#       --override seed=42 --override batch_size=16
#
# Run ID format:  YYYYMMDD-HHMM_<gitsha7>_<cfghash6>
# Resume:         automatic — checks for checkpoint_last.pt at startup
# =============================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ── Import from project ──────────────────────────────────────
# evaluate.py lives in the same package
from .evaluate import compute_eer
from .augment import Augmenter, check_ffmpeg_codecs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 0.  Audio I/O helper (soundfile primary, torchaudio fallback)
# ═══════════════════════════════════════════════════════════════

def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and return (waveform, sample_rate).

    waveform shape: (n_channels, n_samples), dtype=float32.

    Uses soundfile as primary loader (no torchcodec dependency on
    torchaudio>=2.5 on Windows) with torchaudio as fallback.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile: (n_samples, n_channels) → (n_channels, n_samples)
        waveform = torch.from_numpy(data.T)
        return waveform, sr
    except Exception:
        waveform, sr = torchaudio.load(path)
        return waveform, sr

def _git_sha7() -> str:
    """Return the first 7 chars of the current git commit SHA, or 'nogit'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def _cfg_hash6(yaml_content: str) -> str:
    """First 6 hex chars of SHA-256 of the raw YAML config string."""
    return hashlib.sha256(yaml_content.encode()).hexdigest()[:6]


def make_run_id(yaml_content: str) -> str:
    """YYYYMMDD-HHMM_<gitsha7>_<cfghash6>"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return f"{ts}_{_git_sha7()}_{_cfg_hash6(yaml_content)}"


def _manifest_hash(manifest_path: str) -> str:
    """SHA-256 of the manifest file for provenance tracking."""
    h = hashlib.sha256()
    with open(manifest_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
# 2.  Seeding
# ═══════════════════════════════════════════════════════════════

def set_seeds(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("[seed] All RNGs seeded to %d", seed)


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently but deterministically."""
    base = torch.initial_seed() % (2**31)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ═══════════════════════════════════════════════════════════════
# 3.  Checkpoint helpers
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Atomic checkpoint write.
    Writes to a .tmp file first, then os.replace() for atomicity.
    On Windows, os.replace is best-effort atomic.
    """
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state_dict() -> Dict[str, Any]:
    """Capture full RNG state for exact resume."""
    return {
        "torch":  torch.get_rng_state(),
        "numpy":  np.random.get_state(),
        "python": random.getstate(),
        "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from a checkpoint."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


# ═══════════════════════════════════════════════════════════════
# 4.  AAM-Softmax loss
# ═══════════════════════════════════════════════════════════════

class AAMSoftmax(nn.Module):
    """
    [CUSTOM] Additive Angular Margin Softmax (ArcFace) loss head.

    Parameters
    ----------
    in_features  : embedding dimension (192 for ECAPA)
    out_features : number of speaker classes
    margin       : angular margin m (radians); ramped during training
    scale        : feature scale s
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale  = scale

        # Class-centre weight matrix — [CUSTOM]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def set_margin(self, margin: float) -> None:
        """Update margin (called each step for margin ramp)."""
        self.margin = margin
        self.cos_m  = math.cos(margin)
        self.sin_m  = math.sin(margin)
        self.th     = math.cos(math.pi - margin)
        self.mm     = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2-normalise weights and embeddings
        W   = F.normalize(self.weight, dim=1)
        emb = F.normalize(embeddings, dim=1)

        # Cosine similarity: (B, C)
        cosine = emb @ W.T

        # ArcFace angular margin
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-9, 1.0))
        phi    = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


# ═══════════════════════════════════════════════════════════════
# 5.  Dataset
# ═══════════════════════════════════════════════════════════════

class SpeakerDataset(torch.utils.data.Dataset):
    """
    Reads a CSV manifest with columns: speaker_id, wav_path.
    Randomly crops each waveform to crop_len seconds.
    crop_len is chosen per batch from hparams['crop_lengths'].
    """

    def __init__(
        self,
        manifest_path: str,
        data_folder: str,
        augmenter: Optional[Augmenter],
        speaker_to_idx: Dict[str, int],
        crop_len_seconds: float = 2.0,
        sample_rate: int = 16000,
        shuffle_labels: bool = False,
        seed: int = 0,
    ) -> None:
        self.data_folder      = data_folder
        self.augmenter        = augmenter
        self.speaker_to_idx   = speaker_to_idx
        self.crop_len_seconds = crop_len_seconds
        self.sample_rate      = sample_rate

        self.samples: List[Tuple[str, str]] = []  # (wav_path, speaker_id)
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both schema: new (wav_path/speaker_id) and
                # legacy SpeechBrain CSV (wav/spk_id, {data_root} prefix)
                wp = row.get("wav_path", row.get("wav", ""))
                wp = wp.replace("{data_root}/", "").replace("{data_root}\\", "")
                sp = row.get("speaker_id", row.get("spk_id", ""))
                if wp and sp:
                    self.samples.append((wp, sp))

        if shuffle_labels:
            # [EXPERIMENT] Shuffled labels — sanity test only; must fail to train
            logger.warning("[EXPERIMENT] debug_shuffle_labels=True — this run should NOT converge!")
            rng = random.Random(seed)
            labels = [s[1] for s in self.samples]
            rng.shuffle(labels)
            self.samples = [(wav, lbl) for (wav, _), lbl in zip(self.samples, labels)]

    def set_crop_len(self, crop_len_seconds: float) -> None:
        """Update crop length (called per batch from the training loop)."""
        self.crop_len_seconds = crop_len_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav_path, speaker_id = self.samples[idx]

        # Resolve relative paths against data_folder
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(self.data_folder, wav_path)

        waveform, sr = _load_audio(wav_path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Random crop — if clip is shorter, use full clip (no padding)
        crop_samples = int(self.crop_len_seconds * self.sample_rate)
        if waveform.shape[-1] > crop_samples:
            start = random.randint(0, waveform.shape[-1] - crop_samples)
            waveform = waveform[:, start : start + crop_samples]

        # Augment
        if self.augmenter is not None:
            waveform = self.augmenter.augment(waveform, self.sample_rate)

        label = self.speaker_to_idx[speaker_id]
        return waveform.squeeze(0), label


# ═══════════════════════════════════════════════════════════════
# 6.  LR scheduler — cosine with linear warmup
# ═══════════════════════════════════════════════════════════════

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup for `warmup_steps`, then cosine decay to 0.
    Works at the step (not epoch) level.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


# ═══════════════════════════════════════════════════════════════
# 7.  Model helpers
# ═══════════════════════════════════════════════════════════════

def load_pretrained_ecapa(hparams: Dict[str, Any]) -> nn.Module:
    """
    [BASELINE] Load pretrained ECAPA-TDNN weights.

    Strategy (Windows-safe):
      1. First try loading the .ckpt directly from the HF cache snapshot
         via torch.load() — avoids SpeechBrain's symlink creation which
         requires elevated privilege on Windows (WinError 1314).
      2. Fall back to SpeechBrain.from_hparams() if the direct path is
         not available (HPC / Linux where symlinks work fine).

    Returns (embedding_model, pretrained_sb_or_None).
    pretrained_sb_or_None is None when the direct-load path is used.
    """
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    logger.info("[BASELINE] Loading pretrained ECAPA-TDNN from: %s", hparams["pretrained_path"])

    # Build the ECAPA-TDNN architecture matching speechbrain/spkrec-ecapa-voxceleb
    embedding_model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        groups=[1, 1, 1, 1, 1],
        attention_channels=128,
        lin_neurons=192,
    )

    # ── Path 1: direct .ckpt load (Windows-safe, no symlink required) ──
    direct_ckpt = hparams.get("pretrained_ckpt_direct")
    if direct_ckpt:
        direct_ckpt = os.path.expanduser(str(direct_ckpt))
    if direct_ckpt and os.path.isfile(direct_ckpt):
        logger.info("[BASELINE] Direct ckpt load (Windows-safe): %s", direct_ckpt)
        state = torch.load(direct_ckpt, map_location="cpu", weights_only=False)
        # SpeechBrain .ckpt files store state_dict under key '' or directly
        if isinstance(state, dict) and "" in state:
            state = state[""]
        embedding_model.load_state_dict(state)
        logger.info("[BASELINE] Pretrained ECAPA-TDNN weights loaded (192-dim). [direct ckpt]")
        return embedding_model, None

    # ── Path 2: SpeechBrain from_hparams (Linux / HPC) ──────────────────
    from speechbrain.inference.speaker import SpeakerRecognition

    pretrained = SpeakerRecognition.from_hparams(
        source=hparams["pretrained_path"],
        savedir=os.path.expanduser(hparams["pretrained_savedir"]),
        run_opts={"device": "cpu"},
    )
    embedding_model.load_state_dict(pretrained.mods.embedding_model.state_dict())
    logger.info("[BASELINE] Pretrained ECAPA-TDNN weights loaded (192-dim). [from_hparams]")
    return embedding_model, pretrained


def build_feature_pipeline():
    """
    [BASELINE] Return (Fbank, InputNormalization) matching pretrained config.
    """
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    compute_features = Fbank(n_mels=80)
    mean_var_norm    = InputNormalization(norm_type="sentence", std_norm=False)
    return compute_features, mean_var_norm


@torch.no_grad()
def init_class_centres(
    embedding_model: nn.Module,
    compute_features,
    mean_var_norm,
    classifier: AAMSoftmax,
    dataset: SpeakerDataset,
    speaker_to_idx: Dict[str, int],
    max_per_speaker: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    [EXPERIMENT] Initialise AAM-Softmax class centres from mean embeddings.

    For each speaker, extracts up to max_per_speaker embeddings from the
    training set, averages them, L2-normalises, and sets the corresponding
    weight row in the classifier.  Falls back to random-normal if no data
    is available for a speaker (logged clearly).
    """
    logger.info("[EXPERIMENT] Initialising class centres from reference embeddings …")
    embedding_model.eval()
    embedding_model.to(device)

    idx_to_embs: Dict[int, List[torch.Tensor]] = {i: [] for i in speaker_to_idx.values()}

    for wav_path, speaker_id in dataset.samples:
        spk_idx = speaker_to_idx[speaker_id]
        if len(idx_to_embs[spk_idx]) >= max_per_speaker:
            continue

        full_path = wav_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(dataset.data_folder, wav_path)

        try:
            waveform, sr = _load_audio(full_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)

            feats = compute_features(waveform.to(device))
            lens  = torch.tensor([1.0], device=device)
            feats = mean_var_norm(feats, lens)
            emb   = embedding_model(feats)
            emb   = F.normalize(emb.squeeze(), dim=0)
            idx_to_embs[spk_idx].append(emb.cpu())
        except Exception as exc:  # noqa: BLE001
            logger.debug("[EXPERIMENT] Skipping %s during centre init: %s", wav_path, exc)

    with torch.no_grad():
        for idx, embs in idx_to_embs.items():
            if embs:
                centre = torch.stack(embs).mean(0)
                centre = F.normalize(centre, dim=0)
                classifier.weight[idx].copy_(centre)
            else:
                # Random-normal fallback
                nn.init.normal_(classifier.weight[idx : idx + 1])
                logger.warning(
                    "[EXPERIMENT] Speaker idx=%d has no reference — using random-normal centre.",
                    idx,
                )

    logger.info("[EXPERIMENT] Class centre initialisation complete.")


# ═══════════════════════════════════════════════════════════════
# 8.  Validation — EER on val split
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    embedding_model: nn.Module,
    compute_features,
    mean_var_norm,
    val_manifest: str,
    data_folder: str,
    device: torch.device,
    sample_rate: int = 16000,
    max_pairs: int = 5000,
) -> Tuple[float, float, float]:
    """
    Compute val EER and FRR@FAR1%.

    1. Extract embeddings for all val utterances.
    2. Build same-speaker (+) and cross-speaker (-) trial pairs.
    3. Compute cosine similarity and EER via compute_eer().

    Returns
    -------
    (eer, frr_at_far1, threshold)
    """
    embedding_model.eval()

    # Load val manifest
    samples: List[Tuple[str, str]] = []
    with open(val_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wp = row.get("wav_path", row.get("wav", ""))
        wp = wp.replace("{data_root}/", "").replace("{data_root}\\", "")
        sp = row.get("speaker_id", row.get("spk_id", ""))
        if wp and sp:
            samples.append((wp, sp))

    if not samples:
        logger.warning("[val] Val manifest is empty — skipping evaluation.")
        return 1.0, 1.0, 0.0

    # Group by speaker
    spk_to_wavs: Dict[str, List[str]] = {}
    for wav_path, spk in samples:
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(data_folder, wav_path)
        spk_to_wavs.setdefault(spk, []).append(wav_path)

    # Extract embeddings (cached)
    emb_cache: Dict[str, torch.Tensor] = {}

    def _get_emb(path: str) -> Optional[torch.Tensor]:
        if path in emb_cache:
            return emb_cache[path]
        try:
            waveform, sr = _load_audio(path)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            feats = compute_features(waveform.to(device))
            lens  = torch.tensor([1.0], device=device)
            feats = mean_var_norm(feats, lens)
            emb   = embedding_model(feats)
            emb   = F.normalize(emb.squeeze(), dim=0)
            emb_cache[path] = emb.cpu()
            return emb_cache[path]
        except Exception as exc:  # noqa: BLE001
            logger.debug("[val] Could not embed %s: %s", path, exc)
            return None

    for _, wavs in spk_to_wavs.items():
        for w in wavs:
            _get_emb(w)

    # Build trial pairs
    speakers = list(spk_to_wavs.keys())
    scores, labels = [], []
    rng = random.Random(42)

    # Same-speaker pairs (+)
    for spk, wavs in spk_to_wavs.items():
        if len(wavs) < 2:
            continue
        for i in range(min(len(wavs) - 1, 10)):
            w1, w2 = rng.sample(wavs, 2)
            e1, e2 = _get_emb(w1), _get_emb(w2)
            if e1 is not None and e2 is not None:
                scores.append(float(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()))
                labels.append(1)

    # Cross-speaker pairs (-)
    n_cross = len(scores) * 3
    for _ in range(n_cross):
        if len(speakers) < 2:
            break
        s1, s2 = rng.sample(speakers, 2)
        w1 = rng.choice(spk_to_wavs[s1])
        w2 = rng.choice(spk_to_wavs[s2])
        e1, e2 = _get_emb(w1), _get_emb(w2)
        if e1 is not None and e2 is not None:
            scores.append(float(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()))
            labels.append(0)

    if not scores:
        logger.warning("[val] No trial pairs could be built — skipping EER computation.")
        return 1.0, 1.0, 0.0

    eer, threshold = compute_eer(scores, labels)

    # FRR@FAR1%
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n_neg = (labels_arr == 0).sum()
    n_pos = (labels_arr == 1).sum()
    frr_at_far1 = 1.0
    if n_neg > 0 and n_pos > 0:
        for thr in np.sort(scores_arr)[::-1]:
            far = ((scores_arr >= thr) & (labels_arr == 0)).sum() / n_neg
            frr = ((scores_arr <  thr) & (labels_arr == 1)).sum() / n_pos
            if far <= 0.01:
                frr_at_far1 = float(frr)
                break

    return float(eer), float(frr_at_far1), float(threshold)


# ═══════════════════════════════════════════════════════════════
# 9.  Run-directory setup
# ═══════════════════════════════════════════════════════════════

def setup_run_dir(
    run_id: str,
    hparams: Dict[str, Any],
    yaml_content: str,
    runs_dir: str,
    git_sha: str,
    manifest_path: str,
) -> str:
    """Create run directory and write provenance files."""
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Copy config
    with open(os.path.join(run_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # pip freeze
    try:
        pip_out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).decode()
    except Exception:  # noqa: BLE001
        pip_out = "(pip freeze failed)"
    with open(os.path.join(run_dir, "pip_freeze.txt"), "w", encoding="utf-8") as f:
        f.write(pip_out)

    # git diff
    try:
        diff_out = subprocess.check_output(
            ["git", "diff", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).decode()
    except Exception:  # noqa: BLE001
        diff_out = "(git diff failed)"
    with open(os.path.join(run_dir, "git_diff.txt"), "w", encoding="utf-8") as f:
        f.write(diff_out)

    # GPU info
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(f"GPU {i}: {props.name}, {props.total_memory // 1024**2} MB")
    else:
        gpu_info.append("No CUDA devices available.")
    with open(os.path.join(run_dir, "gpu_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(gpu_info))

    # Seed
    with open(os.path.join(run_dir, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(str(hparams["seed"]))

    # Manifest hash
    mhash = _manifest_hash(manifest_path) if os.path.exists(manifest_path) else "N/A"
    with open(os.path.join(run_dir, "manifest_hash.txt"), "w", encoding="utf-8") as f:
        f.write(mhash)

    logger.info("[run] Run directory: %s", run_dir)
    return run_dir


# ═══════════════════════════════════════════════════════════════
# 10.  Config loading
# ═══════════════════════════════════════════════════════════════

def load_hparams(config_path: str, overrides: List[str]) -> Tuple[Dict[str, Any], str]:
    """
    Load hparams from YAML via hyperpyyaml.
    Overrides are applied as 'key=val' strings.
    Returns (hparams_dict, raw_yaml_content).
    """
    import hyperpyyaml

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    # Build override dict from 'key=val' strings
    override_dict = {}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must be in 'key=val' format.")
        k, v = ov.split("=", 1)
        override_dict[k.strip()] = v.strip()

    hparams = hyperpyyaml.load_hyperpyyaml(
        yaml_content,
        overrides=override_dict or None,
    )

    # Apply defaults for optional fields
    defaults = {
        "backbone_lr":        1e-4,
        "head_lr":            1e-3,
        "warmup_steps":       500,
        "aam_margin":         0.2,
        "aam_scale":          30.0,
        "grad_clip":          5.0,
        "grad_accum_steps":   1,
        "patience":           5,
        "log_every_n_steps":  50,
        "num_workers":        0,
        "seed":               42,
        "debug_shuffle_labels": False,
        "crop_lengths":       [1, 2, 3, 4],
        "runs_dir":           "runs",
        "musan_dir":          None,
        "rirs_dir":           None,
    }
    for k, v in defaults.items():
        hparams.setdefault(k, v)

    return hparams, yaml_content


# ═══════════════════════════════════════════════════════════════
# 11.  Main training loop
# ═══════════════════════════════════════════════════════════════

def train(config_path: str, overrides: List[str]) -> None:
    # ── Load config ──────────────────────────────────────────
    hparams, yaml_content = load_hparams(config_path, overrides)
    run_id = make_run_id(yaml_content)
    git_sha = _git_sha7()

    logger.info("=" * 65)
    logger.info("  AAA Engine B1 Training — Run ID: %s", run_id)
    logger.info("=" * 65)

    # ── Seed ────────────────────────────────────────────────
    set_seeds(hparams["seed"])

    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[device] Using: %s", device)

    # ── Paths ───────────────────────────────────────────────
    runs_dir     = hparams["runs_dir"]
    data_folder  = hparams["data_folder"]
    train_manifest = hparams.get("train_manifest",
                        os.path.join(data_folder, "train.csv"))
    val_manifest   = hparams.get("val_manifest",
                        os.path.join(data_folder, "val.csv"))

    # ── Run directory and provenance ─────────────────────────
    run_dir = setup_run_dir(
        run_id, hparams, yaml_content, runs_dir, git_sha, train_manifest
    )
    ckpt_last = os.path.join(run_dir, "checkpoint_last.pt")
    ckpt_best = os.path.join(run_dir, "checkpoint_best.pt")

    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    metrics_csv   = os.path.join(run_dir, "metrics.csv")

    # Initialise CSV header
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "epoch", "train_loss",
                "val_eer", "val_frr_at_far1", "lr_backbone", "lr_head",
            ])

    # ── Speaker index ────────────────────────────────────────
    all_speakers: List[str] = []
    with open(train_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = row.get("speaker_id", row.get("spk_id", ""))
            if sp and sp not in all_speakers:
                all_speakers.append(sp)
    all_speakers.sort()
    speaker_to_idx = {spk: i for i, spk in enumerate(all_speakers)}
    n_speakers = len(speaker_to_idx)
    logger.info("[data] Training speakers: %d", n_speakers)

    # ── Augmenter ────────────────────────────────────────────
    ffmpeg_codecs = check_ffmpeg_codecs()
    augmenter = Augmenter(
        seed=hparams["seed"],
        musan_dir=hparams.get("musan_dir"),
        rirs_dir=hparams.get("rirs_dir"),
        ffmpeg_available=bool(ffmpeg_codecs),
    )

    # ── Dataset & DataLoader ─────────────────────────────────
    train_dataset = SpeakerDataset(
        manifest_path=train_manifest,
        data_folder=data_folder,
        augmenter=augmenter,
        speaker_to_idx=speaker_to_idx,
        crop_len_seconds=float(random.choice(hparams["crop_lengths"])),
        sample_rate=16000,
        shuffle_labels=hparams["debug_shuffle_labels"],
        seed=hparams["seed"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=_pad_collate,
    )

    # ── Models ──────────────────────────────────────────────
    embedding_model, pretrained_sb = load_pretrained_ecapa(hparams)
    compute_features, mean_var_norm = build_feature_pipeline()
    embedding_model.to(device)
    compute_features.to(device)
    mean_var_norm.to(device)

    # ── AAM-Softmax head ─────────────────────────────────────
    classifier = AAMSoftmax(
        in_features=192,
        out_features=n_speakers,
        margin=hparams["aam_margin"],
        scale=hparams["aam_scale"],
    ).to(device)

    # Initialise class centres from pretrained model
    init_class_centres(
        embedding_model, compute_features, mean_var_norm,
        classifier, train_dataset, speaker_to_idx, device=device,
    )

    # ── Freeze backbone for epoch 1 ──────────────────────────
    for p in embedding_model.parameters():
        p.requires_grad_(False)
    logger.info("[train] Backbone frozen for epoch 1.")

    # ── Optimizer ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_model.parameters(), "lr": hparams["backbone_lr"],
             "weight_decay": 1e-2},
            {"params": classifier.parameters(),      "lr": hparams["head_lr"],
             "weight_decay": 1e-2},
        ]
    )

    # ── Scheduler ────────────────────────────────────────────
    total_steps = (
        len(train_loader)
        // hparams["grad_accum_steps"]
        * hparams["number_of_epochs"]
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=hparams["warmup_steps"],
        total_steps=total_steps,
    )

    # ── Mixed precision ──────────────────────────────────────
    use_amp = device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        # No-op scaler shim for CPU
        scaler = _NoOpScaler()

    # ── Resume ──────────────────────────────────────────────
    start_epoch  = 1
    global_step  = 0
    best_eer     = float("inf")
    patience_ctr = 0

    if os.path.exists(ckpt_last):
        try:
            ckpt = torch.load(ckpt_last, map_location=device)
            embedding_model.load_state_dict(ckpt["model"])
            classifier.load_state_dict(ckpt["classifier"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if use_amp and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_eer    = ckpt.get("best_eer", float("inf"))
            restore_rng_state(ckpt["rng_state"])
            logger.info(
                "[RESUME] Resuming from epoch %d step %d",
                ckpt["epoch"], global_step,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RESUME] Checkpoint load failed (%s) — starting fresh.", exc)

    # ── Training loop ────────────────────────────────────────
    log_every = hparams["log_every_n_steps"]

    for epoch in range(start_epoch, hparams["number_of_epochs"] + 1):

        # Unfreeze backbone from epoch 2 onwards
        if epoch == 2:
            for p in embedding_model.parameters():
                p.requires_grad_(True)
            logger.info("[train] Backbone unfrozen at epoch 2.")

        # Random crop length for this epoch
        crop_len = float(random.choice(hparams["crop_lengths"]))
        train_dataset.set_crop_len(crop_len)
        logger.info("[epoch %d] crop_len=%.1f s, %d batches", epoch, crop_len, len(train_loader))

        embedding_model.train()
        classifier.train()

        epoch_loss = 0.0
        micro_step = 0
        optimizer.zero_grad()

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels    = labels.to(device)

            # Margin ramp
            current_margin = min(
                hparams["aam_margin"],
                hparams["aam_margin"] * global_step / max(1, 0.2 * total_steps),
            )
            classifier.set_margin(current_margin)

            with (torch.cuda.amp.autocast() if use_amp else _null_ctx()):
                # Feature extraction
                feats = compute_features(waveforms)
                lens  = torch.ones(waveforms.shape[0], device=device)
                feats = mean_var_norm(feats, lens)

                # Forward
                embeddings = embedding_model(feats)
                # ECAPA-TDNN returns (B, 1, 192) — squeeze to (B, 192)
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(1)
                loss = classifier(embeddings, labels)
                loss = loss / hparams["grad_accum_steps"]

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * hparams["grad_accum_steps"]
            micro_step += 1

            # Gradient accumulation step
            if micro_step % hparams["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(embedding_model.parameters()) + list(classifier.parameters()),
                    hparams["grad_clip"],
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_every == 0:
                    lr_bb  = optimizer.param_groups[0]["lr"]
                    lr_hd  = optimizer.param_groups[1]["lr"]
                    record = {
                        "run_id":      run_id,
                        "epoch":       epoch,
                        "global_step": global_step,
                        "loss":        round(loss.item() * hparams["grad_accum_steps"], 6),
                        "margin":      round(current_margin, 6),
                        "lr_backbone": lr_bb,
                        "lr_head":     lr_hd,
                        "grad_norm":   round(float(grad_norm), 6),
                    }
                    with open(metrics_jsonl, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(record) + "\n")

                    logger.info(
                        "[%d/%d] step=%d loss=%.4f margin=%.3f lr_bb=%.2e lr_hd=%.2e gn=%.2f",
                        epoch, hparams["number_of_epochs"],
                        global_step, record["loss"], current_margin,
                        lr_bb, lr_hd, float(grad_norm),
                    )

        avg_loss = epoch_loss / max(1, len(train_loader))

        # ── Validation ──────────────────────────────────────
        logger.info("[epoch %d] Running validation …", epoch)
        val_eer, val_frr1, _thr = validate(
            embedding_model, compute_features, mean_var_norm,
            val_manifest, data_folder, device,
        )
        logger.info(
            "[epoch %d] val_EER=%.4f  val_FRR@FAR1%%=%.4f",
            epoch, val_eer, val_frr1,
        )

        # CSV epoch row
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id, epoch, round(avg_loss, 6),
                round(val_eer, 6), round(val_frr1, 6),
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
            ])

        # ── Checkpoint ──────────────────────────────────────
        state = {
            "model":       embedding_model.state_dict(),
            "classifier":  classifier.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "scaler":      scaler.state_dict() if use_amp else {},
            "epoch":       epoch,
            "global_step": global_step,
            "rng_state":   rng_state_dict(),
            "config_hash": _cfg_hash6(yaml_content),
            "manifest_hash": _manifest_hash(train_manifest) if os.path.exists(train_manifest) else "",
            "git_sha":     git_sha,
            "run_id":      run_id,
            "best_eer":    best_eer,
        }
        save_checkpoint(state, ckpt_last)
        logger.info("[ckpt] Saved checkpoint_last.pt (epoch %d)", epoch)

        if val_eer < best_eer:
            best_eer     = val_eer
            patience_ctr = 0
            save_checkpoint(state, ckpt_best)
            logger.info("[ckpt] New best EER=%.4f — saved checkpoint_best.pt", best_eer)
        else:
            patience_ctr += 1
            logger.info(
                "[early-stop] No improvement. Patience %d/%d",
                patience_ctr, hparams["patience"],
            )
            if patience_ctr >= hparams["patience"]:
                logger.info("[early-stop] Patience exhausted — stopping training.")
                break

    logger.info("=" * 65)
    logger.info("  Training complete.  Best val EER: %.4f", best_eer)
    logger.info("  Run ID: %s", run_id)
    logger.info("=" * 65)


# ═══════════════════════════════════════════════════════════════
# 12.  Utilities
# ═══════════════════════════════════════════════════════════════

def _pad_collate(batch):
    """
    Custom collate: pad waveforms in a batch to the same length
    with ZEROS (not repeat-padding).  Returns (waveforms, labels).
    """
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[-1] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[-1]] = w
    return padded, torch.tensor(labels, dtype=torch.long)


class _NoOpScaler:
    """No-op GradScaler shim for CPU training."""
    def scale(self, loss):   return loss
    def unscale_(self, opt): pass
    def step(self, opt):     opt.step()
    def update(self):        pass
    def state_dict(self):    return {}
    def load_state_dict(self, d): pass


class _null_ctx:
    """Context manager that does nothing (CPU fallback for autocast)."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ═══════════════════════════════════════════════════════════════
# 13.  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AAA Engine B1 ECAPA-TDNN fine-tune"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML hparams file (e.g. configs/train_local_smoke.yaml)",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VAL",
        help="Override a hparam: e.g. --override seed=42 --override batch_size=8",
    )
    args = parser.parse_args()
    train(config_path=args.config, overrides=args.override)
