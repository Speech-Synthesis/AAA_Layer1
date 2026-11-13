# =============================================================
# training/train_aaa_sv.py
# AAA Engine – AAA-SV Joint Training (B1 + Stability Head q)
# [CUSTOM]
# =============================================================
#
# This script extends B1 fine-tuning with the auxiliary stability
# head q(x) and joint loss:
#
#   L = L_AAM + lambda_q * L_q
#
# Key design choices (spec §4, §12):
#   • B1 backbone + classifier weights load unchanged via
#     --b1-checkpoint.  The stability head is a SEPARATE module.
#   • Pooled stats vector (B, 6144) is DETACHED before entering
#     the q head — no backbone gradient from L_q.
#   • For L_q we need two views per utterance:
#       x_deg  : augmented short crop (from the normal training batch)
#       x_ref  : clean, longer crop of the same recording (up to
#                q_ref_max_duration seconds, no augmentation)
#   • The reference embedding is stop-grad (computed with
#     torch.no_grad()).
#   • Separate AdamW param group for q head with q_head_lr.
#   • Checkpoint saves/loads the q head + its optimizer state.
#
# Usage:
#   python -m aaa_engine.training.train_aaa_sv \
#       --config configs/train_aaa_sv.yaml \
#       --b1-checkpoint runs/<run_id>/checkpoint_best.pt
#
# Resume (automatic when checkpoint_last.pt exists in the run dir):
#   python -m aaa_engine.training.train_aaa_sv \
#       --config configs/train_aaa_sv.yaml
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ── Project imports ──────────────────────────────────────────
from .evaluate import compute_eer
from .augment import Augmenter, check_ffmpeg_codecs
from .stability_head import (
    StabilityHead,
    StabilityHeadConfig,
    stability_target,
    stability_loss,
    joint_loss,
)
from .train_b1 import (
    # Shared utilities
    _load_audio,
    _git_sha7,
    _cfg_hash6,
    make_run_id,
    _manifest_hash,
    set_seeds,
    worker_init_fn,
    save_checkpoint,
    rng_state_dict,
    restore_rng_state,
    AAMSoftmax,
    WarmupCosineScheduler,
    load_pretrained_ecapa,
    build_feature_pipeline,
    init_class_centres,
    validate,
    setup_run_dir,
    load_hparams,
    _pad_collate,
    _NoOpScaler,
    _null_ctx,
    SpeakerDataset,
)

# SpeechBrain 0.5.x / huggingface_hub >= 0.23 compatibility
try:
    import huggingface_hub
    _orig_hf_download = huggingface_hub.hf_hub_download
    def _compat_hf_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _compat_hf_download
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1.  Dual-view dataset helper
# ═══════════════════════════════════════════════════════════════

class DualViewDataset(SpeakerDataset):
    """
    Extends SpeakerDataset to yield (wav_deg, wav_ref, label) triples.

    wav_deg : augmented short crop  (same as the standard training view)
    wav_ref : clean long crop, up to ref_max_duration seconds, NO augmentation

    Both crops are taken from the SAME recording for the same index.
    The reference crop is taken from a different random offset to
    avoid overlap with the degraded crop whenever possible.
    """

    def __init__(
        self,
        ref_max_duration: float = 6.0,
        **kwargs,
    ) -> None:
        # Strip ref_max_duration before passing to parent
        super().__init__(**kwargs)
        self.ref_max_duration = ref_max_duration

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        wav_path, speaker_id = self.samples[idx]

        if not os.path.isabs(wav_path):
            wav_path = os.path.join(self.data_folder, wav_path)

        waveform, sr = _load_audio(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        total = waveform.shape[-1]

        # ── Degraded view: short crop + augmentation ──────────
        crop_samples_deg = int(self.crop_len_seconds * self.sample_rate)
        if total > crop_samples_deg:
            start_deg = random.randint(0, total - crop_samples_deg)
            wav_deg = waveform[:, start_deg : start_deg + crop_samples_deg]
        else:
            wav_deg = waveform.clone()
        if self.augmenter is not None:
            wav_deg = self.augmenter.augment(wav_deg, self.sample_rate)

        # ── Reference view: long clean crop, NO augmentation ──
        crop_samples_ref = int(self.ref_max_duration * self.sample_rate)
        if total > crop_samples_ref:
            start_ref = random.randint(0, total - crop_samples_ref)
            wav_ref = waveform[:, start_ref : start_ref + crop_samples_ref]
        else:
            wav_ref = waveform.clone()
        # No augmentation on reference view

        label = self.speaker_to_idx[speaker_id]
        return wav_deg.squeeze(0), wav_ref.squeeze(0), label


def _dual_pad_collate(batch):
    """
    Collate (wav_deg, wav_ref, label) triples.
    Returns (wavs_deg, wavs_ref, labels) each padded independently.
    """
    wavs_deg, wavs_ref, labels = zip(*batch)

    max_deg = max(w.shape[-1] for w in wavs_deg)
    max_ref = max(w.shape[-1] for w in wavs_ref)

    pad_deg = torch.zeros(len(wavs_deg), max_deg)
    pad_ref = torch.zeros(len(wavs_ref), max_ref)

    for i, (wd, wr) in enumerate(zip(wavs_deg, wavs_ref)):
        pad_deg[i, : wd.shape[-1]] = wd
        pad_ref[i, : wr.shape[-1]] = wr

    return pad_deg, pad_ref, torch.tensor(labels, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# 2.  Pooled-stats extractor
# ═══════════════════════════════════════════════════════════════

def _extract_pooled_stats(
    embedding_model: nn.Module,
    compute_features,
    mean_var_norm,
    waveforms: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the ECAPA-TDNN forward pass and return BOTH:
      - embeddings    : (B, 192)  — L2-normalised speaker embeddings
      - pooled_stats  : (B, 6144) — raw ASP output BEFORE the
                        fully-connected projection layer

    For ECAPA-TDNN the ASP output is (B, 6144) = 3072 mean + 3072 std
    concatenated (spec §6).  The final Linear(6144→192) projector maps
    this to the 192-dim embedding.

    We hook into the penultimate layer to capture pooled_stats.
    """
    pooled_buf: List[torch.Tensor] = []

    # Register a forward hook on the last asp/pool module.
    # In SpeechBrain's ECAPA_TDNN the attentive stats pooling output
    # feeds into self.fc (a Conv1d(6144, 192, 1)).
    # We hook self.asp_bn (the BN right after the concat) or self.fc.
    target_layer = None
    for name, module in embedding_model.named_modules():
        # Look for the final batch-norm before projection, or the
        # projection conv itself.
        if name in ("asp_bn", "fc"):
            target_layer = module
            # Prefer asp_bn (outputs 6144-dim); fall back to fc.
            if name == "asp_bn":
                break

    def _hook(module, inp, out):
        # out shape may be (B, 6144, 1) for Conv1d or (B, 6144) for BN
        t = out.detach()
        if t.dim() == 3:
            t = t.squeeze(-1)   # (B, 6144)
        pooled_buf.append(t)

    handle = None
    if target_layer is not None:
        handle = target_layer.register_forward_hook(_hook)

    feats = compute_features(waveforms)
    lens  = torch.ones(waveforms.shape[0], device=device)
    feats = mean_var_norm(feats, lens)
    emb   = embedding_model(feats)
    if emb.dim() == 3:
        emb = emb.squeeze(1)   # (B, 192)

    if handle is not None:
        handle.remove()

    if pooled_buf:
        pooled_stats = pooled_buf[0]
        # If asp_bn hooked, pooled_stats is already (B, 6144).
        # If fc hooked, it is the fc output (B, 192) — fall back to
        # repeating emb to match expected pool_dim (should not happen
        # on a standard ECAPA build, but keep code safe).
        if pooled_stats.shape[-1] != 6144:
            # Fallback: tile to 6144 — log a warning once
            if not hasattr(_extract_pooled_stats, "_pool_dim_warned"):
                logger.warning(
                    "[StabilityHead] Could not hook asp_bn (pool_dim=%d != 6144). "
                    "Tiling embedding. Set pretrained_pool_dim in config to "
                    "match actual hook output.",
                    pooled_stats.shape[-1],
                )
                _extract_pooled_stats._pool_dim_warned = True
            repeat = 6144 // pooled_stats.shape[-1]
            remainder = 6144 % pooled_stats.shape[-1]
            pooled_stats = torch.cat(
                [pooled_stats] * repeat
                + ([pooled_stats[:, :remainder]] if remainder else []),
                dim=1,
            )
    else:
        # No hook fired — use embedding tiled to pool_dim
        logger.warning(
            "[StabilityHead] Forward hook did not fire. "
            "Using tiled 192-dim embedding as pooled_stats fallback."
        )
        repeat = 6144 // emb.shape[-1]
        remainder = 6144 % emb.shape[-1]
        pooled_stats = torch.cat(
            [emb.detach()] * repeat
            + ([emb.detach()[:, :remainder]] if remainder else []),
            dim=1,
        )

    return emb, pooled_stats


# ═══════════════════════════════════════════════════════════════
# 3.  Config loading (extends train_b1 defaults)
# ═══════════════════════════════════════════════════════════════

def load_aaa_hparams(config_path: str, overrides: List[str]) -> Tuple[Dict[str, Any], str]:
    """
    Load hparams and apply AAA-SV specific defaults on top of B1 defaults.
    """
    hparams, yaml_content = load_hparams(config_path, overrides)

    # AAA-SV additional defaults
    aaa_defaults = {
        "lambda_q":             0.1,
        "q_head_lr":            1e-3,
        "q_ref_max_duration":   6.0,
        "q_pool_dim":           6144,
        "q_hidden_dim":         256,
        "enable_stability_head": True,
        "b1_checkpoint":        None,   # path to warm-start B1 ckpt
    }
    for k, v in aaa_defaults.items():
        hparams.setdefault(k, v)

    return hparams, yaml_content


# ═══════════════════════════════════════════════════════════════
# 4.  Main training loop
# ═══════════════════════════════════════════════════════════════

def train_aaa_sv(config_path: str, overrides: List[str], b1_checkpoint: Optional[str] = None) -> None:
    # ── Load config ──────────────────────────────────────────
    hparams, yaml_content = load_aaa_hparams(config_path, overrides)

    # CLI --b1-checkpoint takes priority over config
    if b1_checkpoint:
        hparams["b1_checkpoint"] = b1_checkpoint

    run_id  = make_run_id(yaml_content)
    git_sha = _git_sha7()

    logger.info("=" * 65)
    logger.info("  AAA Engine AAA-SV Training — Run ID: %s", run_id)
    logger.info("=" * 65)

    # ── Seed ────────────────────────────────────────────────
    set_seeds(hparams["seed"])

    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[device] Using: %s", device)

    # ── Paths ───────────────────────────────────────────────
    runs_dir       = hparams["runs_dir"]
    data_folder    = hparams["data_folder"]
    train_manifest = hparams.get("train_manifest",
                        os.path.join(data_folder, "train.csv"))
    val_manifest   = hparams.get("val_manifest",
                        os.path.join(data_folder, "val.csv"))

    # ── Run directory ───────────────────────────────────────
    run_dir = setup_run_dir(
        run_id, hparams, yaml_content, runs_dir, git_sha, train_manifest
    )
    ckpt_last = os.path.join(run_dir, "checkpoint_last.pt")
    ckpt_best = os.path.join(run_dir, "checkpoint_best.pt")

    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    metrics_csv   = os.path.join(run_dir, "metrics.csv")

    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "epoch", "train_loss", "train_loss_aam",
                "train_loss_q", "val_eer", "val_frr_at_far1",
                "lr_backbone", "lr_head", "lr_q_head",
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

    # ── Dual-view dataset & DataLoader ───────────────────────
    train_dataset = DualViewDataset(
        manifest_path=train_manifest,
        data_folder=data_folder,
        augmenter=augmenter,
        speaker_to_idx=speaker_to_idx,
        crop_len_seconds=float(random.choice(hparams["crop_lengths"])),
        sample_rate=16000,
        shuffle_labels=hparams["debug_shuffle_labels"],
        seed=hparams["seed"],
        ref_max_duration=float(hparams["q_ref_max_duration"]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=_dual_pad_collate,
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

    # ── Stability Head (q) ───────────────────────────────────
    q_config = StabilityHeadConfig(
        pool_dim=int(hparams["q_pool_dim"]),
        hidden_dim=int(hparams["q_hidden_dim"]),
        lambda_q=float(hparams["lambda_q"]),
        q_ref_max_duration=float(hparams["q_ref_max_duration"]),
    )
    q_head = StabilityHead(config=q_config).to(device)
    logger.info("[StabilityHead] pool_dim=%d, hidden_dim=%d, lambda_q=%s",
                q_config.pool_dim, q_config.hidden_dim, q_config.lambda_q)

    # ── Load B1 checkpoint (warm-start) ──────────────────────
    b1_ckpt_path = hparams.get("b1_checkpoint")
    if b1_ckpt_path and os.path.isfile(b1_ckpt_path):
        logger.info("[B1] Loading backbone + classifier from: %s", b1_ckpt_path)
        b1_state = torch.load(b1_ckpt_path, map_location=device)
        embedding_model.load_state_dict(b1_state["model"])
        try:
            classifier.load_state_dict(b1_state["classifier"])
            logger.info("[B1] Classifier loaded from B1 checkpoint.")
        except Exception as exc:
            logger.warning("[B1] Classifier load failed (%s) — using fresh head.", exc)
        logger.info("[B1] B1 backbone + classifier loaded successfully.")
    else:
        # No B1 checkpoint: initialise class centres from pretrained model
        logger.info("[B1] No B1 checkpoint — initialising class centres from pretrained.")
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
            {"params": embedding_model.parameters(),
             "lr": hparams["backbone_lr"],  "weight_decay": 1e-2},
            {"params": classifier.parameters(),
             "lr": hparams["head_lr"],      "weight_decay": 1e-2},
            {"params": q_head.parameters(),
             "lr": hparams["q_head_lr"],    "weight_decay": 1e-2},
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
    scaler  = torch.cuda.amp.GradScaler() if use_amp else _NoOpScaler()

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
            if "q_head" in ckpt:
                q_head.load_state_dict(ckpt["q_head"])
                logger.info("[RESUME] q_head state loaded.")
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
        except Exception as exc:
            logger.warning("[RESUME] Checkpoint load failed (%s) — starting fresh.", exc)

    # ── Training loop ────────────────────────────────────────
    log_every     = hparams["log_every_n_steps"]
    lambda_q      = float(hparams["lambda_q"])
    enable_q_head = bool(hparams.get("enable_stability_head", True))

    for epoch in range(start_epoch, hparams["number_of_epochs"] + 1):

        # Unfreeze backbone from epoch 2
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
        q_head.train()

        epoch_loss     = 0.0
        epoch_loss_aam = 0.0
        epoch_loss_q   = 0.0
        micro_step     = 0
        optimizer.zero_grad()

        for batch_idx, (wavs_deg, wavs_ref, labels) in enumerate(train_loader):
            wavs_deg = wavs_deg.to(device)
            wavs_ref = wavs_ref.to(device)
            labels   = labels.to(device)

            # Margin ramp
            current_margin = min(
                hparams["aam_margin"],
                hparams["aam_margin"] * global_step / max(1, 0.2 * total_steps),
            )
            classifier.set_margin(current_margin)

            with (torch.cuda.amp.autocast() if use_amp else _null_ctx()):
                # ── Degraded view forward (with grad) ──────────
                feats_deg = compute_features(wavs_deg)
                lens_deg  = torch.ones(wavs_deg.shape[0], device=device)
                feats_deg = mean_var_norm(feats_deg, lens_deg)

                emb_deg, pooled_deg = _extract_pooled_stats(
                    embedding_model, compute_features, mean_var_norm,
                    wavs_deg, device,
                )
                # Squeeze if 3-dim
                if emb_deg.dim() == 3:
                    emb_deg = emb_deg.squeeze(1)

                # AAM-Softmax loss
                l_aam = classifier(emb_deg, labels)

                # ── Stability Head loss ─────────────────────────
                if enable_q_head:
                    # Reference view: stop-grad forward pass
                    with torch.no_grad():
                        emb_ref, _ = _extract_pooled_stats(
                            embedding_model, compute_features, mean_var_norm,
                            wavs_ref, device,
                        )
                        if emb_ref.dim() == 3:
                            emb_ref = emb_ref.squeeze(1)

                    # Target y_q: cosine sim between degraded and clean embs,
                    # clamped to [0,1] (spec §7)
                    y_q = stability_target(emb_deg, emb_ref)   # (B,)

                    # q head input: pooled stats DETACHED (spec §4, §12)
                    q_scores = q_head(pooled_deg.detach())      # (B,)
                    l_q      = stability_loss(q_scores, y_q)
                    l_total  = joint_loss(l_aam, l_q, lambda_q=lambda_q)
                else:
                    l_q     = torch.zeros(1, device=device)
                    l_total = l_aam

                l_total = l_total / hparams["grad_accum_steps"]

            scaler.scale(l_total).backward()
            epoch_loss     += l_total.item() * hparams["grad_accum_steps"]
            epoch_loss_aam += l_aam.item()
            epoch_loss_q   += l_q.item()
            micro_step     += 1

            if micro_step % hparams["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (list(embedding_model.parameters())
                     + list(classifier.parameters())
                     + list(q_head.parameters())),
                    hparams["grad_clip"],
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    lr_bb = optimizer.param_groups[0]["lr"]
                    lr_hd = optimizer.param_groups[1]["lr"]
                    lr_qh = optimizer.param_groups[2]["lr"]
                    record = {
                        "run_id":      run_id,
                        "epoch":       epoch,
                        "global_step": global_step,
                        "loss":        round(l_total.item() * hparams["grad_accum_steps"], 6),
                        "loss_aam":    round(l_aam.item(), 6),
                        "loss_q":      round(l_q.item(), 6),
                        "margin":      round(current_margin, 6),
                        "lr_backbone": lr_bb,
                        "lr_head":     lr_hd,
                        "lr_q_head":   lr_qh,
                        "grad_norm":   round(float(grad_norm), 6),
                    }
                    with open(metrics_jsonl, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(record) + "\n")

                    logger.info(
                        "[%d/%d] step=%d loss=%.4f (aam=%.4f q=%.4f) "
                        "margin=%.3f lr_bb=%.2e gn=%.2f",
                        epoch, hparams["number_of_epochs"],
                        global_step, record["loss"],
                        record["loss_aam"], record["loss_q"],
                        current_margin, lr_bb, float(grad_norm),
                    )

        avg_loss     = epoch_loss     / max(1, len(train_loader))
        avg_loss_aam = epoch_loss_aam / max(1, len(train_loader))
        avg_loss_q   = epoch_loss_q   / max(1, len(train_loader))

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
                run_id, epoch,
                round(avg_loss, 6), round(avg_loss_aam, 6), round(avg_loss_q, 6),
                round(val_eer, 6), round(val_frr1, 6),
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
                optimizer.param_groups[2]["lr"],
            ])

        # ── Checkpoint ──────────────────────────────────────
        state = {
            "model":       embedding_model.state_dict(),
            "classifier":  classifier.state_dict(),
            "q_head":      q_head.state_dict(),           # [CUSTOM] q head
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
            "q_config":    {
                "pool_dim":           q_config.pool_dim,
                "hidden_dim":         q_config.hidden_dim,
                "lambda_q":           q_config.lambda_q,
                "q_ref_max_duration": q_config.q_ref_max_duration,
            },
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
    logger.info("  AAA-SV Training complete.  Best val EER: %.4f", best_eer)
    logger.info("  Run ID: %s", run_id)
    logger.info("=" * 65)


# ═══════════════════════════════════════════════════════════════
# 5.  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AAA Engine AAA-SV: B1 backbone + stability head q(x)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML hparams file (e.g. configs/train_aaa_sv.yaml)",
    )
    parser.add_argument(
        "--b1-checkpoint", default=None,
        help="Path to a B1 checkpoint_best.pt to warm-start the backbone.",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VAL",
        help="Override a hparam: e.g. --override seed=42",
    )
    args = parser.parse_args()
    train_aaa_sv(
        config_path=args.config,
        overrides=args.override,
        b1_checkpoint=args.b1_checkpoint,
    )
