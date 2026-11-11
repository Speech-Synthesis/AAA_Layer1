# =============================================================
# training/augment.py
# AAA Engine – B1 Training Pipeline
# [CUSTOM] Seeded, reproducible audio augmentation pipeline
# =============================================================
# Applies a stochastic chain of augmentations for data-
# augmentation during speaker-verification fine-tuning.
#
# Augmentation chain (applied independently per sample):
#   1. MUSAN noise injection  (prob=0.6) — additive, random SNR
#   2. RIRS_NOISES reverb     (prob=0.5) — full convolution
#   3. Band-limit low-pass    (prob=0.3) — torchaudio resample
#   4. Resample 16→8→16 kHz   (prob=0.2) — codec-like degradation
#   5. Codec encode/decode     (prob=0.3) — ffmpeg subprocess
#
# Design principles
#   - All randomness via a seeded torch.Generator (reproducible)
#   - Missing MUSAN/RIRS dirs: skip silently, warn once
#   - Never pads short clips; never repeats to fill duration
#   - Codec support probed at startup via check_ffmpeg_codecs()
# =============================================================

from __future__ import annotations

import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)

# ── Supported codec list ──────────────────────────────────────
_CODEC_CANDIDATES: List[str] = ["opus", "aac", "amr_nb", "g711_ulaw"]

# Map codec name → (encoder, decoder, container extension)
_CODEC_MAP = {
    "opus":     ("libopus",    "libopus",    "ogg"),
    "aac":      ("aac",        "aac",        "m4a"),
    "amr_nb":   ("libopencore_amrnb", "libopencore_amrnb", "amr"),
    "g711_ulaw":("pcm_mulaw",  "pcm_mulaw",  "wav"),
}

# ── Codec probing ─────────────────────────────────────────────

def check_ffmpeg_codecs() -> List[str]:
    """
    [CUSTOM] Probe ffmpeg for each candidate codec.

    For each codec in _CODEC_CANDIDATES, encodes a tiny silent
    clip and checks the return code.  Returns only codecs whose
    round-trip succeeds.  Returns [] if ffmpeg is not on PATH.
    """
    supported: List[str] = []

    # Quick PATH check
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        logger.warning("[augment] ffmpeg not found on PATH — codec augmentation disabled.")
        return []

    with tempfile.TemporaryDirectory() as tmp:
        # 0.5-s silent WAV at 16 kHz, mono (raw bytes)
        silent_wav = os.path.join(tmp, "probe.wav")
        # Write a minimal valid WAV via torchaudio
        torchaudio.save(
            silent_wav,
            torch.zeros(1, 8000),
            16000,
            format="wav",
        )

        for codec in _CODEC_CANDIDATES:
            if codec not in _CODEC_MAP:
                continue
            enc, _dec, ext = _CODEC_MAP[codec]
            out_file = os.path.join(tmp, f"probe.{ext}")
            try:
                r = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", silent_wav,
                        "-acodec", enc,
                        "-ar", "8000" if codec == "amr_nb" else "16000",
                        out_file,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                if r.returncode == 0 and os.path.exists(out_file):
                    supported.append(codec)
            except Exception:  # noqa: BLE001
                pass

    logger.info("[augment] Supported codecs: %s", supported or "(none)")
    return supported


# ── Augmenter ─────────────────────────────────────────────────

class Augmenter:
    """
    [CUSTOM] Seeded, reproducible audio augmentation pipeline.

    Parameters
    ----------
    seed : int
        Master seed for the internal torch.Generator.
    musan_dir : str | None
        Root directory of the MUSAN corpus.  Pass None to skip
        noise injection silently.
    rirs_dir : str | None
        Root directory of RIRS_NOISES corpus.  Pass None to skip
        reverb silently.
    ffmpeg_available : bool
        Whether to enable codec augmentation.  The caller should
        set this to ``bool(check_ffmpeg_codecs())``.
    """

    # Augmentation probabilities
    P_NOISE   = 0.6
    P_REVERB  = 0.5
    P_BANDLIM = 0.3
    P_RESAMP  = 0.2
    P_CODEC   = 0.3

    # SNR candidates for noise injection (dB)
    SNR_CHOICES = [0, 5, 10, 20]

    def __init__(
        self,
        seed: int,
        musan_dir: Optional[str] = None,
        rirs_dir: Optional[str] = None,
        ffmpeg_available: bool = False,
    ) -> None:
        self.seed = seed
        self.ffmpeg_available = ffmpeg_available

        # Seeded generator — do NOT share with training dataloader
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

        # ── MUSAN ──
        self._musan_files: List[str] = []
        self._warn_musan = False
        if musan_dir and Path(musan_dir).is_dir():
            self._musan_files = _collect_wavs(musan_dir)
            if not self._musan_files:
                logger.warning(
                    "[augment] MUSAN dir found but contains no .wav files: %s", musan_dir
                )
            else:
                logger.info(
                    "[augment] Loaded %d MUSAN noise files from %s",
                    len(self._musan_files), musan_dir,
                )
        elif musan_dir:
            logger.warning(
                "[augment] MUSAN dir not found: %s — noise injection disabled.", musan_dir
            )

        # ── RIRS_NOISES ──
        self._rir_files: List[str] = []
        self._warn_rirs = False
        if rirs_dir and Path(rirs_dir).is_dir():
            self._rir_files = _collect_wavs(rirs_dir)
            if not self._rir_files:
                logger.warning(
                    "[augment] RIRS dir found but contains no .wav files: %s", rirs_dir
                )
            else:
                logger.info(
                    "[augment] Loaded %d RIR files from %s",
                    len(self._rir_files), rirs_dir,
                )
        elif rirs_dir:
            logger.warning(
                "[augment] RIRS dir not found: %s — reverb disabled.", rirs_dir
            )

        # Supported codecs (probe once)
        self._codecs: List[str] = check_ffmpeg_codecs() if ffmpeg_available else []

    # ── Public API ───────────────────────────────────────────

    def augment(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Apply the stochastic augmentation chain.

        Parameters
        ----------
        waveform : torch.Tensor  shape (1, T) or (T,)
        sr : int                 original sample rate (expected 16000)

        Returns
        -------
        torch.Tensor  same shape/dtype as input.
        No padding is ever applied — if an operation shortens the clip,
        the shorter tensor is returned as-is.
        """
        # Ensure (1, T) layout
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        orig_len = waveform.shape[-1]

        # 1. Noise injection
        if self._musan_files and self._rand() < self.P_NOISE:
            waveform = self._add_noise(waveform, sr)

        # 2. Reverb
        if self._rir_files and self._rand() < self.P_REVERB:
            waveform = self._apply_reverb(waveform, sr)

        # 3. Band-limit (low-pass via downsample)
        if self._rand() < self.P_BANDLIM:
            waveform = self._band_limit(waveform, sr)

        # 4. Resample 16 → 8 → 16 kHz
        if self._rand() < self.P_RESAMP:
            waveform = self._resample_chain(waveform, sr)

        # 5. Codec round-trip
        if self._codecs and self._rand() < self.P_CODEC:
            waveform = self._codec_roundtrip(waveform, sr)

        # Safety: never return a longer tensor than input
        if waveform.shape[-1] > orig_len:
            waveform = waveform[..., :orig_len]

        return waveform

    # ── Private helpers ──────────────────────────────────────

    def _rand(self) -> float:
        """Draw a uniform float in [0, 1) from the seeded generator."""
        return torch.rand(1, generator=self._gen).item()

    def _rand_int(self, low: int, high: int) -> int:
        """Draw a uniform int in [low, high) from the seeded generator."""
        return int(torch.randint(low, high, (1,), generator=self._gen).item())

    def _rand_choice(self, lst: list):
        """Pick a random element from a list using the seeded generator."""
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    # ── Augmentation steps ───────────────────────────────────

    def _add_noise(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        [CUSTOM] Additive MUSAN noise at a random SNR from SNR_CHOICES.
        Loads a random noise file; if the noise is shorter than the
        signal, the noise is tiled (this is noise, not speech, so
        tiling is acoustically reasonable).  If longer, it is cropped.
        """
        noise_path = self._rand_choice(self._musan_files)
        try:
            noise, noise_sr = torchaudio.load(noise_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[augment] Could not load noise file %s: %s", noise_path, exc)
            return waveform

        # Resample noise if needed
        if noise_sr != sr:
            noise = torchaudio.functional.resample(noise, noise_sr, sr)

        # Mix to mono
        if noise.shape[0] > 1:
            noise = noise.mean(0, keepdim=True)

        sig_len = waveform.shape[-1]
        noise_len = noise.shape[-1]

        if noise_len < sig_len:
            # Tile noise to cover the signal (OK for noise, not speech)
            repeats = (sig_len // noise_len) + 1
            noise = noise.repeat(1, repeats)

        # Crop to signal length
        noise = noise[..., :sig_len]

        # Random SNR
        snr_db = float(self._rand_choice(self.SNR_CHOICES))

        # Compute RMS-based scaling
        sig_rms  = waveform.pow(2).mean().sqrt().clamp_min(1e-9)
        noi_rms  = noise.pow(2).mean().sqrt().clamp_min(1e-9)
        snr_lin  = 10 ** (snr_db / 20.0)
        scale    = sig_rms / (snr_lin * noi_rms)

        return (waveform + scale * noise).clamp(-1.0, 1.0)

    def _apply_reverb(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        [CUSTOM] Convolve signal with a random RIR (Room Impulse Response).
        Output is trimmed to the original length — no padding.
        """
        rir_path = self._rand_choice(self._rir_files)
        try:
            rir, rir_sr = torchaudio.load(rir_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[augment] Could not load RIR file %s: %s", rir_path, exc)
            return waveform

        if rir_sr != sr:
            rir = torchaudio.functional.resample(rir, rir_sr, sr)

        # Use only the first channel of RIR
        if rir.shape[0] > 1:
            rir = rir[0:1]

        orig_len = waveform.shape[-1]

        # FFT-based convolution
        try:
            # torchaudio >= 0.9 ships fftconvolve
            reverbed = torchaudio.functional.fftconvolve(waveform, rir)
        except AttributeError:
            # Fallback: direct torch conv1d
            reverbed = torch.nn.functional.conv1d(
                waveform.unsqueeze(0),
                rir.flip(-1).unsqueeze(0),
                padding=rir.shape[-1] - 1,
            ).squeeze(0)

        # Trim to original length (reverb tail is discarded)
        if reverbed.shape[-1] > orig_len:
            reverbed = reverbed[..., :orig_len]

        # Normalise to prevent clipping
        peak = reverbed.abs().max().clamp_min(1e-9)
        if peak > 1.0:
            reverbed = reverbed / peak

        return reverbed

    def _band_limit(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        [CUSTOM] Simulate band-limited channel via torchaudio resample.
        Picks a random low-pass cutoff in [2000, 7000] Hz by downsampling
        to 2*cutoff then back to sr.  Returns original tensor on failure.
        """
        # Random cutoff in [2000, 7000] Hz
        cutoff = self._rand_int(2000, 7001)
        target_sr = cutoff * 2  # Nyquist

        if target_sr >= sr:
            return waveform  # no degradation needed

        try:
            down = torchaudio.functional.resample(waveform, sr, target_sr)
            up   = torchaudio.functional.resample(down, target_sr, sr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[augment] band_limit failed (cutoff=%d): %s", cutoff, exc)
            return waveform

        orig_len = waveform.shape[-1]
        if up.shape[-1] > orig_len:
            up = up[..., :orig_len]

        return up

    def _resample_chain(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        [CUSTOM] Downsample to 8 kHz then back to sr (codec-like degradation).
        """
        if sr <= 8000:
            return waveform
        try:
            down = torchaudio.functional.resample(waveform, sr, 8000)
            up   = torchaudio.functional.resample(down, 8000, sr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[augment] resample_chain failed: %s", exc)
            return waveform

        orig_len = waveform.shape[-1]
        if up.shape[-1] > orig_len:
            up = up[..., :orig_len]

        return up

    def _codec_roundtrip(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        [CUSTOM] Encode/decode via ffmpeg subprocess with a random codec.
        On any failure, returns the original waveform unchanged.
        """
        if not self._codecs:
            return waveform

        codec = self._rand_choice(self._codecs)
        enc, _dec, ext = _CODEC_MAP[codec]

        # AMR-NB requires 8 kHz — downsample first
        codec_sr = 8000 if codec == "amr_nb" else sr

        with tempfile.TemporaryDirectory() as tmp:
            in_wav   = os.path.join(tmp, "in.wav")
            out_enc  = os.path.join(tmp, f"out.{ext}")
            out_wav  = os.path.join(tmp, "out.wav")

            try:
                # Prepare source (may need resample for codec constraints)
                src = waveform
                if codec_sr != sr:
                    src = torchaudio.functional.resample(waveform, sr, codec_sr)

                torchaudio.save(in_wav, src, codec_sr, format="wav")

                # Encode
                r1 = subprocess.run(
                    ["ffmpeg", "-y", "-i", in_wav, "-acodec", enc, out_enc],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                if r1.returncode != 0:
                    return waveform

                # Decode back to WAV
                r2 = subprocess.run(
                    ["ffmpeg", "-y", "-i", out_enc, "-acodec", "pcm_s16le", out_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                if r2.returncode != 0:
                    return waveform

                decoded, dec_sr = torchaudio.load(out_wav)

                # Resample back to original sr if needed
                if dec_sr != sr:
                    decoded = torchaudio.functional.resample(decoded, dec_sr, sr)

                # Trim to original length — never pad
                orig_len = waveform.shape[-1]
                if decoded.shape[-1] > orig_len:
                    decoded = decoded[..., :orig_len]

                # Mono
                if decoded.shape[0] > 1:
                    decoded = decoded.mean(0, keepdim=True)

                return decoded

            except Exception as exc:  # noqa: BLE001
                logger.debug("[augment] codec_roundtrip failed (%s): %s", codec, exc)
                return waveform


# ── Utilities ─────────────────────────────────────────────────

def _collect_wavs(root: str) -> List[str]:
    """Recursively collect all .wav files under root."""
    wav_files: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(dirpath, fname))
    return wav_files
