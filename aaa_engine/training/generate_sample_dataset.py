# =============================================================
# training/generate_sample_dataset.py
# AAA Engine - Training Pipeline
# Generate a synthetic sample speaker dataset for training
# =============================================================
# Creates 20 synthetic speakers × 30 utterances each.
# Each speaker has unique audio characteristics:
#   - Unique fundamental frequency (F0: 100–300 Hz)
#   - Unique formant patterns (F1, F2, F3)
#   - Speaker-specific noise profile
#   - Randomized amplitude modulation for natural variation
#
# Output structure:
#   sample_dataset/
#     speaker_0001/
#       utterance_01.wav
#       utterance_02.wav
#       ...
#     speaker_0002/
#       ...
#
# Usage:
#   cd aaa_engine
#   python -m training.generate_sample_dataset
# =============================================================

import os
import numpy as np
import soundfile as sf
import argparse

# ── Configuration ────────────────────────────────────────────
NUM_SPEAKERS      = 20       # Number of synthetic speakers
UTTERANCES_PER_SPK = 30      # Utterances per speaker
SAMPLE_RATE       = 16000    # 16kHz mono (required by ECAPA-TDNN)
DURATION          = 3.0      # Seconds per utterance
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), '..', 'sample_dataset')


def generate_speaker_profile(speaker_id: int, seed: int) -> dict:
    """
    Generate a unique acoustic profile for a synthetic speaker.

    Each speaker gets:
      - A unique fundamental frequency (F0)
      - Three formant frequencies (F1, F2, F3)
      - A unique vibrato rate and depth
      - A speaker-specific noise level

    Args:
        speaker_id: Integer speaker index (0-based)
        seed:       Random seed for reproducibility

    Returns:
        Dictionary containing the speaker's acoustic parameters.
    """
    rng = np.random.RandomState(seed + speaker_id * 137)

    # Fundamental frequency: spread across male (100-180 Hz) and female (180-300 Hz) ranges
    f0 = 100 + (speaker_id / NUM_SPEAKERS) * 200 + rng.uniform(-10, 10)

    # Formant frequencies — unique per speaker
    f1 = rng.uniform(300, 900)     # First formant
    f2 = rng.uniform(900, 2500)    # Second formant
    f3 = rng.uniform(2500, 3500)   # Third formant

    # Formant bandwidths
    bw1 = rng.uniform(40, 120)
    bw2 = rng.uniform(60, 150)
    bw3 = rng.uniform(80, 200)

    # Vibrato characteristics (natural voice variation)
    vibrato_rate = rng.uniform(4, 7)       # Hz
    vibrato_depth = rng.uniform(0.005, 0.02)  # Proportion of F0

    # Speaker-specific noise level
    noise_level = rng.uniform(0.001, 0.008)

    # Amplitude modulation (simulates natural speech rhythm)
    am_rate = rng.uniform(2, 6)   # Hz
    am_depth = rng.uniform(0.2, 0.5)

    return {
        'f0': f0,
        'f1': f1, 'f2': f2, 'f3': f3,
        'bw1': bw1, 'bw2': bw2, 'bw3': bw3,
        'vibrato_rate': vibrato_rate,
        'vibrato_depth': vibrato_depth,
        'noise_level': noise_level,
        'am_rate': am_rate,
        'am_depth': am_depth,
    }


def synthesize_utterance(
    profile: dict,
    utterance_id: int,
    duration: float = DURATION,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Synthesize a single utterance using a speaker's acoustic profile.

    Creates a complex waveform by combining:
      1. Fundamental frequency (F0) with harmonics
      2. Three formant resonances
      3. Natural vibrato modulation
      4. Amplitude modulation (speech-like rhythm)
      5. Speaker-specific noise
      6. Per-utterance random variation (intra-speaker variability)

    Args:
        profile:      Speaker acoustic parameters dictionary
        utterance_id: Integer utterance index (adds natural variation)
        duration:     Audio duration in seconds
        sample_rate:  Output sample rate

    Returns:
        Numpy array of audio samples, normalized to [-1.0, 1.0].
    """
    rng = np.random.RandomState(utterance_id * 31 + int(profile['f0'] * 100))
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float64)

    # -- Per-utterance variation (simulates natural intra-speaker variability) --
    f0_variation = profile['f0'] * (1.0 + rng.uniform(-0.03, 0.03))
    f1_variation = profile['f1'] * (1.0 + rng.uniform(-0.05, 0.05))
    f2_variation = profile['f2'] * (1.0 + rng.uniform(-0.05, 0.05))
    f3_variation = profile['f3'] * (1.0 + rng.uniform(-0.05, 0.05))

    # -- Vibrato modulation on F0 --
    vibrato = 1.0 + profile['vibrato_depth'] * np.sin(
        2 * np.pi * profile['vibrato_rate'] * t
    )
    f0_t = f0_variation * vibrato

    # -- Build waveform: F0 + harmonics --
    # Phase accumulation for time-varying frequency
    phase = 2 * np.pi * np.cumsum(f0_t) / sample_rate
    signal = np.sin(phase)                              # Fundamental
    signal += 0.5 * np.sin(2 * phase)                   # 2nd harmonic
    signal += 0.25 * np.sin(3 * phase)                  # 3rd harmonic
    signal += 0.125 * np.sin(4 * phase)                 # 4th harmonic
    signal += 0.0625 * np.sin(5 * phase)                # 5th harmonic

    # -- Formant resonances (additive sine components at formant frequencies) --
    formant1 = 0.3 * np.sin(2 * np.pi * f1_variation * t) * np.exp(
        -np.pi * profile['bw1'] * (t % (1.0 / max(f0_variation, 1)))
    )
    formant2 = 0.2 * np.sin(2 * np.pi * f2_variation * t) * np.exp(
        -np.pi * profile['bw2'] * (t % (1.0 / max(f0_variation, 1)))
    )
    formant3 = 0.1 * np.sin(2 * np.pi * f3_variation * t) * np.exp(
        -np.pi * profile['bw3'] * (t % (1.0 / max(f0_variation, 1)))
    )
    signal += formant1 + formant2 + formant3

    # -- Amplitude modulation (speech-like envelope) --
    am_variation = profile['am_rate'] * (1.0 + rng.uniform(-0.3, 0.3))
    am = 1.0 - profile['am_depth'] * (
        0.5 * (1.0 + np.sin(2 * np.pi * am_variation * t))
    )
    signal *= am

    # -- Add speech onset/offset (fade in/out) --
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out

    # -- Add speaker-specific noise --
    noise = rng.randn(len(t)) * profile['noise_level']
    signal += noise

    # -- Normalize to [-1, 1] --
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.95  # Leave headroom

    return signal.astype(np.float32)


def generate_dataset(
    output_dir: str = OUTPUT_DIR,
    num_speakers: int = NUM_SPEAKERS,
    utterances_per_speaker: int = UTTERANCES_PER_SPK,
    seed: int = 42
) -> None:
    """
    Generate the complete synthetic speaker dataset.

    Creates the output directory structure and generates all audio files.
    Prints progress as it goes.

    Args:
        output_dir:             Root directory for the dataset
        num_speakers:           Number of speakers to generate
        utterances_per_speaker: Utterances per speaker
        seed:                   Master random seed for reproducibility
    """
    output_dir = os.path.abspath(output_dir)
    print("=" * 65)
    print("  AAA Engine - Synthetic Speaker Dataset Generator")
    print("=" * 65)
    print(f"  Speakers:   {num_speakers}")
    print(f"  Utterances: {utterances_per_speaker} per speaker")
    print(f"  Duration:   {DURATION}s each @ {SAMPLE_RATE} Hz")
    print(f"  Total files: {num_speakers * utterances_per_speaker}")
    print(f"  Output:     {output_dir}")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    for spk_idx in range(num_speakers):
        spk_id = f"speaker_{spk_idx + 1:04d}"
        spk_dir = os.path.join(output_dir, spk_id)
        os.makedirs(spk_dir, exist_ok=True)

        # Generate unique speaker profile
        profile = generate_speaker_profile(spk_idx, seed)

        for utt_idx in range(utterances_per_speaker):
            utt_id = f"utterance_{utt_idx + 1:02d}.wav"
            filepath = os.path.join(spk_dir, utt_id)

            # Synthesize and save
            audio = synthesize_utterance(profile, utt_idx)
            sf.write(filepath, audio, SAMPLE_RATE)
            total_files += 1

        print(f"  [OK] {spk_id}: {utterances_per_speaker} utterances generated "
              f"(F0={profile['f0']:.1f} Hz)")

    print("=" * 65)
    print(f"  [OK] Dataset generation complete: {total_files} files")
    print(f"  [OK] Location: {output_dir}")
    print("=" * 65)


# ── Main Entry Point ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic speaker dataset for ECAPA-TDNN training'
    )
    parser.add_argument(
        '--output-dir', type=str, default=OUTPUT_DIR,
        help='Output directory for the dataset'
    )
    parser.add_argument(
        '--num-speakers', type=int, default=NUM_SPEAKERS,
        help='Number of synthetic speakers'
    )
    parser.add_argument(
        '--utterances', type=int, default=UTTERANCES_PER_SPK,
        help='Utterances per speaker'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        utterances_per_speaker=args.utterances,
        seed=args.seed
    )
