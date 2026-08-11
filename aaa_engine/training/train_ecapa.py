# =============================================================
# training/train_ecapa.py
# AAA Engine - Training Pipeline
# Fine-tune ECAPA-TDNN for Speaker Verification
# =============================================================
# Adapted from SpeechBrain's official VoxCeleb recipe:
#   recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py
#
# This script fine-tunes the pre-trained ECAPA-TDNN model on a
# custom speaker dataset using AAM-Softmax loss.
#
# Usage:
#   cd aaa_engine
#   python -m training.train_ecapa training/train_hparams.yaml \
#       --data_folder=sample_dataset
#
# The script will:
#   1. Load pre-trained VoxCeleb weights (embedding model only)
#   2. Replace the classifier head for our speaker count
#   3. Train for the configured number of epochs
#   4. Save checkpoints to results/ecapa_sample/<seed>/save/
#
# Authors (original SpeechBrain recipe):
#   * Mirco Ravanelli 2020
#   * Hwidong Na 2020
#   * Nauman Dawalatabad 2020
#
# Adapted for AAA Engine by the AAA team.
# =============================================================

import os
import sys
import csv
import random
import glob
import signal

import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed=1986):
    """
    Set all random seeds for reproducible training.

    Ensures deterministic behavior across:
        - Python random
        - NumPy
        - PyTorch CPU & CUDA
        - cuDNN
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)


class SpeakerBrain(sb.core.Brain):
    """
    SpeechBrain Brain class for speaker embedding training.

    Implements the forward pass and loss computation for
    ECAPA-TDNN speaker verification training using
    AAM-Softmax (Additive Angular Margin Softmax) loss.
    """

    def compute_forward(self, batch, stage):
        """
        Forward pass: audio → features → embeddings → classifier.

        Pipeline:
            1. Extract Mel filterbank features (80 bins)
            2. Apply mean-variance normalization
            3. Pass through ECAPA-TDNN encoder → 192-dim embedding
            4. Pass through classifier → speaker logits

        Args:
            batch: SpeechBrain batch object with 'sig' and 'spk_id_encoded'
            stage: sb.Stage.TRAIN / sb.Stage.VALID / sb.Stage.TEST
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Feature extraction: waveform → 80-dim Mel filterbanks
        feats = self.modules.compute_features(wavs)

        # Mean-variance normalization
        feats = self.modules.mean_var_norm(feats, lens)

        # ECAPA-TDNN encoder → 192-dim embedding
        embeddings = self.modules.embedding_model(feats)

        # Classifier head → speaker logits
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """
        Compute AAM-Softmax loss using speaker ID as the label.

        Args:
            predictions: (logits, lens) from compute_forward
            batch:       SpeechBrain batch with 'spk_id_encoded'
            stage:       Training stage
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Compute AAM-Softmax loss
        loss = self.hparams.compute_cost(predictions, spkid, lens)

        # Update LR on each batch if scheduler supports it
        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, 'on_batch_end'
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # Track error metrics during validation
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Initialize error metrics at the start of validation."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        End-of-epoch: log stats, anneal LR, save checkpoints.

        For validation stage:
            - Compute classification error rate
            - Adjust learning rate
            - Save checkpoint (keeping only the best model)
        """
        stage_stats = {'loss': stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats['ErrorRate'] = self.error_metrics.summarize('average')

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={'epoch': epoch, 'lr': old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={'ErrorRate': stage_stats['ErrorRate']},
                min_keys=['ErrorRate'],
            )

            print(f"  Epoch {epoch}: "
                  f"train_loss={self.train_stats['loss']:.4f}, "
                  f"valid_loss={stage_loss:.4f}, "
                  f"valid_err={stage_stats['ErrorRate']:.4f}, "
                  f"lr={old_lr:.6f}")


def prepare_csv_from_folder(data_folder, save_folder, split_ratio, sentence_len):
    """
    Create SpeechBrain CSV manifests from a folder-structured dataset.

    This replaces the VoxCeleb-specific preparation script.
    Scans speaker subdirectories and creates train/dev CSV splits.

    Args:
        data_folder:  Root directory with speaker subdirectories
        save_folder:  Directory to write train.csv and dev.csv
        split_ratio:  [train%, dev%, test%] split percentages
        sentence_len: Expected utterance duration in seconds
    """
    data_folder = os.path.abspath(data_folder)
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n[Data Prep] Scanning {data_folder}...")

    # Collect all audio files grouped by speaker
    all_entries = []
    speakers = set()

    for spk_folder in sorted(os.listdir(data_folder)):
        spk_path = os.path.join(data_folder, spk_folder)

        if not os.path.isdir(spk_path):
            continue

        # VoxCeleb1 structure:
        # speaker_id/video_id/*.wav
        wav_files = sorted(
            glob.glob(
                os.path.join(spk_path, '**', '*.wav'),
                recursive=True
            )
        )

        if not wav_files:
            continue

        speakers.add(spk_folder)

        for wav_path in wav_files:
            # Avoid expensive soundfile.info() calls on the entire dataset.
            # VoxCeleb1 audio is 16 kHz.
            duration = sentence_len

            basename = os.path.splitext(os.path.basename(wav_path))[0]
            entry_id = f"{spk_folder}_{basename}"

            num_samples = int(duration * 16000)

            all_entries.append({
                'ID': entry_id,
                'duration': f"{duration:.4f}",
                'wav': wav_path,
                'start': 0,
                'stop': num_samples,
                'spk_id': spk_folder,
            })

    print(f"[Data Prep] Found {len(speakers)} speakers, {len(all_entries)} utterances")

    # Shuffle and split per speaker
    rng = random.Random(1986)
    speaker_entries = {}
    for entry in all_entries:
        spk = entry['spk_id']
        if spk not in speaker_entries:
            speaker_entries[spk] = []
        speaker_entries[spk].append(entry)

    train_entries = []
    dev_entries = []

    train_pct = split_ratio[0] / sum(split_ratio)

    for spk, entries in sorted(speaker_entries.items()):
        rng.shuffle(entries)
        n_train = max(1, int(len(entries) * train_pct))
        train_entries.extend(entries[:n_train])
        dev_entries.extend(entries[n_train:])

    # Write CSVs
    for split_name, entries in [('train', train_entries), ('dev', dev_entries)]:
        csv_path = os.path.join(save_folder, f'{split_name}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['ID', 'duration', 'wav', 'start', 'stop', 'spk_id']
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)
        print(f"[Data Prep] [OK] {csv_path} ({len(entries)} utterances)")


def dataio_prep(hparams):
    """
    Create SpeechBrain datasets and data processing pipelines.

    Sets up:
        1. Train and validation DynamicItemDatasets from CSV
        2. Audio pipeline: load WAV → extract chunk → tensor
        3. Label pipeline: speaker_id → encoded integer
        4. Label encoder: fit on training data

    Args:
        hparams: Loaded hyperparameters dictionary

    Returns:
        (train_data, valid_data, label_encoder)
    """
    data_folder = hparams['data_folder']

    # Create datasets from CSV
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams['train_annotation'],
        replacements={'data_root': data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams['valid_annotation'],
        replacements={'data_root': data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams['sample_rate'] * hparams['sentence_len'])

    # Audio pipeline: load and optionally chunk the audio
    @sb.utils.data_pipeline.takes('wav', 'start', 'stop', 'duration')
    @sb.utils.data_pipeline.provides('sig')
    def audio_pipeline(wav, start, stop, duration):
        if hparams.get('random_chunk', False):
            duration_sample = int(float(duration) * hparams['sample_rate'])
            if duration_sample > snt_len_sample:
                start = random.randint(0, duration_sample - snt_len_sample)
                stop = start + snt_len_sample
            else:
                start = 0
                stop = duration_sample
        else:
            start = int(start)
            stop = int(stop)

        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Label pipeline: encode speaker ID to integer
    @sb.utils.data_pipeline.takes('spk_id')
    @sb.utils.data_pipeline.provides('spk_id', 'spk_id_encoded')
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # Fit label encoder on training data
    lab_enc_file = os.path.join(hparams['save_folder'], 'label_encoder.txt')
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key='spk_id',
    )

    # Set output keys
    sb.dataio.dataset.set_output_keys(
        datasets, ['id', 'sig', 'spk_id_encoded']
    )

    return train_data, valid_data, label_encoder


# ── Main Entry Point ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  AAA Engine - ECAPA-TDNN Training")
    print("=" * 65)

    # Enable cuDNN auto-tuner for GPU training
    torch.backends.cudnn.benchmark = True

    # Parse command-line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters
    with open(hparams_file, encoding='utf-8') as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Set seed for reproducibility
    set_seed(hparams.get('seed', 1986))

    # Resolve data folder path
    data_folder = os.path.abspath(hparams['data_folder'])
    hparams['data_folder'] = data_folder

    # ── GPU Information ──────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device:    {device}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Mem:   {gpu_mem:.1f} GB")
        print(f"  CUDA:      {torch.version.cuda}")
    print(f"  Data:      {data_folder}")
    print(f"  Epochs:    {hparams['number_of_epochs']}")
    print(f"  Batch:     {hparams['batch_size']}")
    print(f"  LR:        {hparams['lr']}")
    print(f"  Speakers:  {hparams['out_n_neurons']}")

    # Prepare CSV manifests from folder structure
    run_on_main(
        prepare_csv_from_folder,
        kwargs={
            'data_folder': data_folder,
            'save_folder': hparams['save_folder'],
            'split_ratio': hparams['split_ratio'],
            'sentence_len': hparams['sentence_len'],
        },
    )

    # Create datasets and data pipelines
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams['output_folder'],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Load pretrained weights (embedding model from VoxCeleb)
    pretrained_path = hparams.get('pretrained_path', '')
    if pretrained_path:
        print("\n[Fine-tune] Loading pre-trained VoxCeleb weights...")
        try:
            from speechbrain.pretrained import SpeakerRecognition
            pretrained = SpeakerRecognition.from_hparams(
                source=pretrained_path,
                savedir=os.path.join(hparams['save_folder'], 'pretrained_voxceleb'),
                run_opts={'device': device}
            )
            # Copy embedding model weights from pre-trained to our model
            pretrained_state = pretrained.mods.embedding_model.state_dict()
            hparams['embedding_model'].load_state_dict(pretrained_state)
            print("[Fine-tune] [OK] Pre-trained embedding model loaded")
            del pretrained  # Free memory
            torch.cuda.empty_cache()  # Clear GPU cache after loading
        except Exception as e:
            print(f"[Fine-tune] [!] Could not load pretrained weights: {e}")
            print("[Fine-tune] Training from scratch instead...")

    # Initialize the Brain
    # Force device in run_opts for CPU-only compatibility
    if not torch.cuda.is_available():
        run_opts['device'] = 'cpu'

    speaker_brain = SpeakerBrain(
        modules=hparams['modules'],
        opt_class=hparams['opt_class'],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams['checkpointer'],
    )

    # ── Graceful Interruption Handling ────────────────────────
    # PBS sends SIGTERM before killing a job. Catching this
    # signal lets us save a checkpoint before the process exits.
    def signal_handler(signum, frame):
        sig_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
        print(f"\n[!] Received {sig_name}! Saving checkpoint...")
        try:
            speaker_brain.checkpointer.save_checkpoint(
                meta={'interrupted': True}
            )
            print("[!] Checkpoint saved. Exiting gracefully.")
        except Exception as e:
            print(f"[!] Failed to save checkpoint: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)  # PBS sends SIGTERM
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C

    # Start training
    print("\n" + "=" * 65)
    print("  Starting training...")
    print("=" * 65 + "\n")

    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams['dataloader_options'],
        valid_loader_kwargs=hparams['dataloader_options'],
    )

    # ── Memory Cleanup ───────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print("  [OK] Training complete!")
    print(f"  [OK] Checkpoints saved to: {hparams['save_folder']}")
    print("=" * 65)
