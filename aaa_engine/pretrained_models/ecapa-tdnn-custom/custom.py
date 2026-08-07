# =============================================================
# AAA Engine - Custom-Trained ECAPA-TDNN
# Custom module file (required by SpeechBrain from_hparams)
# =============================================================
# This model uses standard SpeechBrain ECAPA-TDNN components.
# No custom classes are needed, but this file must exist for
# the SpeakerRecognition.from_hparams() loading mechanism.
# =============================================================

# Re-export standard SpeechBrain components
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
