"""
ECAPA-TDNN wrapper for speaker embedding extraction.
"""
import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier


class ECAPAEncoder:
    """
    Pretrained ECAPA-TDNN encoder for extracting 192-dim speaker embeddings.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "pretrained_models/spkrec-ecapa-voxceleb",
        sample_rate: int = 16000,
        device: str = None,
    ):
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        self.model.eval()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and resample to target sample rate. Mono."""
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform  # shape: (1, num_samples)

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract a single 192-dim speaker embedding from an audio file.

        Returns:
            np.ndarray of shape (192,)
        """
        waveform = self.load_audio(audio_path)
        # SpeechBrain expects (batch, time)
        embedding = self.model.encode_batch(waveform)
        # shape: (1, 1, 192) → (192,)
        embedding = embedding.squeeze().cpu().numpy()
        return embedding

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))