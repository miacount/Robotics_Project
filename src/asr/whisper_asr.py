"""
Whisper ASR wrapper.
"""
import torch
import whisper


class WhisperASR:
    """
    Pretrained OpenAI Whisper model for speech-to-text.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        device: str = None,
    ):
        """
        Args:
            model_size: tiny, base, small, medium, large
            language: "en", "ko", or None (auto-detect)
            device: "cuda" or "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language

        print(f"[Whisper] Loading model: {model_size} on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: path to audio file (.wav, .mp3, .flac, etc.)

        Returns:
            recognized text (str)
        """
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            fp16=(self.device == "cuda"),
        )
        return result["text"].strip()