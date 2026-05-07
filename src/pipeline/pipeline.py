"""
End-to-end pipeline: audio → (Tier, Text) → prompt for LLM.
"""
import os
import glob
import numpy as np
from typing import Tuple, Optional

from src.speaker_verification.ecapa import ECAPAEncoder
from src.asr.whisper_asr import WhisperASR


class SpeakerGatedPipeline:
    """
    음성 입력을 받아 (Tier, 인식 텍스트) 를 추출하고,
    LLM에 넘길 프롬프트 형태로 변환.
    """

    def __init__(
        self,
        encoder: ECAPAEncoder,
        asr: WhisperASR,
        enrolled_dir: str = "data/enrolled",
        threshold: float = 0.25,
        default_tier: int = 3,
    ):
        self.encoder = encoder
        self.asr = asr
        self.enrolled_dir = enrolled_dir
        self.threshold = threshold
        self.default_tier = default_tier

        # Load all enrolled speakers
        self.enrolled = self._load_enrolled()
        print(f"[Pipeline] Loaded {len(self.enrolled)} enrolled speakers: "
              f"{[s['name'] for s in self.enrolled]}")

    def _load_enrolled(self) -> list:
        """
        data/enrolled/ 디렉토리의 모든 .npz 파일 로드.

        각 항목: {"name": str, "tier": int, "embedding": np.ndarray}
        """
        enrolled = []
        if not os.path.isdir(self.enrolled_dir):
            return enrolled

        for fpath in sorted(glob.glob(os.path.join(self.enrolled_dir, "*.npz"))):
            data = np.load(fpath, allow_pickle=True)
            enrolled.append({
                "name": str(data["name"]),
                "tier": int(data["tier"]),
                "embedding": data["embedding"],
            })
        return enrolled

    def identify_speaker(
        self, audio_path: str
    ) -> Tuple[int, Optional[str], float]:
        """
        화자 식별.

        Returns:
            (tier, matched_name, best_similarity)
            매치되는 화자가 없으면 (default_tier, None, best_similarity)
        """
        if len(self.enrolled) == 0:
            return self.default_tier, None, 0.0

        query_emb = self.encoder.extract_embedding(audio_path)

        best_sim = -1.0
        best_speaker = None
        for spk in self.enrolled:
            sim = ECAPAEncoder.cosine_similarity(query_emb, spk["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_speaker = spk

        if best_sim >= self.threshold:
            return best_speaker["tier"], best_speaker["name"], best_sim
        else:
            return self.default_tier, None, best_sim

    def transcribe(self, audio_path: str) -> str:
        """음성 → 텍스트."""
        return self.asr.transcribe(audio_path)

    def run(self, audio_path: str) -> dict:
        """
        End-to-end: audio → {tier, text, prompt, ...}.

        Returns:
            {
                "tier": int,
                "speaker": str or None,
                "similarity": float,
                "text": str,
                "prompt": str,    # "[TIERn] text"
            }
        """
        tier, speaker, sim = self.identify_speaker(audio_path)
        text = self.transcribe(audio_path)
        prompt = f"[TIER{tier}] {text}"

        return {
            "tier": tier,
            "speaker": speaker,
            "similarity": sim,
            "text": text,
            "prompt": prompt,
        }