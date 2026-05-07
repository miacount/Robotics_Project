"""
End-to-end inference demo.

사용 예시:
    python demo/run_demo.py --audio ./test.wav
"""
import os
import sys
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speaker_verification.ecapa import ECAPAEncoder
from src.asr.whisper_asr import WhisperASR
from src.pipeline.pipeline import SpeakerGatedPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sv_cfg = cfg["speaker_verification"]
    asr_cfg = cfg["asr"]
    enroll_cfg = cfg["enrollment"]
    tier_cfg = cfg["tiers"]

    # Init modules
    print("=" * 60)
    print("Initializing modules...")
    print("=" * 60)

    encoder = ECAPAEncoder(
        model_source=sv_cfg["model_source"],
        savedir=sv_cfg["savedir"],
        sample_rate=sv_cfg["sample_rate"],
    )

    asr = WhisperASR(
        model_size=asr_cfg["model_size"],
        language=asr_cfg["language"],
    )

    pipeline = SpeakerGatedPipeline(
        encoder=encoder,
        asr=asr,
        enrolled_dir=enroll_cfg["enrolled_dir"],
        threshold=sv_cfg["threshold"],
        default_tier=tier_cfg["default_tier"],
    )

    # Run inference
    print()
    print("=" * 60)
    print(f"Input: {args.audio}")
    print("=" * 60)

    result = pipeline.run(args.audio)

    print()
    print("=" * 60)
    print("[ Result ]")
    print(f"  Speaker     : {result['speaker']}")
    print(f"  Similarity  : {result['similarity']:.4f}")
    print(f"  Tier        : {result['tier']}")
    print(f"  Text        : {result['text']}")
    print(f"  → Prompt    : {result['prompt']}")
    print("=" * 60)


if __name__ == "__main__":
    main()