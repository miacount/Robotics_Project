"""
Full end-to-end demo: audio → speaker ID → ASR → LLM task planner.

Stage 1 (Robotics_Project):
    audio  →  ECAPA-TDNN speaker verification  →  tier assignment
           →  Whisper ASR                       →  text transcription
           →  SpeakerGatedPipeline              →  {"tier": int, "text": str, ...}

Stage 2 (robot_llm_part):
    (tier, text)  →  fine-tuned LLaMA-3.2-3B LoRA  →  robot action plan

Usage:
    python demo/run_full_demo.py --audio ./test.wav
    python demo/run_full_demo.py --audio ./test.wav --config config/config.yaml
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speaker_verification.ecapa import ECAPAEncoder
from src.asr.whisper_asr import WhisperASR
from src.pipeline.pipeline import SpeakerGatedPipeline
from src.llm_planner.robot_planner import RobotPlanner


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: audio → speaker ID + ASR → LLM robot task planner"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sv_cfg = cfg["speaker_verification"]
    asr_cfg = cfg["asr"]
    enroll_cfg = cfg["enrollment"]
    tier_cfg = cfg["tiers"]
    llm_cfg = cfg["llm"]

    # Resolve model_path relative to the project root (where config lives)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.normpath(os.path.join(project_root, llm_cfg["model_path"]))

    print("=" * 60)
    print("  Stage 1: Speaker Verification + ASR")
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
        enrolled_dir=os.path.join(project_root, enroll_cfg["enrolled_dir"]),
        threshold=sv_cfg["threshold"],
        default_tier=tier_cfg["default_tier"],
    )

    result = pipeline.run(args.audio)

    print()
    print(f"  Speaker    : {result['speaker'] or 'Unknown'}")
    print(f"  Similarity : {result['similarity']:.4f}")
    print(f"  Tier       : {result['tier']}")
    print(f"  Text       : {result['text']}")

    print()
    print("=" * 60)
    print("  Stage 2: LLM Robot Task Planner")
    print("=" * 60)

    planner = RobotPlanner(
        model_path=model_path,
        max_seq_length=llm_cfg["max_seq_length"],
        load_in_4bit=llm_cfg["load_in_4bit"],
        max_new_tokens=llm_cfg["max_new_tokens"],
        temperature=llm_cfg["temperature"],
        top_p=llm_cfg["top_p"],
    )

    plan = planner.plan(tier=result["tier"], task_text=result["text"])

    print()
    print("=" * 60)
    print("  [ Final Result ]")
    print("=" * 60)
    print(f"  Speaker    : {result['speaker'] or 'Unknown'}")
    print(f"  Tier       : {result['tier']}")
    print(f"  Task       : {result['text']}")
    print()
    print("  Robot Action Plan:")
    for line in plan.splitlines():
        print(f"    {line}")
    print("=" * 60)


if __name__ == "__main__":
    main()
