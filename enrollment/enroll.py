"""
Speaker enrollment script.

여러 발화에서 추출한 embedding을 평균내서 한 화자의 대표 embedding으로 저장.
사용 예시:
    python enrollment/enroll.py \
        --name geunyoung \
        --tier 1 \
        --audio_dir ./recordings/geunyoung
"""
import os
import sys
import argparse
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speaker_verification.ecapa import ECAPAEncoder


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a")


def collect_audio_files(audio_dir: str) -> list:
    """주어진 디렉토리에서 오디오 파일 경로 리스트 반환."""
    files = []
    for fname in sorted(os.listdir(audio_dir)):
        if fname.lower().endswith(AUDIO_EXTS):
            files.append(os.path.join(audio_dir, fname))
    return files


def enroll_speaker(
    name: str,
    tier: int,
    audio_dir: str,
    enrolled_dir: str,
    encoder: ECAPAEncoder,
    min_utterances: int = 3,
):
    """
    한 화자의 여러 발화를 받아서 평균 embedding을 저장.

    저장 형식:
        data/enrolled/{name}.npz
            - embedding: (192,) 평균 embedding
            - tier: int
            - name: str
            - num_utterances: int
    """
    audio_files = collect_audio_files(audio_dir)

    if len(audio_files) < min_utterances:
        raise ValueError(
            f"At least {min_utterances} utterances required. "
            f"Found {len(audio_files)} in {audio_dir}"
        )

    print(f"[Enroll] Speaker: {name} | Tier: {tier}")
    print(f"[Enroll] Found {len(audio_files)} audio files.")

    # Extract embeddings
    embeddings = []
    for i, fpath in enumerate(audio_files, 1):
        print(f"  ({i}/{len(audio_files)}) {os.path.basename(fpath)}")
        emb = encoder.extract_embedding(fpath)
        embeddings.append(emb)

    # Mean embedding
    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)

    # Save
    os.makedirs(enrolled_dir, exist_ok=True)
    save_path = os.path.join(enrolled_dir, f"{name}.npz")
    np.savez(
        save_path,
        embedding=mean_embedding,
        tier=tier,
        name=name,
        num_utterances=len(audio_files),
    )
    print(f"[Enroll] Saved → {save_path}")
    print(f"[Enroll] Embedding shape: {mean_embedding.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Speaker name (used as filename)")
    parser.add_argument("--tier", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing speaker's utterances")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sv_cfg = cfg["speaker_verification"]
    enroll_cfg = cfg["enrollment"]

    # Init encoder
    encoder = ECAPAEncoder(
        model_source=sv_cfg["model_source"],
        savedir=sv_cfg["savedir"],
        sample_rate=sv_cfg["sample_rate"],
    )

    # Enroll
    enroll_speaker(
        name=args.name,
        tier=args.tier,
        audio_dir=args.audio_dir,
        enrolled_dir=enroll_cfg["enrolled_dir"],
        encoder=encoder,
        min_utterances=enroll_cfg["min_utterances"],
    )


if __name__ == "__main__":
    main()