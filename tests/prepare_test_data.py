"""
LibriSpeech test-clean에서 화자별 샘플 추출.
"""
import os
import urllib.request
import tarfile
import shutil
from collections import defaultdict

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
TAR_PATH = "/tmp/test-clean.tar.gz"
EXTRACT_DIR = "/tmp/librispeech_extract"
OUT_DIR = "recordings"
NUM_SPEAKERS = 2
NUM_UTTERANCES = 5


def main():
    # 1. Download (only if not exists)
    if not os.path.exists(TAR_PATH):
        print(f"[Download] {URL}")
        urllib.request.urlretrieve(URL, TAR_PATH)
        print(f"[Download] Saved to {TAR_PATH}")
    else:
        print("[Download] Already cached.")

    # 2. Extract
    if not os.path.exists(EXTRACT_DIR):
        print(f"[Extract] → {EXTRACT_DIR}")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
    else:
        print("[Extract] Already extracted.")

    # 3. Find speakers
    base = os.path.join(EXTRACT_DIR, "LibriSpeech", "test-clean")
    speakers = sorted(os.listdir(base))[:NUM_SPEAKERS]
    print(f"[Select] Speakers: {speakers}")

    # 4. Copy & convert to wav
    for spk in speakers:
        spk_in_dir = os.path.join(base, spk)
        spk_out_dir = os.path.join(OUT_DIR, f"speaker_{spk}")
        os.makedirs(spk_out_dir, exist_ok=True)

        # Walk subdirs to find .flac files
        flacs = []
        for root, _, files in os.walk(spk_in_dir):
            for f in files:
                if f.endswith(".flac"):
                    flacs.append(os.path.join(root, f))
        flacs = sorted(flacs)[:NUM_UTTERANCES]

        for i, fp in enumerate(flacs, 1):
            # Convert flac → wav using soundfile
            import soundfile as sf
            data, sr = sf.read(fp)
            out_path = os.path.join(spk_out_dir, f"{i:02d}.wav")
            sf.write(out_path, data, sr)
            print(f"  saved → {out_path}")

    print("[Done]")


if __name__ == "__main__":
    main()