# Speaker-Gated Home Robotics

화자 식별(Speaker Verification) + 음성 인식(ASR) 기반의 가정용 서비스 로봇 권한 제어 시스템.

음성이 입력되면:
1. **ECAPA-TDNN** → 등록된 화자 중 누구인지 식별 → Tier 결정
2. **Whisper** → 음성을 텍스트로 변환
3. 두 결과를 합쳐 `[TIERn] text` 형태의 프롬프트 생성 → LLM(LLaMA)으로 전달

---

## Project Structure

```
Robotics_Project/
├── config/
│   └── config.yaml              # 설정 (모델, threshold, tier 등)
├── enrollment/
│   └── enroll.py                # 화자 등록 스크립트
├── data/
│   └── enrolled/                # 저장된 화자 embedding (.npz)
├── src/
│   ├── speaker_verification/
│   │   └── ecapa.py             # ECAPA-TDNN wrapper
│   ├── asr/
│   │   └── whisper_asr.py       # Whisper wrapper
│   └── pipeline/
│       └── pipeline.py          # 통합 파이프라인
├── demo/
│   └── run_demo.py              # End-to-end 데모
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Environment

| Component | Version |
|---|---|
| GPU | RTX 2080 (8GB) |
| CUDA | 12.2 |
| Python | 3.10 |
| PyTorch | 2.1.2 (cu121) |
| SpeechBrain | 1.0.0 |
| Whisper | 20231117 |

---

## Setup (Docker)

### 1. Build image
```bash
docker build -t speaker-gated-robotics .
```

### 2. Run container
```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    speaker-gated-robotics
```

---

## Usage

### Step 1. 화자 등록 (Enrollment)

녹음한 wav 파일들을 화자별 폴더에 둠:
```
recordings/
├── geunyoung/
│   ├── 01.wav
│   ├── 02.wav
│   └── 03.wav
└── daeun/
    ├── 01.wav
    └── ...
```

등록:
```bash
# Tier 1 (parent)
python enrollment/enroll.py \
    --name geunyoung --tier 1 \
    --audio_dir ./recordings/geunyoung

# Tier 2 (child/elderly)
python enrollment/enroll.py \
    --name daeun --tier 2 \
    --audio_dir ./recordings/daeun
```

→ `data/enrolled/{name}.npz` 생성됨.

### Step 2. Inference

```bash
python demo/run_demo.py --audio ./test.wav
```

출력 예시:
```
[ Result ]
  Speaker     : geunyoung
  Similarity  : 0.7421
  Tier        : 1
  Text        : Turn on the gas stove.
  → Prompt    : [TIER1] Turn on the gas stove.
```

이 `prompt` 가 다음 단계(LLaMA 8B Robot Planner)로 전달됨.

---

## Tier Definition

| Tier | Role | Permissions |
|---|---|---|
| 1 | Parent | All commands |
| 2 | Child / Elderly | Basic commands only |
| 3 | Unauthorized (default) | Rejected |

---

## Pipeline

```
audio.wav
   │
   ├──→ ECAPA-TDNN ──→ 192-dim embedding ──→ cosine sim ──→ Tier
   │
   └──→ Whisper ─────→ recognized text
                                            │
                            "[TIERn] text"  │
                                            ▼
                                     LLaMA 8B (next stage)
```