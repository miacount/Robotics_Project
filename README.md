# Speaker-Gated Home Robotics

화자 식별(Speaker Verification) + 음성 인식(ASR) + LLM 태스크 플래너를 결합한 가정용 서비스 로봇 권한 제어 시스템.

음성이 입력되면:
1. **ECAPA-TDNN** → 등록된 화자 중 누구인지 식별 → Tier 결정
2. **Whisper** → 음성을 텍스트로 변환
3. **Fine-tuned LLaMA-3.2-3B (LoRA)** → Tier 기반 접근 제어 + 로봇 동작 시퀀스 생성

---

## Project Structure

```
Robotics_Project/
├── config/
│   └── config.yaml                  # 설정 (모델, threshold, tier, LLM 경로 등)
├── enrollment/
│   └── enroll.py                    # 화자 등록 스크립트
├── data/
│   └── enrolled/                    # 저장된 화자 embedding (.npz)
├── src/
│   ├── speaker_verification/
│   │   └── ecapa.py                 # ECAPA-TDNN wrapper
│   ├── asr/
│   │   └── whisper_asr.py           # Whisper wrapper
│   ├── pipeline/
│   │   └── pipeline.py              # Stage 1 통합 파이프라인 (SV + ASR)
│   └── llm_planner/
│       └── robot_planner.py         # Stage 2 LLM 태스크 플래너
├── demo/
│   ├── run_demo.py                  # Stage 1만 실행 (SV + ASR)
│   └── run_full_demo.py             # 전체 파이프라인 (SV + ASR + LLM)
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md

../robot_llm_part/                   # LLM 파인튜닝 프로젝트 (형제 디렉토리)
    ├── my_lora/                     # 파인튜닝된 LoRA 어댑터 (기본 model_path)
    ├── dataset/train.jsonl          # 학습 데이터 4,503개
    └── main_finetune.py             # 학습 스크립트
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
| Unsloth | latest |

---

## Setup (Docker)

### 1. Build image
```bash
docker build -t speaker-gated-robotics .
```

### 2. Run container
```bash
docker run --gpus all -it --rm \
    -v $(pwd)/..:/workspace \
    speaker-gated-robotics
```

> `-v $(pwd)/..:/workspace` — `robot_llm_part/my_lora` 경로 접근을 위해 상위 디렉토리를 마운트

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
# Tier 1 (parent — full access)
python enrollment/enroll.py \
    --name geunyoung --tier 1 \
    --audio_dir ./recordings/geunyoung

# Tier 2 (child/elderly — restricted)
python enrollment/enroll.py \
    --name daeun --tier 2 \
    --audio_dir ./recordings/daeun
```

→ `data/enrolled/{name}.npz` 생성됨.

### Step 2-A. Stage 1만 실행 (SV + ASR)

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

### Step 2-B. 전체 파이프라인 실행 (SV + ASR + LLM)

```bash
python demo/run_full_demo.py --audio ./test.wav
```

출력 예시:
```
[ Final Result ]
  Speaker    : geunyoung
  Tier       : 1
  Task       : Turn on the gas stove.

  Robot Action Plan:
    Step 1: Go to the kitchen → GotoLocation(kitchen)
    Step 2: Turn on the stove → ToggleObject(stove)
```

---

## Tier Definition

| Tier | Role | Permissions |
|---|---|---|
| 1 | Parent (MAIN_USER) | All commands |
| 2 | Child / Elderly (PROTECTED_USER) | Safe commands only — knife, stove, candle, microwave, glass, doors 차단 |
| 3 | Unauthorized (UNREGISTERED_USER) | All requests rejected |

---

## Full Pipeline

```
audio.wav
   │
   ├──→ ECAPA-TDNN ──→ 192-dim embedding ──→ cosine sim ──→ Tier (1/2/3)
   │
   └──→ Whisper ─────→ recognized text
                                 │
              tier + text        │
                                 ▼
              Fine-tuned LLaMA-3.2-3B LoRA (robot_llm_part/my_lora)
              [TIER: MAIN_USER / PROTECTED_USER / UNREGISTERED_USER]
                                 │
                                 ▼
              Step-by-step robot action plan  (or refusal)
```