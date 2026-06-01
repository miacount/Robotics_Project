import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import peft, accelerate, bitsandbytes

print("All imports OK!")