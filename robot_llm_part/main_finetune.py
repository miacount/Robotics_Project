
CUDA_VISIBLE_DEVICES=0
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FUSED_CE_LOSS"] = "1"
import sys
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
from datetime import datetime
from datasets import Dataset

class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

os.makedirs("output", exist_ok=True)
RUN_LOG_PATH = f"output/run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = Tee(RUN_LOG_PATH)
sys.stderr = sys.stdout

# step 0. check library preparation
import subprocess
result = subprocess.run(["python", "check_library.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("Library check failed. Fix imports before proceeding.")

# step 1. data load

DATASET_PATH = "dataset/train.jsonl"

raw_data = []
with open(DATASET_PATH, "r") as f:
    for line in f:
        raw_data.append(json.loads(line))

# step 2. quantize & load model
r"""
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==
   \\   /|
O^O/ \_/ \
\        /
 "-____-"

UNSLOTH is a tool to use gpu efficiently and fast
- efficiently calculate attention with Triton kernel -> train and infer velocity *2 fast
- use less memory -> VRAM batch size *2
- familiar to QLoRA OR LORA
"""

from unsloth import FastLanguageModel
import torch
import gc
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
max_seq_length = 1024
dtype = None
load_in_4bit = True

LLAMA_MODEL_PATH = "models/Llama-3.2-3B-4bit"

if not os.path.exists(LLAMA_MODEL_PATH):
    print("Quantized model not found. Running quantize_model.py...")
    result = subprocess.run(["python", "download_quantized_llama3.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("download failed.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LLAMA_MODEL_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# base model has no chat template → set Llama3 instruct template manually
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

# step 3. unfinetuned model test
SYSTEM_PROMPT = (
    "You are a robot task planner that enforces user-tier-based access control.\n\n"
    "User tiers and permissions:\n"
    "  MAIN_USER        : Full access. Execute any requested task as planned.\n"
    "  PROTECTED_USER   : Restricted access. Tasks involving dangerous objects\n"
    "                     (e.g. knife, stove burner, candle, microwave, glass items)\n"
    "                     or door operations are partially or fully refused.\n"
    "                     Safe steps before a restricted step are executed normally.\n"
    "                     Execution stops at the first restricted step, followed by\n"
    "                     a Say() action explaining why.\n"
    "  UNREGISTERED_USER: No access. All requests are refused regardless of content.\n\n"
    "Response format when fully permitted:\n"
    "  Step N: <natural language description> → Action(argument)\n\n"
    "Response format when partially blocked (PROTECTED_USER):\n"
    "  Step N: <description> → Action(argument)\n"
    "  STOPPED at Step M: <description> → BlockedAction(argument)\n"
    "  → Say(\"<reason this step is unsafe and what was completed>\")\n\n"
    "Response format when blocked from the first step:\n"
    "  → Say(\"<reason this task cannot be started>\")"
)
TASK = "give me that coffeecan you boil the ice in the pot?"
TIERS = ["MAIN_USER", "PROTECTED_USER", "UNREGISTERED_USER"]

results = []
for tier in TIERS:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[TIER: {tier}] {TASK}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    entry = f"[{tier}]\nOutput: {response}\n"
    print(entry)
    results.append(entry)

with open("unfinetuned_model_result.txt", "w") as f:
    f.write(f"Task: {TASK}\n")
    f.write("=" * 60 + "\n\n")
    f.write("\n".join(results))
print("Saved to unfinetuned_model_result.txt")

# step 1-2. format dataset using loaded tokenizer
import glob
formatted = [
    {"text": tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)}
    for item in raw_data
]
split = int(len(formatted) * 0.9)
train_dataset = Dataset.from_list(formatted[:split])
eval_dataset  = Dataset.from_list(formatted[split:])

FINETUNED_MODEL_PATH = "my_lora"

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback
import gc

if not os.path.exists(FINETUNED_MODEL_PATH):
    print("Fine-tuned model not found. Starting training...")

    # 4. Lora prepare
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    LOG_PATH = f"output/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    class FileLogCallback(TrainerCallback):
        def __init__(self, path):
            self.path = path
            with open(path, "w") as f:
                f.write(f"Training started: {datetime.now()}\n")
                f.write("step,loss,eval_loss,learning_rate\n")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and ("loss" in logs or "eval_loss" in logs):
                with open(self.path, "a") as f:
                    f.write(f"{state.global_step},{logs.get('loss','')},{logs.get('eval_loss','')},{logs.get('learning_rate','')}\n")

    # 5. training prepare
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        callbacks=[FileLogCallback(LOG_PATH), EarlyStoppingCallback(early_stopping_patience=3)],
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=128,
            num_train_epochs=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir="output",
            seed=0,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    )

    # 6. train
    checkpoints = sorted(glob.glob("output/checkpoint-*"))
    trainer_stats = trainer.train(resume_from_checkpoint=checkpoints[-1] if checkpoints else None)

    model.save_pretrained(FINETUNED_MODEL_PATH)
    tokenizer.save_pretrained(FINETUNED_MODEL_PATH)

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model saved to {FINETUNED_MODEL_PATH}")

else:
    print(f"Fine-tuned model found at {FINETUNED_MODEL_PATH}. Skipping training.")
    del model
    gc.collect()
    torch.cuda.empty_cache()

# 7. load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = FINETUNED_MODEL_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

FastLanguageModel.for_inference(model)

# 7.test the robot planning chatbot
VALID_TIERS = ["MAIN_USER", "PROTECTED_USER", "UNREGISTERED_USER"]

eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
eot_id = eot_id[0] if eot_id else tokenizer.eos_token_id
terminators = list(set([tokenizer.eos_token_id, eot_id]))

print("\n" + "="*60)
print("Robot Planning Chatbot")
print(f"Tiers: {', '.join(VALID_TIERS)}")
print("Type 'exit' or 'quit' to stop.")
print("="*60 + "\n")

while True:
    tier = input("Tier: ").strip().upper()
    if tier.lower() in ["exit", "quit"]:
        break
    if tier not in VALID_TIERS:
        print(f"Invalid tier. Choose from: {', '.join(VALID_TIERS)}")
        continue

    task = input("Task: ").strip()
    if task.lower() in ["exit", "quit"]:
        break

    user_input = f"[TIER: {tier}] {task}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=False)
    for stop in ["<|eot_id|>", "<|end_of_text|>"]:
        if stop in raw:
            raw = raw.split(stop)[0]
    assistant_response = raw.strip()

    print("Response:\n" + assistant_response)

print(tokenizer.chat_template)