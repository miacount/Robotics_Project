from huggingface_hub import snapshot_download
"""fp16 original model takes lots of time to download. so we will use already quantized model by meta"""
MODEL_REPO = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
SAVE_PATH = "models/Llama-3.2-3B-4bit"

print(f"Downloading {MODEL_REPO} to {SAVE_PATH}...")
snapshot_download(repo_id=MODEL_REPO, local_dir=SAVE_PATH)
print("Done.")
