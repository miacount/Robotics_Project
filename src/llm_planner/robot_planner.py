"""
Robot task planner using fine-tuned LLaMA (LoRA) model.

Converts numeric tier from the speaker pipeline into the named-tier format
expected by the fine-tuned model, then returns a step-by-step action plan.

Tier mapping:
    1 → MAIN_USER         (full access)
    2 → PROTECTED_USER    (dangerous objects / doors blocked)
    3 → UNREGISTERED_USER (all requests refused)
"""

import torch
from unsloth import FastLanguageModel

TIER_MAP = {
    1: "MAIN_USER",
    2: "PROTECTED_USER",
    3: "UNREGISTERED_USER",
}

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

_CHAT_TEMPLATE = (
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


class RobotPlanner:
    """Wraps the fine-tuned LLaMA LoRA model for robot task planning."""

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 1024,
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"[RobotPlanner] Loading model from: {model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        self.tokenizer.chat_template = _CHAT_TEMPLATE
        FastLanguageModel.for_inference(self.model)

        eot_ids = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        eot_id = eot_ids[0] if eot_ids else self.tokenizer.eos_token_id
        self.terminators = list(set([self.tokenizer.eos_token_id, eot_id]))
        print("[RobotPlanner] Model ready.")

    def plan(self, tier: int, task_text: str) -> str:
        """
        Generate a robot action plan.

        Args:
            tier: Numeric tier from the speaker pipeline (1, 2, or 3).
            task_text: Transcribed task description from ASR.

        Returns:
            Robot action plan (or refusal message) as a string.
        """
        tier_name = TIER_MAP.get(tier, "UNREGISTERED_USER")
        user_input = f"[TIER: {tier_name}] {task_text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        raw = self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:], skip_special_tokens=False
        )
        for stop in ["<|eot_id|>", "<|end_of_text|>"]:
            if stop in raw:
                raw = raw.split(stop)[0]
        return raw.strip()
