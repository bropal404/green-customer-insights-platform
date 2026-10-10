"""Optional LLM rephrasing of each template explanation (FR5), run offline on a local GPU, checked by gcip.explain.guards.

Nothing is trained and nothing leaves the machine: an open instruction model (configs/pipeline.yaml `llm.model`, default
Qwen2.5-1.5B-Instruct, fits a 6 GB laptop GPU in fp16) sees only the structured facts and the template draft. Anything
that fails a guard falls back to the template, so every customer still gets a correct explanation. On a multi-GPU node,
`python -m gcip.explain.run` under torchrun gives each rank every WORLD_SIZE-th record.
"""

from __future__ import annotations

import os
import threading

SYSTEM = ("You rewrite the opening of a bank's explanation of a green product suggestion so it reads clearly and warmly. "
          "Rules: use only facts in the draft; keep every number exactly as written, with the same digits, pound signs and units; "
          "say 'could' or 'about', never 'will' or 'you'll'; no greeting, no exclamation marks, no dates, no disclaimers, no new "
          "claims; speak to the customer as 'you'; plain British English; two to four sentences, at most 70 words. "
          "Reply with the rewritten text only.")
EXAMPLE_DRAFT = ("Why Solar Finance could suit you: you own your home and about 4,200 kWh of electricity a year. 4 kWp rooftop "
                 "solar could cut about 640 kg of CO2e a year (about 16.0 tonnes over 25 years). Running costs would be about "
                 "£640 a year less at today's prices. The upfront cost is about £7,000, paying back in about 11 years.")
EXAMPLE_REPLY = ("You own your home and use about 4,200 kWh of electricity a year, so Solar Finance could be a good fit. "
                 "A 4 kWp rooftop system could cut about 640 kg of CO2e a year, about 16.0 tonnes over 25 years, and lower "
                 "your running costs by about £640 a year at today's prices. It costs about £7,000 upfront and could pay "
                 "back in about 11 years.")
TYPOGRAPHY = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                            "‘": "'", "’": "'", "“": '"', "”": '"', "₂": "2", " ": " ",
                            " ": " ", " ": " "})


def messages(facts: dict, draft: str) -> list[dict]:
    """`draft` is the template body (without the fixed compliance sentence, which is appended after)."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Draft:\n" + EXAMPLE_DRAFT},
            {"role": "assistant", "content": EXAMPLE_REPLY},
            {"role": "user", "content": "Draft:\n" + draft}]


def normalise(text: str) -> str:
    """Plain typography: models may write non-breaking hyphens in dates or CO₂, which the guards would not match."""
    return " ".join(text.translate(TYPOGRAPHY).split())


class LocalExplainer:
    """Loads the model once (on first use) and rephrases in batches; safe to share between threads."""

    def __init__(self, model_name: str, max_new_tokens: int = 220, dtype: str = "float16"):
        self.model_name, self.max_new_tokens, self.dtype = model_name, max_new_tokens, dtype
        self.model = self.tokenizer = self.device = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = getattr(torch, self.dtype) if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch_dtype).to(self.device).eval()

    def generate(self, items: list[dict], batch_size: int = 16) -> list[str]:
        """Greedy rephrasings for items of {'facts', 'draft'}."""
        import torch

        with self.lock:
            if self.model is None:
                self._load()
            out: list[str] = []
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                prompts = [self.tokenizer.apply_chat_template(messages(i["facts"], i.get("body", i["draft"])), tokenize=False,
                                                              add_generation_prompt=True) for i in batch]
                encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    generated = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                                                    pad_token_id=self.tokenizer.pad_token_id)
                out += [normalise(t) for t in self.tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:],
                                                                          skip_special_tokens=True)]
            return out

    def rephrase(self, item: dict) -> str:
        """Rewritten body for one item from gcip.explain.run.build_items."""
        return self.generate([item], batch_size=1)[0]


def generate(items: list[dict], model_name: str, batch_size: int = 8, max_new_tokens: int = 220, dtype: str = "float16") -> list[str]:
    """One-shot helper used by gcip.explain.run."""
    return LocalExplainer(model_name, max_new_tokens, dtype).generate(items, batch_size)
