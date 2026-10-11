"""Optional LLM rephrasing of each template explanation (FR5), checked by gcip.explain.guards.

Nothing is trained. The LLM only rewrites the body of a template explanation; the fixed compliance sentence (data date,
caveat, disclaimer) is appended unchanged, and anything that fails a guard falls back to the template. Two backends:

  gemini  Gemini Flash through the Gemini API (default at inference time; key in GEMINI_API_KEY, e.g. gcip/.env).
          Only the template draft and its facts are sent: no name, customer id, account or transaction.
  local   an open model on the local GPU (Qwen2.5-1.5B-Instruct; `pip install -r requirements-llm.txt`), fully offline.

`make_explainer(cfg)` returns the configured backend, or None when it is not available (templates only).
"""

from __future__ import annotations

import json
import os
import threading

SYSTEM = ("You rewrite the opening of a bank's explanation of a green product suggestion so it reads clearly and warmly. "
          "Rules: use only facts in the draft; keep every number exactly as written, with the same digits, pound signs and units; "
          "say 'could' or 'about', never 'will' or 'you'll'; no greeting, no exclamation marks, no dates, no disclaimers, no new "
          "claims; speak to the customer as 'you'; plain British English; two to four sentences, at most 70 words. "
          "Reply with the rewritten text only.")
# One worked example per product (draft -> rewrite), so the model sees how its own kind of draft keeps every number and
# its period ("in the last 12 months", "a year"); a single solar example left EV rewrites turning 12-month fuel spend
# into "each month" and dropping figures.
EXAMPLES = {
    "GP01": ("Why EV Finance could suit you: you spent £1,850 on road fuel in the last 12 months, you drive about 9,800 miles a "
             "year, estimated from the fuel you bought and you pay for a car (fuel, insurance, tax, servicing or finance). Battery-electric car "
             "for the same distance could cut about 2.6 tonnes of CO2e a year (about 10.4 tonnes over 4 years). Running costs "
             "would be about £1,050 a year less at today's prices.",
             "You spent £1,850 on road fuel in the last 12 months and drive about 9,800 miles a year, so EV Finance could suit "
             "you. A battery-electric car for the same distance could cut about 2.6 tonnes of CO2e a year, about 10.4 tonnes "
             "over 4 years, and your running costs could be about £1,050 a year less at today's prices."),
    "GP02": ("Why Green Home Loan could suit you: you spent £2,400 on DIY and tradespeople in the last 12 months and you use about "
             "15,200 kWh of gas a year (local average 11,000). Air-source heat pump replacing a fossil boiler could cut about 2.4 tonnes of CO2e a year "
             "(about 36.0 tonnes over 15 years). Running costs would be about £90 a year less at today's prices. The upfront "
             "cost is about £4,500.",
             "You spent £2,400 on DIY and tradespeople in the last 12 months and use about 15,200 kWh of gas a year, above the local "
             "average of 11,000, so a Green Home Loan could suit you. Replacing your fossil boiler with an air-source heat pump "
             "could cut about 2.4 tonnes of CO2e a year, about 36.0 tonnes over 15 years. Running costs could be about £90 a "
             "year less at today's prices, with an upfront cost of about £4,500."),
    "GP03": ("Why Solar Finance could suit you: you own your home and you use about 4,200 kWh of electricity a year. 4 kWp rooftop "
             "solar could cut about 640 kg of CO2e a year (about 16.0 tonnes over 25 years). Running costs would be about "
             "£640 a year less at today's prices. The upfront cost is about £7,000, paying back in about 11 years.",
             "You own your home and use about 4,200 kWh of electricity a year, so Solar Finance could be a good fit. A 4 kWp "
             "rooftop system could cut about 640 kg of CO2e a year, about 16.0 tonnes over 25 years, and lower your running "
             "costs by about £640 a year at today's prices. It costs about £7,000 upfront and could pay back in about 11 years."),
    "GP04": ("Why Paperless Banking could suit you: you have 2 account(s) still on paper statements and you log in about 9 times a month. "
             "Online statements for every account could cut about 0.4 kg of CO2e a year.",
             "You still get paper statements for 2 accounts and log in about 9 times a month, so Paperless Banking could suit "
             "you. Switching every account to online statements could cut about 0.4 kg of CO2e a year."),
    "GP05": ("Why Green Investments could suit you: you invested in 12 of the last 12 months and you told us sustainability "
             "matters to how you invest. A quarter of investments moved to a sustainability-focused fund could cut about 850 kg of CO2e "
             "a year.",
             "You invested in 12 of the last 12 months and have told us sustainability matters to how you invest, so Green "
             "Investments could suit you. Moving a quarter of your investments to a sustainability-focused fund could cut "
             "about 850 kg of CO2e a year."),
    "GP06": ("Why Green Loans could suit you: you spent £640 at bike or appliance shops in the last 12 months and you already "
             "borrow with a loan or card. E-bike replacing short car trips could cut about 410 kg of CO2e a year (about 2.0 tonnes over 5 years). "
             "Running costs would be about £265 a year less at today's prices. The upfront cost is about £2,000, paying back "
             "in about 8 years.",
             "You spent £640 at bike or appliance shops in the last 12 months and already borrow with a loan or card, so Green Loans "
             "could suit you. An e-bike replacing short car trips could cut about 410 kg of CO2e a year, about 2.0 tonnes over "
             "5 years, and lower your running costs by about £265 a year at today's prices. It costs about £2,000 upfront and "
             "could pay back in about 8 years."),
}
EXAMPLE_DRAFT, EXAMPLE_REPLY = EXAMPLES["GP03"]
TYPOGRAPHY = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                            "‘": "'", "’": "'", "“": '"', "”": '"', "₂": "2", " ": " ",
                            " ": " ", " ": " "})


def messages(facts: dict, draft: str, product: str | None = None) -> list[dict]:
    """`draft` is the template body (without the fixed compliance sentence, which is appended after); the worked example
    is the one for the same product."""
    example_draft, example_reply = EXAMPLES.get(product, (EXAMPLE_DRAFT, EXAMPLE_REPLY))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Draft:\n" + example_draft},
            {"role": "assistant", "content": example_reply},
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
                prompts = [self.tokenizer.apply_chat_template(messages(i["facts"], i.get("body", i["draft"]), i.get("product")), tokenize=False,
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


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]


def load_env(path=ROOT / ".env") -> None:
    """KEY=value lines from gcip/.env (git-ignored) into the environment, without overriding what is already set."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


class GeminiExplainer:
    """Gemini Flash over HTTPS (standard library only), temperature 0. When one model's quota is used up (HTTP 429
    "quota"), the next model in `models` is tried. Replies are cached on disk, so the same rewrite never costs twice."""

    def __init__(self, models: list[str], max_new_tokens: int = 300, timeout: float = 30.0, cache=None):
        load_env()
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in gcip/.env)")
        self.models, self.max_new_tokens, self.timeout = list(models), max_new_tokens, timeout
        self.exhausted: set[str] = set()
        self.cache_path = cache
        self.cache: dict[str, str] = {}
        self.lock = threading.Lock()
        if cache is not None and cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["text"]
        self.last_model: str | None = None

    def _body(self, item: dict) -> dict:
        msgs = messages(item["facts"], item.get("body", item["draft"]), item.get("product"))
        role = {"user": "user", "assistant": "model"}
        return {"systemInstruction": {"parts": [{"text": msgs[0]["content"]}]},
                "contents": [{"role": role[m["role"]], "parts": [{"text": m["content"]}]} for m in msgs[1:]],
                # thinking off: the answer is a short rewrite, and thinking tokens would eat the output allowance
                "generationConfig": {"temperature": 0, "maxOutputTokens": max(self.max_new_tokens, 1024),
                                     "thinkingConfig": {"thinkingBudget": 0}}}

    def _call(self, model: str, body: dict) -> str:
        import time
        import urllib.error
        import urllib.request

        request = urllib.request.Request(GEMINI_URL.format(model=model), data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    reply = json.loads(response.read())
                parts = reply["candidates"][0]["content"].get("parts", [])
                return normalise("".join(p.get("text", "") for p in parts if not p.get("thought")))
            except urllib.error.HTTPError as err:
                detail = err.read()[:400].decode(errors="replace")
                if err.code == 429 and "quota" in detail.lower():
                    raise QuotaExhausted(model) from None
                if err.code in (429, 500, 503) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini API error {err.code}: {detail[:200]}") from None
            except (TimeoutError, urllib.error.URLError, OSError, KeyError, IndexError) as err:  # network or odd reply
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini API did not answer: {err}") from None
        raise RuntimeError("Gemini API did not answer")

    def rephrase(self, item: dict) -> str:
        import hashlib

        body = self._body(item)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        for model in self.models:
            if model in self.exhausted:
                continue
            try:
                text = self._call(model, body)
            except QuotaExhausted:
                self.exhausted.add(model)
                continue
            self.last_model = model
            with self.lock:
                self.cache[key] = text
                if self.cache_path is not None:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.cache_path.open("a") as fh:
                        fh.write(json.dumps({"key": key, "model": model, "text": text}) + "\n")
            return text
        raise RuntimeError("the Gemini quota for this API key is used up for today (all configured models)")

    def generate(self, items: list[dict], batch_size: int = 4) -> list[str | None]:
        """Rewrites for many items, `batch_size` requests at a time; a failed request gives None (template kept)."""
        from concurrent.futures import ThreadPoolExecutor

        def one(item):
            try:
                return self.rephrase(item)
            except RuntimeError:
                return None
        with ThreadPoolExecutor(max_workers=max(1, batch_size)) as pool:
            return list(pool.map(one, items))


class QuotaExhausted(Exception):
    pass


def make_explainer(cfg: dict, cache=None):
    """The configured LLM backend (`llm.provider`: gemini, local or none), or None when it cannot run here."""
    provider = cfg.get("provider", "none")
    try:
        if provider == "gemini":
            return GeminiExplainer([cfg["model"], *cfg.get("fallback_models", [])], cfg.get("max_new_tokens", 300), cache=cache)
        if provider == "local":
            import torch  # noqa: F401  (fails fast when the optional packages are not installed)
            return LocalExplainer(cfg.get("local_model", "Qwen/Qwen2.5-1.5B-Instruct"), cfg.get("max_new_tokens", 220),
                                  cfg.get("dtype", "float16"))
    except (RuntimeError, ImportError):
        return None
    return None
