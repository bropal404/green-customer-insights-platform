# LLM implementation

What the LLM does, the model and API, the prompts word for word, the guards, and how it is evaluated. The prompt text
below is copied from `src/gcip/explain/llm.py`; the results from `demo/pipeline/metrics/explanations.json` and
`explanations_gemini_sample.json`.

## Role

Every suggestion first gets a template explanation written from its record: the reason codes and evidence, the
CO2e and money impact, the cost, and the data date. The LLM may only **rewrite the body of that text** so it reads
more naturally. It never sees or changes the propensity, eligibility or impact, and it never decides what is shown. The
compliance sentence (data date, "prices and your situation can change", the investment disclaimer) is fixed text,
appended after the rewrite. Any rewrite that fails a guard is discarded and the template is kept. Nothing is trained or
fine-tuned.

## Model and API

| | Inference (dashboards, `llm.provider: gemini`) | Offline alternative and the 360-item evaluation (`llm.provider: local`) |
|---|---|---|
| Model | Gemini Flash: `gemini-3.8-flash`, falling back to `gemini-3.5-flash`, then `gemini-3.1-flash-lite` when a model's daily quota is used up (HTTP 429) | `Qwen/Qwen2.5-1.5B-Instruct` (Hugging Face `transformers`, float16 on a 6 GB laptop GPU) |
| Access | REST `generateContent` over HTTPS, Python standard library only; key in `GEMINI_API_KEY` (`gcip/.env`) | local weights, `pip install -r requirements-llm.txt` |
| Decoding | temperature 0, thinking budget 0, at most 1,024 output tokens | greedy (`do_sample=False`), at most 300 new tokens, batches of 16 |
| Robustness | 30 s timeout, network errors retried, then the template; replies cached on disk (`<run-dir>/explanations/llm_cache.jsonl`) | runs offline; nothing leaves the machine |
| Data sent | the template body of one suggestion (spending, mileage, CO2e, savings): no name, customer ID, account or address | none |

Configuration: `configs/pipeline.yaml` (`llm:`). Without a key or a GPU, `llm.provider: none` keeps every template.

## Prompt

Four messages per suggestion: the system prompt; a worked example for the same product (a draft and its rewrite); then
this suggestion's draft.

```
system:    <system prompt>
user:      Draft:\n<worked example draft for this product>
assistant: <worked example rewrite>
user:      Draft:\n<this suggestion's template body>
```

**System prompt**

> You rewrite the opening of a bank's explanation of a green product suggestion so it reads clearly and warmly. Rules: use only facts in the draft; keep every number exactly as written, with the same digits, pound signs and units; say 'could' or 'about', never 'will' or 'you'll'; no greeting, no exclamation marks, no dates, no disclaimers, no new claims; speak to the customer as 'you'; plain British English; two to four sentences, at most 70 words. Reply with the rewritten text only.

**Worked examples, one per product** (one solar-only example made EV rewrites turn 12-month fuel spend into "each
month" and drop figures; per-product examples fixed it)

*GP01 EV Finance*

- Draft: Why EV Finance could suit you: you spent £1,850 on road fuel in the last 12 months, you drive about 9,800 miles a year, estimated from the fuel you bought and you pay for a car (fuel, insurance, tax, servicing or finance). Battery-electric car for the same distance could cut about 2.6 tonnes of CO2e a year (about 10.4 tonnes over 4 years). Running costs would be about £1,050 a year less at today's prices.
- Rewrite: You spent £1,850 on road fuel in the last 12 months and drive about 9,800 miles a year, so EV Finance could suit you. A battery-electric car for the same distance could cut about 2.6 tonnes of CO2e a year, about 10.4 tonnes over 4 years, and your running costs could be about £1,050 a year less at today's prices.

*GP02 Green Home Loan*

- Draft: Why Green Home Loan could suit you: you spent £2,400 on DIY and tradespeople in the last 12 months and you use about 15,200 kWh of gas a year (local average 11,000). Air-source heat pump replacing a fossil boiler could cut about 2.4 tonnes of CO2e a year (about 36.0 tonnes over 15 years). Running costs would be about £90 a year less at today's prices. The upfront cost is about £4,500.
- Rewrite: You spent £2,400 on DIY and tradespeople in the last 12 months and use about 15,200 kWh of gas a year, above the local average of 11,000, so a Green Home Loan could suit you. Replacing your fossil boiler with an air-source heat pump could cut about 2.4 tonnes of CO2e a year, about 36.0 tonnes over 15 years. Running costs could be about £90 a year less at today's prices, with an upfront cost of about £4,500.

*GP03 Solar Finance*

- Draft: Why Solar Finance could suit you: you own your home and you use about 4,200 kWh of electricity a year. 4 kWp rooftop solar could cut about 640 kg of CO2e a year (about 16.0 tonnes over 25 years). Running costs would be about £640 a year less at today's prices. The upfront cost is about £7,000, paying back in about 11 years.
- Rewrite: You own your home and use about 4,200 kWh of electricity a year, so Solar Finance could be a good fit. A 4 kWp rooftop system could cut about 640 kg of CO2e a year, about 16.0 tonnes over 25 years, and lower your running costs by about £640 a year at today's prices. It costs about £7,000 upfront and could pay back in about 11 years.

*GP04 Paperless Banking*

- Draft: Why Paperless Banking could suit you: you have 2 account(s) still on paper statements and you log in about 9 times a month. Online statements for every account could cut about 0.4 kg of CO2e a year.
- Rewrite: You still get paper statements for 2 accounts and log in about 9 times a month, so Paperless Banking could suit you. Switching every account to online statements could cut about 0.4 kg of CO2e a year.

*GP05 Green Investments*

- Draft: Why Green Investments could suit you: you invested in 12 of the last 12 months and you told us sustainability matters to how you invest. A quarter of investments moved to a sustainability-focused fund could cut about 850 kg of CO2e a year.
- Rewrite: You invested in 12 of the last 12 months and have told us sustainability matters to how you invest, so Green Investments could suit you. Moving a quarter of your investments to a sustainability-focused fund could cut about 850 kg of CO2e a year.

*GP06 Green Loans*

- Draft: Why Green Loans could suit you: you spent £640 at bike or appliance shops in the last 12 months and you already borrow with a loan or card. E-bike replacing short car trips could cut about 410 kg of CO2e a year (about 2.0 tonnes over 5 years). Running costs would be about £265 a year less at today's prices. The upfront cost is about £2,000, paying back in about 8 years.
- Rewrite: You spent £640 at bike or appliance shops in the last 12 months and already borrow with a loan or card, so Green Loans could suit you. An e-bike replacing short car trips could cut about 410 kg of CO2e a year, about 2.0 tonnes over 5 years, and lower your running costs by about £265 a year at today's prices. It costs about £2,000 upfront and could pay back in about 8 years.

## Guards (`src/gcip/explain/guards.py`)

A rewrite is accepted only if all of these hold; otherwise the template is shown.

- every number in the draft is kept (same digits), and the pound sign stays on every amount;
- no number appears that is not in the facts (the 2 in CO2e is not counted as a figure);
- a yearly or 12-month figure is not turned into a monthly one;
- no banned claim: guarantees and promises ("will", "you'll", "guarantee", "will save", "definitely"), approval
  ("pre-approved", "approved", "you qualify"), "risk-free", investment advice ("best investment", "recommend you
  invest"), the bank speaking of the customer's things as its own ("our investments", "we've been investing"), and hype
  ("perfect", "ideal", "amazing", "great news");
- the data date and any required disclaimer are present (they are appended, so this checks the final text);
- at most 120 words.

## Evaluation approach

- **Evaluation set:** 360 live suggestions, 60 per product, sampled with a fixed seed (`explanations/eval_set.parquet`).
- **Per item:** the guards above (pass or fail, with the failure reason), word count and Flesch reading ease, compared
  with the template.
- **Headline metric:** guard pass rate, overall and by product. A pass means the rewrite keeps every fact. Reading ease
  shows whether the rewrite is easier to read than the template.
- **Templates are checked too:** 5,437 of 5,437 templates pass every guard.

## Results

- **Local Qwen2.5-1.5B, 360 items:**
  - **89.7% pass**, by product: EV Finance 92%, Green Home Loan 88%, Solar Finance 97%, Paperless Banking 93%, Green Investments 93%, Green Loans 75%.
  - Failures: numbers dropped 36, banned claim 5, currency symbol dropped 3, numbers not in the facts 2. Every failure fell back to the template.
  - Mean length: LLM 71.9 words, template 75.6. Reading ease: LLM
    67.3, template 73.1.
  - The templates read slightly more easily, so they remain the default and the LLM rewrite is offered on demand.
- **Gemini API sample** (gemini-3.5-flash, gemini-3.1-flash-lite; 2026-10-11 07:10): **12 of 12 pass**,
  two per product. The sample is small because a free-tier key allows about 20 requests a day per model. Example
  (GP01):
  > You spent £2,574 on road fuel in the last 12 months and drive about 13,884 miles a year, so EV Finance could suit you. A battery-electric car for the same distance could cut about 3.8 tonnes of CO2e a year, or about 15.1 tonnes over 4 years. Running costs could also be about £1,519 a year less at today's prices. These are estimates from your account data up to 2026-09-30; prices and your situation can change.
- **How the prompt and guards got here:**
  1. The first local run passed 0 of 360, because the model reformatted the date and paraphrased the caveat. So the
     compliance sentence was moved outside the LLM's reach.
  2. A run then passed 97%, but reading the rewrites showed "£1,337 on fuel each month" (really 12 months) and an
     invented payback period. So the completeness and time-period guards were added.
  3. EV Finance passed only 22% with a single solar example. Per-product examples raised it to 92%.
  4. Green Investments rewrites spoke as the bank ("our investments"). Second-person evidence sentences and the "our
     ..." guard fixed it.

## Run it

```bash
cp .env.example .env                       # set GEMINI_API_KEY for the Gemini provider
.venv/bin/python scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline --stages explain_llm evaluate report \
    --force explain_llm evaluate report --llm-provider local     # or gemini
```

In the dashboards, the "Rewrite" button on a suggestion calls the configured provider for that one explanation.
