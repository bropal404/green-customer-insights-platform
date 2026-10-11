# Responsible AI note

How the platform handles privacy, bias, explainability, hallucination, human oversight and the limits of its data.
Numbers are from the 30,000-customer run (`demo/pipeline/`); the fairness and evaluation tables are in
`docs/EVALUATION.md`.

## Guardrails that hold everywhere

Each output has one owner, and nothing downstream may change it (the brief's guardrails):

| Output | Owner | Cannot be changed by |
|---|---|---|
| Propensity and confidence | FR3 models (calibrated LightGBM) | memory, the LLM, the dashboards |
| Eligibility | the bank's product rules (`configs/products.yaml`) | the models, memory, the LLM |
| CO2e and money impact | FR4 carbon engine (DESNZ 2026 factors, Ofgem prices) | the models, memory, the LLM |

Memory changes only the order and timing of suggestions (priority = propensity x confidence x memory weight). It never
changes the propensity. The LLM rewrites wording only, and a rewrite that changes any fact is thrown away.

## Privacy

- **No real personal data.** All customers, accounts and transactions are synthetic, generated from public UK statistics
  (Census 2021, DESNZ NEED, DVSA MOT, Ofgem). No real banking data was used.
- **Data the bank already holds, used for its stated purpose.** Inputs are the bank's own records: Customer 360,
  transactions, product holdings and campaign responses. Digital-analytics events are keyed by a salted SHA-256
  pseudonym, not the customer ID.
- **Consent.** No suggestion is made without marketing consent. 21,498 customer x product pairs are withheld for this
  reason, and the decision is recorded in `decisions.parquet`.
- **Minimisation.** Each product's model uses a curated list of 21 to 40 inputs (`curated_columns`), not the whole table.
- **What leaves the bank.** Only the Gemini API call leaves the machine. It receives the wording of one suggestion: spending,
  mileage, estimated CO2e and savings, and the data date. It never receives a name, customer ID, account number or
  address. Replies are cached locally (`<run-dir>/explanations/llm_cache.jsonl`).
  - The prototype uses a free-tier key, and free-tier prompts may be used by the provider.
  - A production deployment would use a contract with no data retention, or `llm.provider: local`. That option runs
    the same prompts on a local Qwen2.5-1.5B so nothing leaves the bank's network.
- **Customers see only themselves.** The customer app shows one customer's own suggestions. The staff dashboard shows
  aggregates, plus individual records for staff.

## Bias and fairness

- **Measured at the threshold that decides who sees a suggestion**, on validation and test, by age band, declared income
  band, region and vulnerability. The full results are in `docs/EVALUATION.md`, "Fairness at the suggestion threshold".
- **Calibration holds within every group.** A group's take-up rate and its mean propensity differ by at most 4.7
  percentage points on test, so a propensity means the same thing whoever it is for.
- **Selection rates differ where take-up differs.**
  - EV Finance on test: 42.4% of customers on £100k+ are above the threshold, against 0% on £15-25k. Their observed
    take-up is 10.6% against 2.0%.
  - Paperless on test: 59.2% of customers aged 18-34 against 6.5% of those aged 55+, with take-up of 59.4% against 7.1%.
  - The model reflects affordability and habits rather than inventing them. But a threshold on likelihood alone means
    lower-income customers are rarely offered EV or solar finance.
  - The bank should decide this as policy, for example by reserving a share of offers, or by leading with Green Loans
    and paperless banking, which reach every income band.
  - The fairness table is regenerated on every run so the effect stays visible. The staff dashboard shows each
    customer's income band and region next to their suggestions.
- **Protected characteristics.** The data holds no gender, ethnicity or disability fields. Age and income band are
  context features: they may inform a score but are never a reason code shown to anyone (`gcip.models.reasons`).
- **Vulnerability.**
  - Credit offers to vulnerable customers go to human review, never straight to the customer (see Human oversight).
    Before review, the model selects them slightly less often than other customers: EV Finance 11.9% against 15.8% on
    test, Green Loans 6.2% against 7.4%.
  - The eligibility rules exclude them from paperless banking, because paper may be an adjustment they need, and from
    green investments.

## Explainability

- **Reason codes from the model itself.** LightGBM's per-feature contributions (TreeSHAP) are summed per reason code,
  such as HIGH_FUEL_SPEND or HIGH_ANNUAL_MILEAGE. Codes that push the score up are reported with an evidence sentence
  from the customer's own data, for example: "you spent £1,446 on road fuel in the last 12 months".
  - A code is shown only when the data supports it (`SUPPORTED` predicates). For example, HIGH_GAS_USE is never shown
    with 0 kWh of gas.
- **Every record says what it rests on:**
  - propensity, confidence, reason codes and evidence;
  - the evidence window and data timestamp;
  - the CO2e and money impact, with how it was calculated;
  - a PCAF data-quality score shown as High / Good / Partial / Low.
- **Customer wording.** Template explanations average 79 words, at a reading ease of 75 ("fairly easy"). Each ends with
  a fixed sentence saying these are estimates from the customer's account data up to a date, and that prices and their
  situation can change.
- **Staff view.** The staff dashboard shows every decision with its reason (`decisions.parquet`). That includes "below
  the propensity threshold", "blocked by rule …", "memory: paused …" and "human review".

## Hallucination

The LLM can only rephrase. Each explanation is built in three steps:

1. A template writes the explanation from the facts, and every template is checked by the guards. All 5,437 pass.
2. The LLM (Gemini API at inference) rewrites the body only. The compliance sentence is fixed and appended afterwards.
3. Guards check the rewrite against the facts. Any failure keeps the template, so no customer ever sees an unchecked
   sentence. The guards check that:
   - every number in the facts is kept, and no new number appears;
   - the currency symbols are kept;
   - a yearly figure is not rewritten as a monthly one;
   - there are no banned claims, such as guarantees ("will", "guarantee"), "pre-approved", "you qualify", "risk-free",
     the bank describing the customer's things as its own ("our investments"), or hype ("amazing", "perfect").

Measured results:
- Gemini: 12 of 12 rewrites pass (a sample, limited by the free-tier quota).
- Local Qwen2.5-1.5B on the 360-item evaluation set: 89.7% pass. The failures were mostly dropped numbers (36), and
  every one of them fell back to the template.

## Human oversight

- **Human review.** Credit offers (EV, home and green loans, solar finance) to a customer with a vulnerability flag go
  to staff review instead of the customer. There are 291 such suggestions. The customer app shows only suggestions with
  the status `recommended`.
- **Human control and audit.** Thresholds, cool-downs, fatigue limits and eligibility rules are all configuration
  (`configs/pipeline.yaml`, `configs/products.yaml`), owned by the bank. Every customer x product pair gets one
  audited decision with its reason.
- **The customer stays in control.** "Not interested" pauses a product for 180 days; "dismiss" pauses it for 90 days;
  "I already have this" stops it. Three contacts in a row without interest pause it for 180 days. Every response can
  be undone.
- **Measurement before claims.** The business case reports the uplift measured against a 10% hold-out control group
  (+0.7 points, 95% interval -5.7 to +7.1), not an assumed one. Value is only claimed once uplift is measured.

## Data limitations

The main ones are below. The complete list, stage by stage, with every assumption's value and source, is in
`docs/ASSUMPTIONS_AND_LIMITATIONS.md`.

- **Synthetic bank.** Customers, behaviour and take-up come from a simulator calibrated on public data, so the metrics
  describe a realistic simulated bank, not Lloyds customers. Take-up in the simulator does not depend on being shown a
  suggestion, which is why the measured uplift is near zero.
- **Small positive counts.** Some products have few take-ups in held-out data (Green Home Loan: 49 validation, 31 test),
  so their metrics have wide uncertainty. Validation and test still agree on the ranking.
- **Impact inputs are estimates from spending:**
  - fuel litres are recovered within 20% for 82.7% of customers;
  - gas and electricity kWh only for 38.8% (median error 28-30%), because a bill shows money, not kWh.
  - The PCAF data-quality score says this to the customer and to the bank.
- **Categoriser boundary.** Electricity/Gas vs Utilities is the weakest boundary when a bill names no fuel (12-17%
  cross-boundary error). Merchants never seen in training are categorised at 90%, against 99% for known merchants.
- **Coverage gaps.**
  - England and Wales only (Census 2021).
  - Area context is at local-authority level.
  - Energy prices are GB averages.
  - MOT data sees only cars aged 3+.
  - Spending through other banks is invisible.
  - The full list is in `docs/EXPLAINER.md` section 16 and `docs/DATA_CARD.md`.
