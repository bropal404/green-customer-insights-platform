# Pipeline robustness

The pipeline survives missing and broken data. All 11 judge scenarios pass in 7 seconds on the shipped models. 5% nulls
in the transaction feed cause no crash and no fuel or energy guess.

## Two concerns, both closed

| Concern | Risk before | Fix | Proof |
|---|---|---|---|
| Judge scenario tests | No single runnable check for the brief's scenarios | `scripts/run_scenario_tests.py` changes one thing about real customers, reruns the real pipeline and checks the decision | 11 of 11 pass, 7 s |
| Null or unmapped transaction fields | A null merchant code or payment type crashed the FR2 categoriser; a null description was read as the word "NAN" | Null-safe categoriser and feature code | 4,000 rows with 5% nulls: 0 crashes, 0 fuel or energy guesses |

## How each layer handles missing data

1. **Ingestion (FR1).**
   - Control totals and SHA-256 checks stop a broken load.
   - 41,301 duplicates are removed.
   - 2,341 unmatched records are quarantined with a reason, never guessed.
2. **Categoriser (FR2).**
   - A null description, merchant code or payment type becomes empty, never the text "NAN".
   - With nothing to read, a payment is Other at 0.3 confidence, never Fuel, Electricity or Gas.
   - A merchant code still counts as evidence: code 5541 is still Fuel.
3. **Features.**
   - A row with no amount is dropped.
   - A row with no category counts as Other.
   - Each fuel is scaled to a year from its own months of bills.
4. **Carbon engine (FR4).**
   - Missing inputs give the status `insufficient_data` and no number.
   - Weaker inputs give a worse PCAF data-quality score, shown to the customer as lower confidence.
5. **Suggestions (FR5-FR7).**
   - No impact estimate means "impact cannot be estimated", and nothing is shown.
   - Confidence below 0.45 is withheld.

## Judge scenario results

| Scenario | Brief family | What the platform does | Result |
|---|---|---|---|
| Unchanged customers | Reproducibility | Every shipped decision reproduced exactly | 228/228 |
| Vehicle record lost, fuel still visible | Missing data | EV impact still estimated (assumes petrol), lower data quality | 25/25 |
| No vehicle data at all | Missing data | EV Finance blocked (NO_VEHICLE) | 25/25 |
| Already owns an EV under 8 years old | Should not be recommended | EV Finance blocked (EV_OWNER_NO_REPLACEMENT); a 10-year-old EV is not blocked | 25/25 |
| No energy data | Insufficient impact data | Fallback estimate with worse data quality; with nothing known, no number and no suggestion | 25/25 |
| Customer becomes vulnerable | Should not be recommended | Credit offers go to human review and none reach the app; paperless and investments blocked | 25/25 |
| Customer in financial difficulty | Should not be recommended | Every credit product blocked | 25/25 |
| Marketing consent withdrawn | Should not be recommended | No suggestion of any product | 25/25 |
| Product now held, or "I already have this" | Changed circumstances | The offer stops | 24/24 |
| 5% null descriptions, merchant codes, payment types | Missing data | No crash; 98 rows with nothing to read become Other; 3,453 untouched rows identical | 4,000/4,000 |
| 5% null categories, 1% null amounts | Missing data | Features build with no missing values and never overstate months; median change: litres 5.7%, electricity 0.9%, gas 0% | 40/40 |

## A bug the tests found and fixed

- **The bug.** Gas and electricity were scaled to a year using one shared month count, so a gap in one fuel's bills was
  not scaled up.
- **Scale.** On our data it affected about 10% of gas and 4% of electricity estimates.
- **The fix.** Each fuel now uses its own months. Under 5% nulls, the largest electricity change fell from 30% to a
  median of 0.9%.
- **Status.** The shipped 30,000-customer results predate the fix; `./run.sh --quick` and `--rebuild` use it.

## No regressions

- On 102,605 real transaction patterns, every category is identical before and after the fix.
- All 228 shipped decisions are reproduced exactly.
- All 16 unit tests pass.

## Run it

```bash
cd gcip
.venv/bin/python scripts/run_scenario_tests.py
```

Code: `scripts/run_scenario_tests.py`, `src/gcip/categorise/model.py`, `src/gcip/categorise/normalise.py`,
`src/gcip/features/store.py`. More detail: `docs/EVALUATION.md` ("Scenario tests") and
`docs/ASSUMPTIONS_AND_LIMITATIONS.md`.
