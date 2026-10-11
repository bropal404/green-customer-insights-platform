"""Orchestrates the synthetic bank build with checkpoints and progress bars.

Output layout (``out_dir``):
    sources/         raw extracts from seven source systems plus _control.json (what FR1 ingestion reads)
    truth_vault/     simulator-only truth: state, labels, transaction_truth/part-*.parquet, merchant
                     directory, and system_of_record.json (fingerprints the ingestion audit checks against)
    meta/            generation manifest and label model report
    _checkpoints/    one marker per finished stage; re-running skips finished stages and chunks

FR1 ingestion (gcip.fr1.ingest) then builds warehouse/ from sources/; validation
and the data card run after it.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from gcip.config import LEXICON_DIR, deep_update, load_config
from gcip.io.files import write_json
from gcip.io.fingerprint import TABLE_KEYS, table_fingerprint
from gcip.simulate import sources as src
from gcip.simulate import warehouse as wh
from gcip.simulate.banking import build_banking
from gcip.simulate.events import build_events
from gcip.simulate.labels import PRODUCTS, assign_prior_holdings, draw_labels, state_at_cutoff
from gcip.simulate.lexicon import SURNAMES, build_merchants, stable_int
from gcip.simulate.population import build_population, town_name
from gcip.simulate.reference import Reference
from gcip.simulate.render import render
from gcip.simulate.transactions import ChunkGenerator
from gcip.simulate.world import build_homes, build_vehicles

log = logging.getLogger("gcip.simulate")
_G: dict = {}  # shared with forked workers
STREAMS = ["pop", "veh", "homes", "bank", "events", "suppliers", "prior", "labels", "takeups", "ids", "merchants",
           "warehouse", "sources"]
MONEY_COLUMNS = {"credit_limit", "original_amount", "collateral_value_at_origination", "balance", "valuation_at_origination"}


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    """One independent random stream per build step (adding a stream never changes the others)."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(s) for name, s in zip(STREAMS, children)}


def _done(out: Path, stage: str) -> bool:
    return (out / "_checkpoints" / f"{stage}.done").exists()


def _mark(out: Path, stage: str, info: dict) -> None:
    write_json(out / "_checkpoints" / f"{stage}.done", {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info})


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def _assign_suppliers(pop: pd.DataFrame, bank: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    spec = yaml.safe_load((LEXICON_DIR / "merchants.yaml").read_text())["sub_categories"]["energy_supplier"]["merchants"]
    keys = np.array([m["key"] for m in spec])
    weights = np.array([m["weight"] for m in spec], dtype=float)
    green = np.array([m.get("green", 0.0) for m in spec], dtype=float)
    affinity = bank["green_supplier_affinity"].values[:, None]
    w = weights[None, :] * np.power(affinity, green[None, :])
    cumulative = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
    idx = np.minimum((rng.random(len(pop))[:, None] > cumulative).sum(axis=1), len(keys) - 1)
    return pd.DataFrame({"customer_id": pop["customer_id"].values, "supplier_key": keys[idx],
                         "green_supplier": green[idx] >= 0.4})


def _identity_fields(state: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    spec = yaml.safe_load((LEXICON_DIR / "merchants.yaml").read_text())["sub_categories"]["income"]
    employers, pensions = spec["employers"], spec["pensions"]
    n = len(state)
    out = pd.DataFrame({"customer_id": state["customer_id"].values})
    out["customer_num"] = np.arange(n) + 100000
    out["account_suffix"] = [f"{stable_int(c, 'acct', modulo=10000):04d}" for c in state["customer_id"]]
    out["debit_card_last4"] = [f"{stable_int(c, 'debit', modulo=10000):04d}" for c in state["customer_id"]]
    out["credit_card_last4"] = [f"{stable_int(c, 'credit', modulo=10000):04d}" for c in state["customer_id"]]
    picks = rng.integers(0, len(employers), n)
    surnames = np.array(SURNAMES)[rng.integers(0, len(SURNAMES), n)]
    towns = state["town"].values
    out["employer_name"] = [employers[p].replace("{la}", t).replace("{loc}", t).replace("{surname}", s)
                            for p, t, s in zip(picks, towns, surnames)]
    picks = rng.integers(0, len(pensions), n)
    out["pension_provider"] = [pensions[p].replace("{la}", t).replace("{loc}", t) for p, t in zip(picks, towns)]
    return out


def _takeup_fields(state: pd.DataFrame, labels: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(state)
    wide = labels.pivot(index="customer_id", columns="product", values="takeup_date").loc[state["customer_id"]]
    out = pd.DataFrame({"customer_id": state["customer_id"].values})
    for product in PRODUCTS:
        out[f"{product.lower()}_date"] = wide[product].values.astype("datetime64[D]")
    gas_home = state["main_heating"].values == "gas"
    out["gp01_monthly"] = rng.uniform(380, 720, n).round(2)
    out["gp02_measure"] = np.where(gas_home & (rng.random(n) < 0.35), "heat_pump", "insulation")
    out["gp02_install_date"] = out["gp02_date"].values + np.timedelta64(45, "D")
    out["gp02_monthly"] = rng.uniform(90, 260, n).round(2)
    out["gp03_install_date"] = out["gp03_date"].values + np.timedelta64(40, "D")
    out["gp03_kwp"] = np.where(~np.isnat(out["gp03_date"].values), (4.0 * np.exp(rng.normal(0, 0.18, n))).round(2), 0.0)
    out["gp03_monthly"] = rng.uniform(70, 150, n).round(2)
    out["gp06_monthly"] = rng.uniform(60, 190, n).round(2)
    has_ev_start = state["has_car"].values & np.isin(state["car_fuel"].values, ["bev", "phev"])
    new_ev = np.where(state["new_car_fuel"].values == "bev", state["new_car_date"].values.astype("datetime64[D]"), np.datetime64("NaT"))
    gp01 = out["gp01_date"].values
    ev_from = np.where(np.isnat(new_ev), gp01, np.where(np.isnat(gp01), new_ev, np.minimum(new_ev, gp01)))
    out["ev_from_date"] = np.where(has_ev_start, np.datetime64("NaT"), ev_from).astype("datetime64[D]")
    out["has_ev_at_start"] = has_ev_start
    km = np.where(state["annual_km_true"].values > 0, state["annual_km_true"].values, 11000.0)
    out["ev_home_kwh_if_ev"] = (km * 0.205 * (1 - state["public_charging_share"].values)).round(0)
    return out


def system_of_record(state: pd.DataFrame, labels: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """The bank's tables as its systems hold them, before extraction: money in pence, rates in basis points."""
    months = np.arange(np.datetime64(cfg["transactions"]["start"], "M"), np.datetime64(cfg["transactions"]["end"], "M") + 1)
    prior = state[[c for c in state.columns if c.startswith("prior_")] + ["customer_id"]]
    takeups = state[["customer_id"] + [f"{p.lower()}_date" for p in PRODUCTS]]
    customers = wh.build_customers(state, rng)
    accounts = wh.build_accounts(state, prior, takeups, rng)
    tables = {
        "customers": customers,
        "accounts": accounts,
        "balances": wh.build_balances(state, accounts, months, rng),
        "product_holdings": wh.build_holdings(accounts, state, takeups),
        "preference_changes": wh.build_preference_changes(state, takeups),
        "assets": wh.build_assets(state),
        "digital_activity": wh.build_digital(state, takeups, months, rng),
        "campaign_responses": wh.build_campaigns(state, labels, prior, rng),
    }
    for frame in tables.values():
        for column in frame.columns:
            if column in MONEY_COLUMNS:
                frame[column] = frame[column].astype(float).round(2)
        if "interest_rate" in frame:
            frame["interest_rate"] = frame["interest_rate"].astype(float).round(4)
        if "term_months" in frame:
            frame["term_months"] = frame["term_months"].astype(float).round(0)
    return tables


def _chunk_worker(args: tuple[int, int, int]) -> dict:
    chunk_id, start, end = args
    out: Path = _G["out"]
    sources = out / "sources"
    targets = [sources / "card_processor" / f"part-{chunk_id:04d}.parquet",
               sources / "payments_hub" / f"part-{chunk_id:04d}.csv.gz",
               out / "truth_vault" / "transaction_truth" / f"part-{chunk_id:04d}.parquet"]
    if all(p.exists() for p in targets):
        return {"chunk": chunk_id, "rows": -1, "skipped": True}
    seed = np.random.SeedSequence([_G["seed"], 7, chunk_id])
    rng = np.random.default_rng(seed)
    state = _G["state"].iloc[start:end].reset_index(drop=True)
    frame = ChunkGenerator(state, _G["merchants"], _G["ref"], _G["cfg"], rng).run()
    observed, truth = render(frame, state, _G["merchants"], chunk_id, rng,
                             mcc_present=_G["cfg"]["observability"]["mcc_present_card"])
    extract_rng = np.random.default_rng(np.random.SeedSequence([_G["seed"], 8, chunk_id]))
    info = src.write_transaction_extracts(observed, chunk_id, sources, extract_rng)
    _write(truth, targets[2])
    return {"chunk": chunk_id, "rows": int(len(observed)), "skipped": False, **info}


def generate(out_dir: str | Path, n_customers: int | None = None, workers: int | None = None,
             chunk_size: int = 1000, overrides: dict | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("simulation")
    if overrides:
        cfg = deep_update(cfg, overrides)
    if n_customers:
        cfg["population"]["n_customers"] = int(n_customers)
    n = cfg["population"]["n_customers"]
    seed = int(cfg["seed"])
    ref = Reference.load()
    streams = make_streams(seed)

    state_path = out / "truth_vault" / "state.parquet"
    if not _done(out, "world"):
        log.info("building world for %s customers", n)
        steps = tqdm(total=8, desc="world", unit="step")
        pop = build_population(n, cfg, ref, streams["pop"]); steps.update()
        veh = build_vehicles(pop, cfg, ref, streams["veh"]); steps.update()
        homes = build_homes(pop, veh, cfg, ref, streams["homes"]); steps.update()
        bank = build_banking(pop, veh, homes, cfg, streams["bank"], ref); steps.update()
        events = build_events(pop, veh, cfg, streams["events"]); steps.update()
        suppliers = _assign_suppliers(pop, bank, streams["suppliers"]); steps.update()
        state = state_at_cutoff(pop, veh, homes, bank, events, suppliers, cfg)
        prior = assign_prior_holdings(state, cfg, streams["prior"])
        labels, report = draw_labels(state, prior, ref.la, cfg, streams["labels"]); steps.update()
        state = state.merge(prior, on="customer_id").merge(_takeup_fields(state, labels, streams["takeups"]), on="customer_id")
        state = state.merge(_identity_fields(state, streams["ids"]), on="customer_id").copy()
        state["split"] = wh.split_of(state["customer_id"], cfg["splits"]["hash_salt"], cfg["splits"]["train"],
                                     cfg["splits"]["validation"])
        steps.update(); steps.close()
        _write(state, state_path)
        _write(labels, out / "truth_vault" / "labels.parquet")
        (out / "meta").mkdir(parents=True, exist_ok=True)
        report.to_json(out / "meta" / "label_model_report.json", orient="records", indent=2)
        _mark(out, "world", {"customers": n})
    state = pd.read_parquet(state_path)
    labels = pd.read_parquet(out / "truth_vault" / "labels.parquet")

    towns = {code: town_name(name) for code, name in zip(ref.la["la_code"], ref.la["la_name"])}
    merchants = build_merchants(ref.la, towns, streams["merchants"])
    merchants.table.assign(variants=merchants.table["variants"].map(json.dumps),
                           regions=merchants.table["regions"].map(json.dumps)).to_parquet(
        out / "truth_vault" / "merchant_directory.parquet", index=False)

    if not _done(out, "transactions"):
        chunks = [(i, s, min(s + chunk_size, n)) for i, s in enumerate(range(0, n, chunk_size))]
        _G.update(state=state, merchants=merchants, ref=ref, cfg=cfg, out=out, seed=seed)
        workers = workers or max(1, min(len(chunks), (os.cpu_count() or 2) - 1))
        log.info("generating transactions: %s chunks on %s workers", len(chunks), workers)
        totals = {"rows": 0, "card_rows": 0, "card_replays": 0, "payment_rows": 0, "legacy_account_numbers": 0}

        def tally(result: dict) -> None:
            for key in totals:
                totals[key] += max(result.get(key, 0), 0)

        if workers == 1:
            for result in tqdm(map(_chunk_worker, chunks), total=len(chunks), desc="transactions", unit="chunk"):
                tally(result)
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers) as pool:
                for result in tqdm(pool.imap_unordered(_chunk_worker, chunks), total=len(chunks), desc="transactions", unit="chunk"):
                    tally(result)
        _mark(out, "transactions", {"chunks": len(chunks), "written_this_run": totals})

    extract_date = str(cfg["transactions"]["end"])
    if not _done(out, "sources"):
        sources = out / "sources"
        steps = tqdm(total=8, desc="source extracts", unit="step")
        tables = system_of_record(state, labels, cfg, streams["warehouse"]); steps.update()
        write_json(out / "truth_vault" / "system_of_record.json",
                   {name: table_fingerprint(frame, TABLE_KEYS[name]) for name, frame in tables.items()}); steps.update()
        rng = streams["sources"]
        src.write_core(tables["customers"], tables["accounts"], tables["balances"], sources, rng); steps.update()
        src.write_card_registry(tables["accounts"], state, sources); steps.update()
        src.write_product_system(tables["product_holdings"], tables["preference_changes"], sources); steps.update()
        src.write_lending_assets(tables["assets"], tables["accounts"], sources, extract_date); steps.update()
        src.write_crm(tables["campaign_responses"], tables["customers"], sources, rng); steps.update()
        src.write_digital(tables["digital_activity"], sources); steps.update()
        steps.close()
        _mark(out, "sources", {name: len(frame) for name, frame in tables.items()})

    if not _done(out, "controls"):
        control = src.write_controls(out / "sources", extract_date)
        _mark(out, "controls", {"files": len(control["files"])})

    manifest = {
        "customers": int(n),
        "seed": seed,
        "extract_date": extract_date,
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": cfg,
        "files": {str(p.relative_to(out)): p.stat().st_size
                  for p in sorted(out.rglob("*")) if p.is_file() and p.parts[len(out.parts)] in ("sources", "truth_vault")},
    }
    write_json(out / "meta" / "generation_manifest.json", manifest)
    return manifest
