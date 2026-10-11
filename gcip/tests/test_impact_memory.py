"""FR4 golden cases worked by hand, and customer memory cool-downs in the recommendation engine."""

import numpy as np
import pandas as pd
import pytest

from gcip.config import load_config
from gcip.impact.engine import Factors, Prices, assess
from gcip.memory.replay import apply_events
from gcip.recommend.engine import recommend

AS_OF = "2026-09-30"
PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]


def driver(**overrides) -> pd.DataFrame:
    row = {"customer_id": "C1", "fuel_months_12m": 12, "fuel_litres_12m": 1000.0, "asset_vehicle_fuel_type": "petrol",
           "vehicle_evidence": 1, "fuel_visible": 1, "house_known": 1, "homeowner_evidence": 1}
    return pd.DataFrame([row | overrides])


def test_factors_and_prices_are_the_published_values():
    f, p = Factors.load(load_config("carbon")), Prices.at(AS_OF)
    assert f.petrol_l == pytest.approx(2.65594, abs=1e-5)     # DESNZ 2026, combustion + well-to-tank, per litre
    assert f.grid_kwh == pytest.approx(0.18436, abs=1e-5)     # generation + T&D + their well-to-tank, per kWh
    assert p.petrol == pytest.approx(1.7346, abs=1e-4)        # DESNZ weekly pump price in force on 30 Sep 2026
    assert p.elec_unit == pytest.approx(0.2611, abs=1e-4)     # Ofgem cap unit rate, Q4 2026


def test_ev_golden_case():
    """1,000 litres of petrol a year, house with a drive.
    Baseline 1,000 x 2.65594 = 2,656 kg. Same distance 1,000 x 12.85 = 12,850 km x 0.205 = 2,634.25 kWh x 0.18436 = 486 kg.
    Avoided 2,170 kg. Money: charging 2,634.25 x (0.85 x 0.2611 + 0.15 x 0.70) = 861.23 vs fuel 1,000 x 1.7346 = 1,734.60,
    so 873 a year less."""
    out = assess(driver(), ["GP01"], AS_OF).iloc[0]
    assert out["status"] == "ok"
    assert out["baseline_co2e_kg_per_year"] == pytest.approx(2656, abs=1)
    assert out["co2e_avoided_kg_per_year"] == pytest.approx(2170, abs=1)
    assert out["running_cost_change_gbp_per_year"] == pytest.approx(-873, abs=1)
    assert out["data_quality"] == 2                            # vehicle on record + fuel visible


def test_no_number_without_the_inputs():
    out = assess(driver(vehicle_evidence=0, fuel_visible=0, fuel_months_12m=0, fuel_litres_12m=0.0,
                        asset_vehicle_fuel_type=None), ["GP01"], AS_OF).iloc[0]
    assert out["status"] == "insufficient_data"


def _inputs(responses: dict[str, tuple[str, str]]):
    scores = pd.DataFrame({"customer_id": "C1", "product": ["GP01", "GP03"], "as_of": pd.Timestamp(AS_OF), "eligible": True,
                           "blocks": "", "human_review": False, "propensity": [0.30, 0.20], "confidence": [0.8, 0.8],
                           "reason_codes": [[], []], "evidence": [[], []]})
    impact = pd.DataFrame({"customer_id": "C1", "product": ["GP01", "GP03"], "status": "ok", "method": "", "measure": "",
                           "horizon_years": 4, "baseline_co2e_kg_per_year": 2000.0, "co2e_avoided_kg_per_year": [1500.0, 600.0],
                           "co2e_avoided_kg_horizon": 6000.0, "data_quality": 3, "financed_co2e_avoided_kg_per_year": np.nan,
                           "running_cost_change_gbp_per_year": -500.0, "upfront_cost_gbp": np.nan, "payback_years": np.nan,
                           "note": ""})
    row = {"customer_id": "C1", "as_of": pd.Timestamp(AS_OF), "marketing_consent": True, "vulnerability_flag": False,
           "preferred_contact_channel": "app"}
    for p in PRODUCTS:
        response, date = responses.get(p, ("none", None))
        row[f"last_response_{p}"], row[f"last_response_date_{p}"] = response, pd.Timestamp(date) if date else pd.NaT
    return scores, impact, pd.DataFrame([row])


def test_memory_cooldowns():
    cfg = load_config("pipeline")["recommend"]
    shown, _ = recommend(*_inputs({}), cfg)
    assert list(shown["product"]) == ["GP01", "GP03"]

    # "Not interested" 30 days ago: EV Finance pauses for 180 days and solar moves up.
    shown, decisions = recommend(*_inputs({"GP01": ("not_interested", "2026-08-31")}), cfg)
    assert list(shown["product"]) == ["GP03"]
    assert decisions.set_index("product").loc["GP01", "decision"].startswith("memory: not interested 180-day cool-down")

    # After the cool-down it can be shown again.
    shown, _ = recommend(*_inputs({"GP01": ("not_interested", "2026-03-01")}), cfg)
    assert "GP01" in set(shown["product"])


def test_memory_events_override_older_responses():
    scores, impact, c360 = _inputs({"GP01": ("clicked", "2026-01-01")})
    events = pd.DataFrame({"customer_id": ["C1"], "product": ["GP01"], "response": ["dismissed"],
                           "response_date": [pd.Timestamp("2026-09-30")], "source": ["dashboard"]})
    updated = apply_events(c360, events)
    assert updated.loc[0, "last_response_GP01"] == "dismissed"
    shown, _ = recommend(scores, impact, updated, load_config("pipeline")["recommend"])
    assert "GP01" not in set(shown["product"])


def test_reason_needs_supporting_data():
    """A reason whose sentence the data contradicts (high gas use with no gas bills) is skipped for the next one."""
    from gcip.models.reasons import top_reasons

    rows = pd.DataFrame([{"gas_kwh_12m": 0.0, "area_gas_dom_mean_kwh": 9580, "bike_shop_spend_12m": 1680.0,
                          "appliance_spend_12m": 0.0, "existing_borrowing": 1}])
    reasons = pd.DataFrame([[0.9, 0.5, 0.3]], columns=["HIGH_GAS_USE", "GREEN_PURCHASE_INTENT", "EXISTING_BORROWING"])
    codes, evidence = top_reasons(reasons, rows)
    assert codes == [["GREEN_PURCHASE_INTENT", "EXISTING_BORROWING"]]
    assert evidence[0][0] == "you spent £1,680 at bike or appliance shops in the last 12 months"


def test_memory_stops_after_repeated_contacts_without_interest():
    """Three contacts in a row with no engagement pause a product however strong its propensity; interest lifts it."""
    from gcip.memory.signals import signals

    cfg = load_config("pipeline")
    scores, impact, c360 = _inputs({})
    contacts = pd.DataFrame({"customer_id": "C1", "product": "GP01", "response": "no_response", "source": "CRM campaign",
                             "response_date": pd.to_datetime(["2026-06-15", "2026-07-15", "2026-08-17"])})
    sig = signals(contacts, pd.Timestamp(AS_OF), cfg["memory"], cfg["recommend"]["cooldown_days"])
    assert sig.loc[("C1", "GP01"), "streak"] == 3
    assert sig.loc[("C1", "GP01"), "weight"] < 0.5
    shown, decisions = recommend(scores, impact, c360, cfg["recommend"], memory=sig)
    assert list(shown["product"]) == ["GP03"]
    assert "no interest after 3 contacts" in decisions.set_index("product").loc["GP01", "decision"]

    clicked = pd.concat([contacts, contacts.iloc[:1].assign(response="clicked", response_date=pd.Timestamp("2026-09-01"))])
    sig = signals(clicked, pd.Timestamp(AS_OF), cfg["memory"], cfg["recommend"]["cooldown_days"])
    assert sig.loc[("C1", "GP01"), "streak"] == 0 and sig.loc[("C1", "GP01"), "state"] == ""


def test_business_value_worked_by_hand():
    """EV Finance at the default assumptions: GBP 24,000 loan, 7.9% APR vs 4.0% funding over 4 years."""
    from gcip.insights.value import suggestion_value

    rows = pd.DataFrame([{"customer_id": "C1", "product": "GP01", "propensity": 0.10, "status": "recommended", "impact_status": "ok",
                          "running_cost_change_gbp_per_year": -800.0, "upfront_cost_gbp": None, "horizon_years": 4,
                          "co2e_avoided_kg_per_year": 2000.0, "co2e_avoided_kg_horizon": 8000.0, "payback_years": None,
                          "financed_co2e_avoided_kg_per_year": 1600.0}])
    c360 = pd.DataFrame([{"customer_id": "C1", "paper_accounts": 0, "balance_investment": 0.0, "preferred_contact_channel": "app"}])
    v = suggestion_value(rows, c360).iloc[0]
    assert v["bank_new_lending_gbp"] == 24000                     # 30,000 x 0.80
    assert v["bank_nii_gbp"] == 1872                              # 24,000 x (0.079 - 0.040) x 4 / 2
    assert v["bank_expected_credit_loss_gbp"] == 288              # 24,000 x 0.03 x 0.40
    assert v["bank_value_if_taken_gbp"] == 1872 - 288 - 150       # less origination
    assert v["bank_expected_value_gbp"] == pytest.approx(0.10 * 1434 - 0.05)
    assert v["customer_lifetime_net_gbp"] == 3200                 # 800 a year x 4, no upfront in this product
    assert v["customer_npv_gbp"] == pytest.approx(800 * (1 - 1.035 ** -4) / 0.035, abs=1)
    assert v["abatement_cost_gbp_per_t"] < 0                      # saves money per tonne cut
