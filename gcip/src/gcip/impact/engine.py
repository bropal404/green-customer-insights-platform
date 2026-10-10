"""FR4 impact engine: one carbon statement and one money statement per customer and product.

Deterministic: the same inputs and configs/carbon.yaml give the same output. Carbon uses DESNZ 2026 factors on a
well-to-wheel boundary (combustion or generation, plus upstream production, plus grid losses).
  Customer lens: avoided emissions per year and over the product's horizon (GHG Protocol / WBCSD avoided-emissions
  framing), from the customer's own litres and kWh where visible, else a documented fallback with a lower
  data-quality score (PCAF-style 1-5, 1 never claimed).
  Bank lens: PCAF financed emissions avoided, where PCAF Part A covers the asset class (motor loans, mortgages,
  listed equity proxies for GP05); attribution = outstanding / value at origination.
Money: running-cost change per year at prices in force on the as-of date, upfront cost, payback. None for
Paperless (no cost) and Green Investments (returns are uncertain; information, not advice).
When the inputs a method needs are missing, the status is 'insufficient_data' and no number is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gcip.config import REFERENCE_DIR, load_config

COLUMNS = ["customer_id", "product", "status", "method", "measure", "horizon_years", "baseline_co2e_kg_per_year",
           "co2e_avoided_kg_per_year", "co2e_avoided_kg_horizon", "data_quality", "financed_co2e_avoided_kg_per_year",
           "running_cost_change_gbp_per_year", "upfront_cost_gbp", "payback_years", "note"]


@dataclass
class Factors:
    petrol_l: float
    diesel_l: float
    gas_kwh: float
    oil_l: float
    lpg_l: float
    grid_kwh: float
    paper_t: float

    @classmethod
    def load(cls, cfg: dict) -> "Factors":
        table = pd.read_csv(REFERENCE_DIR / "ghg_factors_2026.csv", dtype={"factor_id": str}).set_index("factor_id")["kg_co2e"]
        f = {k: float(table[v]) for k, v in cfg["factors"].items() if v in table}
        return cls(petrol_l=f["petrol_litre"] + f["petrol_litre_wtt"], diesel_l=f["diesel_litre"] + f["diesel_litre_wtt"],
                   gas_kwh=f["natural_gas_kwh"] + f["natural_gas_kwh_wtt"], oil_l=f["burning_oil_litre"] + f["burning_oil_litre_wtt"],
                   lpg_l=f["lpg_litre"] + f["lpg_litre_wtt"],
                   grid_kwh=f["grid_kwh"] + f["grid_td_kwh"] + f["grid_wtt_generation_kwh"] + f["grid_wtt_td_kwh"],
                   paper_t=f["paper_primary_tonne"] + f["paper_closed_loop_disposal_tonne"])


@dataclass
class Prices:
    petrol: float     # GBP per litre
    diesel: float
    elec_unit: float  # GBP per kWh
    gas_unit: float
    gas_standing_year: float

    @classmethod
    def at(cls, as_of: str) -> "Prices":
        fuel = pd.read_csv(REFERENCE_DIR / "fuel_prices_weekly.csv", parse_dates=["week_start"]).sort_values("week_start")
        week = fuel[fuel["week_start"] <= pd.Timestamp(as_of)].iloc[-1]
        cap = pd.read_csv(REFERENCE_DIR / "ofgem_price_cap.csv", parse_dates=["start", "end"]).sort_values("start")
        quarter = cap[cap["start"] <= pd.Timestamp(as_of)].iloc[-1]
        return cls(petrol=week["petrol_ppl"] / 100, diesel=week["diesel_ppl"] / 100, elec_unit=quarter["elec_unit_p_per_kwh"] / 100,
                   gas_unit=quarter["gas_unit_p_per_kwh"] / 100, gas_standing_year=quarter["gas_standing_p_per_day"] * 365 / 100)


def _v(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default)
    return pd.to_numeric(frame[column], errors="coerce").astype(float).to_numpy()


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.nan_to_num(_v(frame, column, 0.0)) > 0


def _payback(upfront: np.ndarray, change: np.ndarray) -> np.ndarray:
    saving = -change
    return np.where(saving > 1.0, np.round(upfront / np.maximum(saving, 1e-9), 1), np.nan)


def _frame(table: pd.DataFrame, product: str, horizon: float, **cols) -> pd.DataFrame:
    out = pd.DataFrame({"customer_id": table["customer_id"].to_numpy(), "product": product, "horizon_years": horizon})
    for k, v in cols.items():
        out[k] = v
    out["co2e_avoided_kg_horizon"] = (out["co2e_avoided_kg_per_year"] * horizon).round(0)
    for c in COLUMNS:
        if c not in out:
            out[c] = np.nan
    insufficient = out["status"] == "insufficient_data"
    numeric = ["baseline_co2e_kg_per_year", "co2e_avoided_kg_per_year", "co2e_avoided_kg_horizon", "financed_co2e_avoided_kg_per_year",
               "running_cost_change_gbp_per_year", "upfront_cost_gbp", "payback_years"]
    out.loc[insufficient, numeric] = np.nan
    return out[COLUMNS]


# --------------------------------------------------------------------------- products


def ev_finance(t: pd.DataFrame, cfg: dict, f: Factors, p: Prices) -> pd.DataFrame:
    m = cfg["money"]
    fuel = t.get("asset_vehicle_fuel_type", pd.Series(index=t.index, dtype=object)).fillna("petrol").to_numpy()
    diesel = fuel == "diesel"
    months = _v(t, "fuel_months_12m", 0.0)
    litres_seen = _v(t, "fuel_litres_12m", 0.0)
    kml = np.where(diesel, m["km_per_litre"]["diesel"], m["km_per_litre"]["petrol"])
    seen = months >= 3
    litres = np.where(seen, litres_seen, m["nts_average_km"]["value"] / kml)
    km = litres * kml
    baseline = litres * np.where(diesel, f.diesel_l, f.petrol_l)
    ev_kwh = km * m["ev_kwh_per_km"]["value"]
    after = ev_kwh * f.grid_kwh
    avoided = baseline - after
    home = np.where(_flag(t, "house_known") | _flag(t, "homeowner_evidence"), m["home_charging_share"]["house"], m["home_charging_share"]["other"])
    ev_cost = ev_kwh * (home * p.elec_unit + (1 - home) * cfg["assumptions"]["public_charge_price_gbp_per_kwh"]["value"])
    fuel_cost = litres * np.where(diesel, p.diesel, p.petrol)
    known_vehicle = t.get("asset_vehicle_fuel_type", pd.Series(index=t.index, dtype=object)).notna().to_numpy()
    dq = np.select([known_vehicle & _flag(t, "fuel_visible"), _flag(t, "fuel_visible"), seen], [2, 3, 4], 5)
    status = np.where(_flag(t, "vehicle_evidence"), "ok", "insufficient_data")
    return _frame(t, "GP01", cfg["assumptions"]["horizon_years"]["GP01"], status=status, method="AVOIDED_EV_WTW_v1",
                  measure="battery-electric car for the same distance", baseline_co2e_kg_per_year=baseline.round(0),
                  co2e_avoided_kg_per_year=avoided.round(0), data_quality=dq,
                  financed_co2e_avoided_kg_per_year=(avoided * m["ev_loan_to_value"]["value"]).round(0),
                  running_cost_change_gbp_per_year=(ev_cost - fuel_cost).round(0), upfront_cost_gbp=np.nan, payback_years=np.nan,
                  note=np.where(seen, "fuel bought in the last 12 months; excludes the car price and finance cost",
                                "NTS average mileage (fuel not visible); excludes the car price and finance cost"))


def green_home(t: pd.DataFrame, cfg: dict, f: Factors, p: Prices) -> pd.DataFrame:
    a, m = cfg["assumptions"], cfg["money"]
    heating = t.get("asset_epc_main_heating", pd.Series(index=t.index, dtype=object)).to_numpy().astype(object)
    gas_seen, oil_seen = _v(t, "gas_kwh_12m", 0.0), _v(t, "oil_litres_12m", 0.0)
    heating = np.where(pd.isna(heating), np.where(oil_seen > 0, "oil", "gas"), heating)
    visible = _flag(t, "energy_bills_visible")
    area_gas = _v(t, "area_gas_dom_mean_kwh")
    floor = _v(t, "asset_floor_area_m2")
    epc = t.get("asset_epc_band", pd.Series(index=t.index, dtype=object)).to_numpy()
    epc_factor = pd.Series(epc).map(dict(zip("ABCDEFG", [0.45, 0.60, 0.80, 1.00, 1.25, 1.55, 1.85]))).to_numpy(dtype=float)
    from_epc = area_gas * (floor / 92.0) ** 0.75 * epc_factor
    fuel_kwh = np.where(visible & (gas_seen + oil_seen > 0), gas_seen + oil_seen * 10.35,
                        np.where(np.isfinite(from_epc), from_epc, area_gas))
    dq = np.select([visible, np.isfinite(from_epc), t.get("asset_property_type", pd.Series(index=t.index)).notna().to_numpy()], [2, 3, 4], 5)
    oil = heating == "oil"
    per_kwh = np.where(oil, f.oil_l / 10.35, f.gas_kwh)
    baseline = fuel_kwh * per_kwh
    fossil = np.isin(heating, ["gas", "oil", "lpg"])
    house = ~_flag(t, "flat_known")
    heat_pump = fossil & house
    hp_kwh = fuel_kwh * a["boiler_efficiency"]["value"] / a["heat_pump_efficiency"]["value"]
    avoided = np.where(heat_pump, baseline - hp_kwh * f.grid_kwh, baseline * a["fabric_heating_saving"]["value"])
    fuel_cost = np.where(oil, fuel_kwh / 10.35 * a["heating_oil_gbp_per_litre"]["value"], fuel_kwh * p.gas_unit + p.gas_standing_year)
    change = np.where(heat_pump, hp_kwh * p.elec_unit - fuel_cost,
                      -a["fabric_heating_saving"]["value"] * fuel_kwh * np.where(oil, a["heating_oil_gbp_per_litre"]["value"] / 10.35, p.gas_unit))
    up = m["upfront_gbp"]
    grant = np.where(oil | (heating == "lpg"), up["heat_pump_grant_off_gas"]["value"], up["heat_pump_grant_gas"]["value"])
    upfront = np.where(heat_pump, up["heat_pump"]["value"] - grant, up["insulation"]["value"])
    attribution = np.abs(_v(t, "balance_mortgage", 0.0)) / _v(t, "mortgage_property_value_at_origination")
    status = np.where(np.isfinite(fuel_kwh) & (fuel_kwh > 0), "ok", "insufficient_data")
    return _frame(t, "GP02", a["horizon_years"]["GP02"], status=status, method="AVOIDED_HOME_WTW_v1",
                  measure=np.where(heat_pump, "air-source heat pump replacing a fossil boiler", "insulation and draught-proofing"),
                  baseline_co2e_kg_per_year=baseline.round(0), co2e_avoided_kg_per_year=avoided.round(0), data_quality=dq,
                  financed_co2e_avoided_kg_per_year=(np.clip(attribution, 0, 1) * avoided).round(0),
                  running_cost_change_gbp_per_year=change.round(0), upfront_cost_gbp=upfront, payback_years=_payback(upfront, change),
                  note=np.where(heat_pump, "after the Boiler Upgrade Scheme grant; standing charge saved when gas is removed",
                                "20% less heating energy"))


def solar(t: pd.DataFrame, cfg: dict, f: Factors, p: Prices) -> pd.DataFrame:
    a, m = cfg["assumptions"], cfg["money"]
    generation = a["typical_pv_kwp"]["value"] * _v(t, "area_pv_kwh_per_kwp_year") * a["pv_orientation_factor"]["value"]
    avoided = generation * f.grid_kwh
    elec = _v(t, "elec_kwh_12m", 0.0)
    self_use = np.minimum(generation * m["pv_self_consumption"]["value"], np.where(elec > 0, elec, np.inf))
    saving = self_use * p.elec_unit + (generation - self_use) * m["seg_export_gbp_per_kwh"]["value"]
    upfront = np.full(len(t), float(m["upfront_gbp"]["solar_4kwp"]["value"]))
    dq = np.where(_flag(t, "house_known") & t.get("asset_epc_band", pd.Series(index=t.index)).notna().to_numpy(), 3, 4)
    status = np.where(np.isfinite(generation), "ok", "insufficient_data")
    return _frame(t, "GP03", a["horizon_years"]["GP03"], status=status, method="AVOIDED_SOLAR_WTW_v1", measure="4 kWp rooftop solar",
                  baseline_co2e_kg_per_year=np.nan, co2e_avoided_kg_per_year=avoided.round(0), data_quality=dq,
                  financed_co2e_avoided_kg_per_year=np.nan, running_cost_change_gbp_per_year=(-saving).round(0),
                  upfront_cost_gbp=upfront, payback_years=_payback(upfront, -saving),
                  note="regional PVGIS yield; financed emissions not covered by PCAF for unsecured loans")


def paperless(t: pd.DataFrame, cfg: dict, f: Factors, p: Prices) -> pd.DataFrame:
    a = cfg["assumptions"]
    grams = (a["statement_sheets"]["value"] * a["sheet_grams"]["value"] + a["envelope_grams"]["value"]) * a["statements_per_account_year"]["value"]
    accounts = np.maximum(_v(t, "paper_accounts", 0.0), 0)
    avoided = accounts * grams / 1e6 * f.paper_t
    return _frame(t, "GP04", a["horizon_years"]["GP04"], status=np.where(accounts > 0, "ok", "insufficient_data"),
                  method="AVOIDED_PAPER_v1", measure="online statements for every account", baseline_co2e_kg_per_year=avoided.round(2),
                  co2e_avoided_kg_per_year=avoided.round(2), data_quality=3, financed_co2e_avoided_kg_per_year=np.nan,
                  running_cost_change_gbp_per_year=0.0, upfront_cost_gbp=0.0, payback_years=np.nan,
                  note="no cost and no saving to the customer")


def fund_intensities(cfg: dict) -> tuple[float, float]:
    """kg CO2e per GBP of output for a broad UK equity fund and a sustainability-focused fund (ONS by SIC section)."""
    catalogue = cfg["fund_catalogue"]
    ons = pd.read_csv(REFERENCE_DIR / "ons_ghg_intensity_by_industry.csv")
    ons = ons[ons["year"] == catalogue["intensity_year"]].set_index("sic_section")["intensity"]
    def weighted(weights: dict) -> float:
        return float(sum(w * ons.get(s, np.nan) for s, w in weights.items()) / sum(weights.values()))
    return weighted(catalogue["conventional_uk_equity"]), weighted(catalogue["sustainability_focus"])


def green_investments(t: pd.DataFrame, cfg: dict, f: Factors, p: Prices) -> pd.DataFrame:
    conventional, focus = fund_intensities(cfg)
    invested = np.where(_v(t, "balance_investment", 0.0) > 0, _v(t, "balance_investment", 0.0), _v(t, "investment_transfer_total_12m", 0.0))
    switched = invested * cfg["fund_catalogue"]["switch_share_of_holdings"]
    avoided = switched * (conventional - focus)
    return _frame(t, "GP05", cfg["assumptions"]["horizon_years"]["GP05"], status=np.where(invested > 0, "ok", "insufficient_data"),
                  method="FINANCED_FUND_SWITCH_v1", measure="a quarter of investments moved to a sustainability-focused fund",
                  baseline_co2e_kg_per_year=(switched * conventional).round(0), co2e_avoided_kg_per_year=avoided.round(0), data_quality=5,
                  financed_co2e_avoided_kg_per_year=avoided.round(0), running_cost_change_gbp_per_year=np.nan, upfront_cost_gbp=np.nan,
                  payback_years=np.nan, note="capital at risk; this is information, not advice; sector-average footprints")


def green_loans(t: pd.DataFrame, cfg: dict, f: Factors, p: Prices) -> pd.DataFrame:
    a, m = cfg["assumptions"], cfg["money"]
    km = a["ebike_car_km_replaced"]["value"]
    ebike_litres = km / m["km_per_litre"]["petrol"]
    ebike_kwh = km * a["ebike_kwh_per_km"]["value"]
    bike = _flag(t, "bike_shop_spend_12m") | _flag(t, "pt_micromobility_trips_12m")
    appliance = ~bike & _flag(t, "appliance_spend_12m")
    charger = ~bike & ~appliance & _flag(t, "ev_owner")
    public_kwh = _v(t, "ev_public_kwh_12m", 0.0)
    public_price = a["public_charge_price_gbp_per_kwh"]["value"]
    avoided = np.select([bike, appliance], [ebike_litres * f.petrol_l - ebike_kwh * f.grid_kwh, a["appliance_kwh_saving"]["value"] * f.grid_kwh], 0.0)
    change = np.select([bike, appliance, charger],
                       [ebike_kwh * p.elec_unit - ebike_litres * p.petrol, -a["appliance_kwh_saving"]["value"] * p.elec_unit,
                        -0.8 * public_kwh * (public_price - p.elec_unit)], np.nan)
    up = m["upfront_gbp"]
    upfront = np.select([bike, appliance, charger], [up["ebike"]["value"], up["efficient_appliance"]["value"], up["home_charger"]["value"]], np.nan)
    status = np.where(bike | appliance | charger, "ok", "insufficient_data")
    return _frame(t, "GP06", a["horizon_years"]["GP06"], status=status, method="AVOIDED_GREEN_LOAN_v1",
                  measure=np.select([bike, appliance, charger], ["e-bike replacing short car trips", "efficient appliance",
                                                                 "home EV charger"], "no purchase evidence"),
                  baseline_co2e_kg_per_year=np.nan, co2e_avoided_kg_per_year=avoided.round(0), data_quality=4,
                  financed_co2e_avoided_kg_per_year=np.nan, running_cost_change_gbp_per_year=change.round(0), upfront_cost_gbp=upfront,
                  payback_years=_payback(upfront, change),
                  note=np.where(charger, "cheaper home charging; same electricity, so no carbon change", "assumption-based purchase"))


METHODS = {"GP01": ev_finance, "GP02": green_home, "GP03": solar, "GP04": paperless, "GP05": green_investments, "GP06": green_loans}


def assess(table: pd.DataFrame, products: list[str], as_of: str) -> pd.DataFrame:
    """Impact rows for every customer in ``table`` (the joined Customer 360 + features) and product."""
    cfg = load_config("carbon")
    factors, prices = Factors.load(cfg), Prices.at(as_of)
    return pd.concat([METHODS[p](table.reset_index(drop=True), cfg, factors, prices) for p in products], ignore_index=True)
