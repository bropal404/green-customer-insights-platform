"""The physical world behind each household: cars, mileage, homes, heating and energy use.

This is the ground truth the carbon engine is later scored against. Quantities
here (km, litres, kWh) are never shown to the models; they only surface as money
in the transactions and as fields the bank plausibly records.
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

from gcip.simulate.population import categorical
from gcip.simulate.reference import Reference

MILES_TO_KM = 1.609344
CAR_FUELS = np.array(["petrol", "diesel", "hev", "phev", "bev", "petrol"])  # 'other' fuels treated as petrol
HEATING = np.array(["gas", "electric", "oil", "lpg", "heat_pump", "other"])
EPC_BANDS = np.array(list("ABCDEFG"))
NTS_FUEL_LABEL = {"petrol": "Petrol", "diesel": "Diesel", "hev": "Hybrid", "phev": "Hybrid", "bev": "Battery electric"}
# Census accommodation type -> NEED property types (VOA classification).
NEED_TYPES = {"detached": ["Detached", "Bungalow"], "semi": ["Semi detached"], "terraced": ["Mid terrace", "End terrace"],
              "flat": ["Flat"]}
NEED_FLOOR_BANDS = {1: (25, 50), 2: (51, 100), 3: (101, 150), 4: (151, 200), 5: (201, 320)}  # m2; band 1 floor and band 5 cap are assumptions
CONSTRUCTION = {1: "pre-1930", 2: "1930-1972", 3: "1973-1999", 4: "2000-or-later"}
CTAX_INDEX = {b: i for i, b in enumerate("ABCDEFGHI")}
# Bedrooms given floor-area band (assumption), multiplied by the authority's Census TS050 mix.
BEDROOM_KERNEL = np.array([[0.70, 0.27, 0.03, 0.00], [0.15, 0.45, 0.35, 0.05], [0.02, 0.13, 0.55, 0.30],
                           [0.00, 0.04, 0.36, 0.60], [0.00, 0.02, 0.18, 0.80]])


def sample_need_homes(pop: pd.DataFrame, heating: np.ndarray, ref: Reference, rng: np.random.Generator) -> np.ndarray:
    """Index of a real NEED home for each household (same region, property type and gas/non-gas heating).

    Among eight candidates the household picks one weighted towards higher council-tax bands at higher incomes
    (assumption), so richer customers live in larger, more valuable homes. Caravans get -1 (no NEED record).
    """
    need = ref.need
    census_type = need["prop_type"].map({t: c for c, ts in NEED_TYPES.items() for t in ts})
    key = need["region"].astype(str) + "|" + census_type.astype(str) + "|" + np.where(need["gas_heated"], "gas", "other")
    pools = {k: np.asarray(v) for k, v in pd.Series(np.arange(len(need))).groupby(key.values).groups.items()}
    ctax = need["council_tax_band"].map(CTAX_INDEX).fillna(2).to_numpy()
    decile_scaled = (pop["income_decile"].values - 5.5) / 2.87
    want = (pop["region"].astype(str) + "|" + pop["accommodation"].astype(str) + "|"
            + np.where(heating == "gas", "gas", "other")).values
    chosen = np.full(len(pop), -1, dtype=np.int64)
    for k in np.unique(want):
        idx = np.flatnonzero(want == k)
        pool = pools.get(k)
        if pool is None:  # caravans, or a region with no such homes off gas: drop the heating condition
            region, acc, _ = k.split("|")
            pool = pools.get(f"{region}|{acc}|gas")
            if pool is None:
                continue
        draws = pool[rng.integers(0, len(pool), (len(idx), 8))]
        weights = np.exp(0.35 * (ctax[draws] - 2.5) * decile_scaled[idx][:, None])
        chosen[idx] = draws[np.arange(len(idx)), categorical(rng, weights)]
    return chosen


def _plate(rng: np.random.Generator, n: int) -> np.ndarray:
    letters = np.array(list(string.ascii_uppercase.replace("I", "").replace("Q", "")))
    a = letters[rng.integers(0, len(letters), (n, 2))]
    b = letters[rng.integers(0, len(letters), (n, 3))]
    year = rng.integers(10, 75, n)
    return np.array([f"{x[0]}{x[1]}{y:02d}{z[0]}{z[1]}{z[2]}" for x, y, z in zip(a, year, b)])


def build_vehicles(pop: pd.DataFrame, cfg: dict, ref: Reference, rng: np.random.Generator) -> pd.DataFrame:
    vcfg = cfg["vehicles"]
    n = len(pop)
    la = ref.la.set_index("la_code").loc[pop["la_code"]].reset_index()
    has_car = pop["n_cars"].values > 0
    accommodation = pop["accommodation"].values
    decile_scaled = (pop["income_decile"].values - 5.5) / 2.87

    p_offstreet = np.select(
        [accommodation == "detached", accommodation == "semi", accommodation == "terraced", accommodation == "flat"],
        [0.93, 0.80, 0.35, 0.12], default=0.55,
    )
    offstreet = rng.random(n) < p_offstreet

    # Primary-car fuel from DfT local shares, tilted by green attitude, income and charging access.
    shares = la[["car_share_petrol", "car_share_diesel", "car_share_hev", "car_share_phev", "car_share_bev",
                 "car_share_other"]].values.copy()
    tilt = np.exp(0.55 * pop["z_green"].values + 0.30 * decile_scaled + 0.45 * offstreet)
    tilt = tilt / tilt[has_car].mean() if has_car.any() else tilt
    shares[:, 4] *= tilt
    shares[:, 3] *= np.sqrt(tilt)
    fuel = CAR_FUELS[categorical(rng, shares)]

    # Car age: cars of 3+ years follow the DVSA MOT 2024 age mix for their fuel; the share under 3 years (not yet
    # tested) is an assumption per fuel.
    mot_fuel = np.where(np.isin(fuel, ["hev", "phev"]), "hybrid", fuel)
    young = rng.random(n) < np.array([vcfg["young_car_share"][f] for f in fuel])
    age = rng.uniform(0.2, 3.0, n)
    for kind, group in ref.mot.groupby("fuel"):
        mask = (mot_fuel == kind) & ~young
        if mask.any():
            probs = (group["cars"] / group["cars"].sum()).to_numpy()
            years = group["age_year"].to_numpy()[rng.choice(len(group), mask.sum(), p=probs)]
            age[mask] = years + rng.random(mask.sum())
    age = np.clip(age, 0.2, vcfg["max_age_years"])

    # Annual mileage: NTS0904 band distribution, scaled by fuel type (NTS0901) and use.
    lower, upper, probs = ref.mileage_band_sampler(2025)
    band = rng.choice(len(probs), n, p=probs)
    miles = lower[band] + rng.random(n) * (upper[band] - lower[band])
    all_cars = ref.nts_miles("All cars")
    fuel_factor = np.array([ref.nts_miles(NTS_FUEL_LABEL[f]) / all_cars for f in fuel])
    use = np.where(pop["commute_mode"].values == "car_driver", 1.25, 1.0)
    use = use * np.where(pop["employment"].values == "retired", 0.80, 1.0)
    use = use * np.exp(vcfg["mileage_income_slope"] * (pop["income_decile"].values - 5.5))
    # Newer cars are driven further: MOT 2024 mileage by fuel and age, relative to that fuel's average.
    mot = ref.mot.assign(rel=ref.mot["mean_annual_miles"] / ref.mot["fuel"].map(
        ref.mot.groupby("fuel").apply(lambda g: np.average(g["mean_annual_miles"], weights=g["cars"]), include_groups=False)))
    rel = mot.set_index(["fuel", "age_year"])["rel"]
    keys = pd.MultiIndex.from_arrays([mot_fuel, np.clip(np.floor(age), 3, 25).astype(int)])
    age_factor = rel.reindex(keys).fillna(1.0).to_numpy()
    factor = fuel_factor * use * age_factor
    factor = factor / factor[has_car].mean() if has_car.any() else factor
    annual_km = miles * factor * MILES_TO_KM

    kml_centre = np.array([vcfg["km_per_litre_centre"].get(f, 12.85) for f in fuel])
    km_per_litre = kml_centre * np.exp(rng.normal(0, vcfg["km_per_litre_sd_log"], n))
    kwh_per_km = vcfg["ev_kwh_per_km_centre"] * np.exp(rng.normal(0, vcfg["ev_kwh_per_km_sd_log"], n))
    elec_share = np.select([fuel == "bev", fuel == "phev"], [1.0, vcfg["phev_electric_share"]], default=0.0)

    # Additional household cars: petrol or diesel runabouts.
    extra_cars = np.clip(pop["n_cars"].values - 1, 0, None)
    extra_km = extra_cars * 0.55 * all_cars * MILES_TO_KM * np.exp(rng.normal(0, 0.35, n))
    extra_fuel = np.where(rng.random(n) < 0.72, "petrol", "diesel")
    extra_kml = np.where(extra_fuel == "diesel", 14.96, 12.85) * np.exp(rng.normal(0, 0.15, n))

    # Car finance.
    p_fin = vcfg["motor_finance_share_of_cars"] * np.where(age < 4, 1.9, np.where(age > 8, 0.25, 1.0))
    financed = has_car & (rng.random(n) < np.clip(p_fin, 0, 0.95))
    term = rng.choice(vcfg["motor_finance_term_months"], n)
    elapsed = np.where(financed, rng.integers(0, np.maximum(term, 2)), 0)
    elapsed = np.minimum(elapsed, (age * 12).astype(int))
    start = np.datetime64("2025-01-01", "M") - elapsed.astype("timedelta64[M]")
    end = start + term.astype("timedelta64[M]")
    new_value = np.select([fuel == "bev", fuel == "phev", fuel == "hev", fuel == "diesel"],
                          [41000, 38000, 31000, 30000], default=26000) * np.exp(rng.normal(0, 0.25, n))
    age_at_origination = np.maximum(age - elapsed / 12.0, 0.0)
    value_at_origination = new_value * 0.86 ** age_at_origination
    finance_amount = value_at_origination * rng.uniform(0.7, 0.95, n)
    rate = rng.uniform(0.06, 0.11, n) / 12
    monthly_payment = finance_amount * rate / (1 - (1 + rate) ** (-term))
    with_us = financed & (rng.random(n) < vcfg["motor_finance_with_us_share"])
    outstanding = np.where(financed, finance_amount * (1 - elapsed / term), 0.0)

    london = pop["region"].values == "London"
    caz_region = np.isin(pop["region"].values, ["West Midlands", "North East", "South West", "Yorkshire and The Humber"])
    noncompliant = ((fuel == "diesel") & (age >= 11)) | ((fuel == "petrol") & (age >= 20))

    out = pd.DataFrame({
        "customer_id": pop["customer_id"].values,
        "has_car": has_car,
        "car_fuel": np.where(has_car, fuel, "none"),
        "car_age_years": np.where(has_car, age.round(1), np.nan),
        "car_reg": np.where(has_car, _plate(rng, n), ""),
        "annual_km_true": np.where(has_car, annual_km, 0.0),
        "km_per_litre_true": np.where(has_car & (fuel != "bev"), km_per_litre, np.nan),
        "ev_kwh_per_km_true": np.where(has_car & np.isin(fuel, ["bev", "phev"]), kwh_per_km, np.nan),
        "ev_km_share": np.where(has_car, elec_share, 0.0),
        "extra_cars": extra_cars,
        "extra_km_true": np.where(has_car, extra_km, 0.0),
        "extra_fuel": np.where(extra_cars > 0, extra_fuel, "none"),
        "extra_km_per_litre_true": np.where(extra_cars > 0, extra_kml, np.nan),
        "offstreet_parking": offstreet,
        "motor_finance": financed,
        "motor_finance_with_us": with_us,
        "motor_finance_start": np.where(financed, start.astype("datetime64[D]"), np.datetime64("NaT")),
        "motor_finance_end": np.where(financed, end.astype("datetime64[D]"), np.datetime64("NaT")),
        "motor_finance_monthly": np.where(financed, monthly_payment.round(2), 0.0),
        "motor_finance_outstanding": outstanding.round(0),
        "vehicle_value_at_origination": np.where(financed, value_at_origination.round(0), np.nan),
        "ulez_payer": has_car & noncompliant & london & (rng.random(n) < 0.6),
        "caz_payer": has_car & noncompliant & caz_region & (rng.random(n) < 0.3),
    })
    out["cash_fuel_share"] = np.where(rng.random(n) < vcfg["cash_fuel_customer_share"], vcfg["cash_fuel_fill_share"], 0.0)
    out["fuel_on_external_card"] = rng.random(n) < vcfg["external_card_customer_share"]
    pub = vcfg["public_charging_share_bev"]
    out["public_charging_share"] = np.where(offstreet, pub["offstreet"], pub["onstreet"])
    return out


def build_homes(pop: pd.DataFrame, veh: pd.DataFrame, cfg: dict, ref: Reference, rng: np.random.Generator) -> pd.DataFrame:
    hcfg = cfg["homes"]
    n = len(pop)
    la = ref.la.set_index("la_code").loc[pop["la_code"]].reset_index()
    acc = pop["accommodation"].values
    owner = np.isin(pop["tenure"].values, ["owned_outright", "mortgage"])
    house = acc != "flat"
    decile_scaled = (pop["income_decile"].values - 5.5) / 2.87

    # Heating from Census TS046 shares; flats skew electric, oil and LPG only in houses.
    heat_probs = la[["share_heat_mains_gas", "share_heat_electric", "share_heat_oil", "share_heat_lpg",
                     "share_heat_heat_pump_or_renewable", "share_heat_other"]].values.copy()
    heat_probs[:, 1] *= np.where(acc == "flat", 2.2, 0.85)
    heat_probs[:, 2] *= np.where(house, 1.1, 0.05)
    heat_probs[:, 3] *= np.where(house, 1.1, 0.05)
    heat_probs[:, 4] *= np.where(owner & house, 1.4 * np.exp(0.4 * pop["z_green"].values), 0.4)
    heating = HEATING[categorical(rng, heat_probs)]

    # Each home is a real NEED home of the same region, type and heating (property attributes, metered use).
    need = ref.need
    pick = sample_need_homes(pop, heating, ref, rng)
    has_need = pick >= 0
    rec = need.iloc[np.where(has_need, pick, 0)].reset_index(drop=True)
    band = rec["floor_area_band"].to_numpy()
    low = np.array([NEED_FLOOR_BANDS[b][0] for b in band]); high = np.array([NEED_FLOOR_BANDS[b][1] for b in band])
    need_area = low + rng.random(n) * (high - low)
    median_area = np.array([hcfg["floor_area_m2"][a] for a in acc])
    old_area = median_area * np.exp(rng.normal(0, hcfg["floor_area_sd_log"], n))   # caravans only
    floor_area = np.where(has_need, need_area, old_area)

    # EPC band: the NEED record's band; 'No EPC' homes still have a true band (drawn from homes of the same type
    # and age) but no certificate on the register, so the bank cannot match one.
    epc_text = rec["epc"].astype(str).to_numpy()
    registered = has_need & (epc_text != "No EPC")
    epc_idx = np.full(n, 3)
    epc_idx = np.where(epc_text == "A/B", np.where(rng.random(n) < 0.1, 0, 1), epc_idx)
    epc_idx = np.where(epc_text == "C", 2, epc_idx)
    epc_idx = np.where(epc_text == "D", 3, epc_idx)
    epc_idx = np.where(epc_text == "E", 4, epc_idx)
    epc_idx = np.where(epc_text == "F/G", np.where(rng.random(n) < 0.75, 5, 6), epc_idx)
    unregistered = ~registered
    if unregistered.any():
        rated = need[need["epc"] != "No EPC"]
        mix = rated.groupby(["prop_type", "age_band"])["epc"].value_counts(normalize=True).unstack(fill_value=0.0)
        mix = mix.reindex(columns=["A/B", "C", "D", "E", "F/G"], fill_value=0.0)
        keys = pd.MultiIndex.from_arrays([rec["prop_type"].to_numpy(), rec["age_band"].to_numpy()])
        probs = mix.reindex(keys).fillna(0.2).to_numpy()
        draw = categorical(rng, probs + 1e-9)
        drawn_idx = np.array([1, 2, 3, 4, 5])[draw]
        epc_idx = np.where(unregistered & has_need, drawn_idx, epc_idx)
        caravan_probs = np.array([hcfg["epc_band_probs"]["caravan"]] * n)
        epc_idx = np.where(~has_need, categorical(rng, caravan_probs), epc_idx)
    # Heat pumps sit mostly in better-rated homes.
    epc_idx = np.where((heating == "heat_pump") & (epc_idx > 3), epc_idx - 1, epc_idx)

    # Metered 2023 use from the NEED record, re-anchored from its region to the customer's authority (DESNZ 2024).
    region_gas = need[need["gas_heated"]].groupby("region")["gas_kwh_2023"].mean()
    region_elec = need.groupby("region")["elec_kwh_2023"].mean()
    gas_scale = la["gas_dom_mean_kwh"].values / pop["region"].map(region_gas).to_numpy()
    elec_scale = la["elec_dom_mean_kwh"].values / pop["region"].map(region_elec).to_numpy()
    need_gas = rec["gas_kwh_2023"].to_numpy(dtype=float) * gas_scale
    need_elec = rec["elec_kwh_2023"].to_numpy(dtype=float) * elec_scale

    gas_factor = np.array(hcfg["gas_demand_by_epc"])[epc_idx]
    heat_noise = np.exp(rng.normal(0, 0.28, n))
    # Fallback space and water heating demand in gas-boiler kWh, anchored on the LA mean (DESNZ 2024): used for
    # homes not heated by gas and for gas homes whose NEED reading is missing or invalid.
    formula_heat = (la["gas_dom_mean_kwh"].values * (floor_area / 92.0) ** hcfg["gas_demand_area_elasticity"]
                    * gas_factor / 1.05 * heat_noise)
    gas_from_need = has_need & (heating == "gas") & np.isfinite(need_gas) & (need_gas > 0)
    heat_demand = np.where(gas_from_need, need_gas, formula_heat)
    gas_kwh = np.where(heating == "gas", heat_demand, 0.0)
    oil_litres = np.where(heating == "oil", heat_demand / hcfg["burning_oil_kwh_per_litre"], 0.0)
    lpg_litres = np.where(heating == "lpg", heat_demand / 7.1, 0.0)  # LPG about 7.1 kWh per litre (gross)

    base_elec = (la["elec_dom_median_kwh"].values * (pop["household_size"].values / 2.4) ** hcfg["elec_household_size_elasticity"]
                 * np.exp(hcfg["elec_income_slope"] * (pop["income_decile"].values - 5.5)) * np.exp(rng.normal(0, 0.30, n)))
    electric_heat = np.where(heating == "electric", floor_area * hcfg["electric_heating_kwh_per_m2"]
                             * gas_factor * np.exp(rng.normal(0, 0.2, n)), 0.0)
    heat_pump_elec = np.where(heating == "heat_pump",
                              heat_demand * hcfg["boiler_efficiency"] / hcfg["heat_pump_spf"], 0.0)
    ev_home = (veh["annual_km_true"].values * veh["ev_km_share"].values * np.nan_to_num(veh["ev_kwh_per_km_true"].values)
               * (1 - veh["public_charging_share"].values))
    # Metered NEED electricity already includes electric space heating; heat pumps keep the explicit model.
    elec_from_need = has_need & (heating != "heat_pump") & np.isfinite(need_elec) & (need_elec > 0)
    household_elec = np.where(elec_from_need, need_elec, base_elec + electric_heat + heat_pump_elec)
    demand_elec = household_elec + ev_home

    # Solar PV: DESNZ sites per household, placed on suitable owner-occupied houses.
    # Rate among owner houses = LA rate / share of owner houses, tilted by green
    # attitude and income, then divided by the roof-suitable share so the LA total holds.
    owner_house = owner & house & (acc != "caravan")
    share_owner_house = max(owner_house.mean(), 1e-6)
    tilt = np.exp(0.35 * pop["z_green"].values + 0.2 * decile_scaled)
    tilt = tilt / max(tilt[owner_house].mean() if owner_house.any() else 1.0, 1e-9)
    p_pv = la["pv_rate_per_household"].values / share_owner_house * tilt / hcfg["roof_suitable_share"]
    # Rescale so the expected number of solar homes equals the DESNZ local rates summed over customers.
    expected = (np.clip(p_pv, 0, 0.7) * owner_house * hcfg["roof_suitable_share"]).sum()
    p_pv = p_pv * la["pv_rate_per_household"].values.sum() / max(expected, 1e-9)
    has_pv = owner_house & (rng.random(n) < np.clip(p_pv, 0, 0.7))
    roof_ok = rng.random(n) < hcfg["roof_suitable_share"]
    has_pv = has_pv & roof_ok
    kwp = hcfg["pv_kwp_mean"] * np.exp(rng.normal(0, hcfg["pv_kwp_sd_log"], n))
    yield_kwh = la["pv_kwh_per_kwp_year"].values * rng.uniform(0.80, 0.98, n)  # real roofs miss the optimum
    generation = np.where(has_pv, kwp * yield_kwh, 0.0)
    self_used = np.minimum(generation * hcfg["pv_self_consumption"], demand_elec * 0.6)
    imported = np.maximum(demand_elec - self_used, 400.0)
    exported = np.where(has_pv, generation - self_used, 0.0)

    # Energy billing arrangement.
    billing = hcfg["billing"]
    low_income_or_social = (pop["income_decile"].values <= 3) | (pop["tenure"].values == "social_rent")
    p_prepay = billing["prepayment_share"] * np.where(low_income_or_social, 2.2, 0.55)
    prepay = rng.random(n) < np.clip(p_prepay, 0, 0.6)
    gas_home = heating == "gas"
    separate = gas_home & ~prepay & (rng.random(n) < billing["separate_dd_share"] / (1 - billing["prepayment_share"]))
    billing_type = np.where(prepay, "prepayment", np.where(~gas_home, "electric_only", np.where(separate, "separate", "dual")))
    paid_elsewhere = rng.random(n) < billing["paid_from_other_bank_share"]

    bed_mix = la[["share_bedrooms_1", "share_bedrooms_2", "share_bedrooms_3", "share_bedrooms_4plus"]].values
    floor_band = np.clip(np.digitize(floor_area, [50.5, 100.5, 150.5, 200.5]), 0, 4)
    bed_probs = bed_mix * BEDROOM_KERNEL[floor_band] + 1e-9
    bed_probs = bed_probs / bed_probs.sum(axis=1, keepdims=True)
    target = bed_mix.mean(axis=0)  # customers are drawn in proportion to households, so this is the Census mix
    for _ in range(6):  # fit the overall mix to Census TS050 while keeping the floor-area pattern
        bed_probs = bed_probs * (target / bed_probs.mean(axis=0))[None, :]
        bed_probs = bed_probs / bed_probs.sum(axis=1, keepdims=True)
    bedrooms = categorical(rng, bed_probs) + 1

    out = pd.DataFrame({
        "customer_id": pop["customer_id"].values,
        "need_record": pick,
        "property_detail": np.where(has_need, rec["prop_type"].to_numpy(), "Caravan or mobile home"),
        "construction_period": np.where(has_need, pd.Series(rec["age_band"].to_numpy()).map(CONSTRUCTION).to_numpy(), "unknown"),
        "council_tax_band": np.where(has_need, rec["council_tax_band"].to_numpy(), "A"),
        "imd_quintile": np.where(has_need, rec["imd_quintile"].to_numpy(), np.nan),
        "loft_insulation_scheme": has_need & rec["loft_insulation_scheme"].to_numpy(),
        "cavity_insulation_scheme": has_need & rec["cavity_insulation_scheme"].to_numpy(),
        "epc_registered": registered,
        "bedrooms": bedrooms,
        "floor_area_m2": floor_area.round(0),
        "epc_band": EPC_BANDS[epc_idx],
        "main_heating": heating,
        "gas_kwh_year_true": gas_kwh.round(0),
        "elec_kwh_year_true": imported.round(0),
        "elec_demand_kwh_year_true": demand_elec.round(0),
        "ev_home_kwh_year_true": ev_home.round(0),
        "oil_litres_year_true": oil_litres.round(0),
        "lpg_litres_year_true": lpg_litres.round(0),
        "heat_demand_kwh_year_true": heat_demand.round(0),
        "has_solar": has_pv,
        "pv_kwp": np.where(has_pv, kwp.round(2), 0.0),
        "pv_generation_kwh_year_true": generation.round(0),
        "pv_export_kwh_year_true": exported.round(0),
        "roof_suitable": roof_ok,
        "pv_kwh_per_kwp_region": la["pv_kwh_per_kwp_year"].values,
        "billing_type": billing_type,
        "energy_paid_elsewhere": paid_elsewhere,
    })
    return out
