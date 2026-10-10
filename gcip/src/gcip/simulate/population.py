"""Customers: geography, demographics, tenure, household, employment and hidden traits.

Each customer is the main account holder of one household. Local-authority
marginals come from Census 2021 (la_profile.csv); conditional shapes that the
Census tables do not cross-tabulate are assumptions in configs/simulation.yaml.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gcip.simulate.reference import Reference

TENURES = np.array(["owned_outright", "mortgage", "social_rent", "private_rent"])
ACCOMMODATION = np.array(["detached", "semi", "terraced", "flat", "caravan"])
EMPLOYMENT = np.array(["employed", "self_employed", "retired", "student", "not_working"])
COMMUTE_MODES = ["wfh", "metro_tram", "train", "bus", "taxi", "motorcycle", "car_driver", "car_passenger",
                 "bicycle", "foot", "other"]
SOC_GROUPS = np.array(["managers_directors", "professional", "associate_professional", "administrative", "skilled_trades",
                       "caring_leisure", "sales_customer_service", "process_plant", "elementary"])
# Assumption: higher-income households skew to SOC groups 1-3, lower-income to 6-9 (within each area's Census mix).
SOC_INCOME_TILT = np.array([0.55, 0.65, 0.35, 0.0, -0.1, -0.35, -0.3, -0.3, -0.5])
ACTIVITY_GROUP = {"employed": "work", "unemployed": "not_working", "retired": "retired", "student": "student",
                  "home_family": "not_working", "sick_disabled": "not_working", "other_inactive": "not_working"}


def categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Vectorised draw: one category index per row of a (n, k) probability matrix."""
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random(len(probs))[:, None]
    return np.minimum((draws > cumulative).sum(axis=1), probs.shape[1] - 1)


def calibrate_scale(base: np.ndarray, target: float, odds: bool = True) -> np.ndarray:
    """Scale probabilities (on the odds scale) by one constant so their mean hits ``target``."""
    base = np.clip(base, 1e-6, 1 - 1e-6)
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        k = np.sqrt(lo * hi)
        o = k * base / (1 - base)
        if (o / (1 + o)).mean() > target:
            hi = k
        else:
            lo = k
    o = np.sqrt(lo * hi) * base / (1 - base)
    return o / (1 + o)


def town_name(la_name: str) -> str:
    name = re.sub(r",.*$", "", la_name)
    name = re.sub(r"^(City of |County of )", "", name)
    return name.upper().replace("-UPON-", " UPON ").strip()[:16]


def age_band_label(age: np.ndarray) -> np.ndarray:
    edges = [18, 25, 35, 45, 55, 65, 75, 200]
    labels = np.array(["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-90"])
    return labels[np.digitize(age, edges[1:-1])]


def build_population(n: int, cfg: dict, ref: Reference, rng: np.random.Generator) -> pd.DataFrame:
    pop_cfg = cfg["population"]
    la = ref.la.reset_index(drop=True)

    la_idx = rng.choice(len(la), size=n, p=(la["households"] / la["households"].sum()).values)
    frame = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "la_code": la["la_code"].values[la_idx],
        "la_name": la["la_name"].values[la_idx],
        "region": la["region"].values[la_idx],
    })
    frame["town"] = [town_name(x) for x in frame["la_name"]]
    rows = la.iloc[la_idx].reset_index(drop=True)

    # Tenure from Census TS054 shares.
    tenure_probs = rows[["share_owned_outright", "share_mortgage", "share_social_rent", "share_private_rent"]].values
    tenure_idx = categorical(rng, tenure_probs)
    frame["tenure"] = TENURES[tenure_idx]

    # Age given tenure: band shape is an assumption; the share of households whose reference person is 65+
    # matches Census 2021 RM201 for the customer's own authority and tenure.
    bands = np.array(pop_cfg["age_bands"])
    age_probs = np.array([pop_cfg["age_given_tenure"][t] for t in frame["tenure"]])
    ta = ref.tenure_age.pivot_table(index=["la_code", "tenure"], columns="hrp_age", values="households", aggfunc="sum")
    p65 = (ta["65+"] / (ta["16-64"] + ta["65+"])).reindex(
        pd.MultiIndex.from_arrays([frame["la_code"].values, frame["tenure"].values])).to_numpy()
    p65 = np.where(np.isnan(p65), age_probs[:, 5:].sum(axis=1), p65)
    under, over = age_probs[:, :5], age_probs[:, 5:]
    age_probs = np.hstack([under / under.sum(axis=1, keepdims=True) * (1 - p65)[:, None],
                           over / over.sum(axis=1, keepdims=True) * p65[:, None]])
    band_idx = categorical(rng, age_probs)
    low, high = bands[band_idx, 0], bands[band_idx, 1]
    frame["age"] = (low + rng.random(n) * (high - low + 1)).astype(int).clip(pop_cfg["min_age"], pop_cfg["max_age"])
    frame["age_band"] = age_band_label(frame["age"].values)

    # Income decile given tenure, shifted up in London and down in lower-income regions.
    decile_probs = np.array([pop_cfg["income_decile_given_tenure"][t] for t in frame["tenure"]])
    decile = categorical(rng, decile_probs) + 1
    shifts = pop_cfg.get("income_region_shift", {})
    for region, (direction, prob) in shifts.items():
        mask = (frame["region"].values == region) & (rng.random(n) < prob)
        decile = np.where(mask, np.clip(decile + direction, 1, 10), decile)
    retired_low = (frame["age"].values >= 70) & (rng.random(n) < 0.25)
    decile = np.where(retired_low, np.clip(decile - 2, 1, 10), decile)
    frame["income_decile"] = decile
    lower, upper = ref.income_weekly_bounds()
    weekly = lower[decile - 1] + rng.random(n) * (upper[decile - 1] - lower[decile - 1])
    frame["gross_income_week"] = weekly.round(2)

    # Accommodation (Census TS044) with renters far likelier to live in flats.
    flat_mult = np.array([pop_cfg["flat_multiplier_given_tenure"][t] for t in frame["tenure"]])
    p_flat = np.clip(rows["share_flat"].values * flat_mult, 0.01, 0.97)
    p_flat = calibrate_scale(p_flat, float(rows["share_flat"].mean()))
    house_probs = rows[["share_detached", "share_semi", "share_terraced"]].values.copy()
    owner = np.isin(frame["tenure"].values, ["owned_outright", "mortgage"])
    house_probs[:, 0] *= np.where(owner, 1.35, 0.55)  # owners skew to detached
    house_probs = house_probs / house_probs.sum(axis=1, keepdims=True)
    p_caravan = rows["share_caravan"].values
    probs = np.column_stack([house_probs * (1 - p_flat - p_caravan)[:, None], p_flat, p_caravan])
    frame["accommodation"] = ACCOMMODATION[categorical(rng, np.clip(probs, 1e-6, None))]

    # Household size (Census TS017), older households smaller.
    size_cols = [f"share_hh_size_{k}" for k in ("1", "2", "3", "4", "5", "6", "7", "8plus")]
    size = categorical(rng, rows[size_cols].values) + 1
    older = frame["age"].values >= 65
    size = np.where(older & (rng.random(n) < 0.85), np.minimum(size, 2), size)
    young = frame["age"].values < 25
    size = np.where(young & (rng.random(n) < 0.5), np.minimum(size, 2), size)
    frame["household_size"] = size

    # Employment: the age pattern (assumption) is fitted so each authority-and-tenure cell matches Census 2021
    # RM133 (economic activity of household reference persons); the employed/self-employed split follows TS066.
    emp_probs = np.array([pop_cfg["employment_given_age"][b] for b in frame["age_band"]])
    frame["employment"] = EMPLOYMENT[categorical(rng, fit_employment(emp_probs, frame, rows, ref))]

    # Household composition and dependants (Census TS003 shares of the authority, assumption on age and size).
    composition, dependants = household_composition(frame, rows, rng)
    frame["household_composition"] = composition
    frame["dependants"] = dependants

    # Occupation of people in work: the authority's Census TS063 mix, tilted by income (assumption).
    occ_probs = rows[[f"share_occupation_{g}" for g in SOC_GROUPS]].values * np.exp(
        SOC_INCOME_TILT[None, :] * ((decile - 5.5) / 2.87)[:, None])
    occupation = SOC_GROUPS[categorical(rng, occ_probs)]
    working = np.isin(frame["employment"].values, ["employed", "self_employed"])
    frame["occupation"] = np.where(working, occupation, np.where(frame["employment"].values == "not_working",
                                                                 "not_in_work", frame["employment"].values))

    # Cars (Census TS045): odds of owning any car fall for flats and low incomes.
    car_probs = rows[["share_cars_0", "share_cars_1", "share_cars_2", "share_cars_3plus"]].values.copy()
    any_car = 1.0 - car_probs[:, 0]
    odds = any_car / np.clip(1 - any_car, 1e-6, None)
    mult = cfg["population"]["car_odds_multiplier"]
    odds = odds * np.where(frame["accommodation"].values == "flat", mult["flat"], 1.0)
    odds = odds * np.exp(mult["decile_log_odds_slope"] * (decile - 5.5))
    p_any = calibrate_scale(odds / (1 + odds), float(1.0 - rows["share_cars_0"].mean()))
    has_car = rng.random(n) < p_any
    multi = car_probs[:, 1:] / car_probs[:, 1:].sum(axis=1, keepdims=True)
    n_cars = categorical(rng, multi) + 1
    frame["n_cars"] = np.where(has_car, np.minimum(n_cars, np.where(size == 1, 1, 3)), 0)

    # Commute mode for workers (Census TS061), conditional on having a car.
    commute_probs = rows[[f"share_commute_{m}" for m in COMMUTE_MODES]].values.copy()
    car_col = COMMUTE_MODES.index("car_driver")
    commute_probs[:, car_col] *= np.where(frame["n_cars"].values == 0, 0.0, 1.25)
    mode_idx = categorical(rng, np.clip(commute_probs, 1e-9, None))
    working = np.isin(frame["employment"].values, ["employed", "self_employed"])
    frame["commute_mode"] = np.where(working, np.array(COMMUTE_MODES)[mode_idx], "none")
    student = frame["employment"].values == "student"
    frame.loc[student, "commute_mode"] = np.where(rng.random(student.sum()) < 0.5, "bus", "foot")

    # Vulnerability recorded by the bank: likelier for the very old and the lowest incomes.
    base = pop_cfg["vulnerability_rate"]
    disabled = rows["share_disabled"].values
    p_vul = (base * np.where(frame["age"].values >= 75, 2.2, 1.0) * np.where(decile <= 2, 1.8, 1.0)
             * disabled / disabled.mean())  # areas with more disability (Census TS038) record more vulnerability
    frame["vulnerable"] = rng.random(n) < np.clip(p_vul, 0, 0.6)

    # Hidden traits (standard normal latent variables). Never written to customer tables.
    traits = cfg["hidden_traits"]
    decile_scaled = (decile - 5.5) / 2.87
    age_scaled = (frame["age"].values - 48.0) / 17.0
    london = (frame["region"].values == "London").astype(float)
    # Each trait = an observable part (mu, kept for the evaluation harness) + an
    # unobservable residual. The residual is what keeps labels from being a rule.
    mu = {
        "green": (traits["green"]["age_under_40"] * (frame["age"].values < 40)
                  + traits["green"]["decile_scaled"] * decile_scaled + traits["green"]["london"] * london),
        "liquidity": (traits["liquidity"]["decile_scaled"] * decile_scaled
                      + traits["liquidity"]["age_over_55"] * (frame["age"].values > 55)),
        "stay": traits["stay"]["age_over_45"] * (frame["age"].values > 45) + traits["stay"]["owner"] * owner,
        "digital": traits["digital"]["age_scaled"] * age_scaled + traits["digital"]["decile_scaled"] * decile_scaled,
        "risk": traits["risk"]["decile_scaled"] * decile_scaled + traits["risk"]["age_over_65"] * (frame["age"].values > 65),
    }
    for name, value in mu.items():
        frame[f"zmu_{name}"] = value
        frame[f"z_{name}"] = value + rng.normal(0, traits[name]["residual_sd"], n)

    # What the bank recorded at onboarding. Some answers are out of date.
    declared = np.select(
        [frame["tenure"].values == "owned_outright", frame["tenure"].values == "mortgage"],
        ["homeowner", "homeowner_mortgage"], default="tenant",
    )
    stale = rng.random(n) < 0.10
    alternatives = np.array(["homeowner", "homeowner_mortgage", "tenant", "living_with_family"])
    frame["declared_residential_status"] = np.where(stale, alternatives[rng.integers(0, 4, n)], declared)
    years_customer = np.minimum(rng.gamma(2.0, 6.0, n), frame["age"].values - 17)
    since = np.datetime64("2025-01-01") - (years_customer * 365.25).astype("timedelta64[D]")
    frame["customer_since"] = since.astype("datetime64[D]")
    frame["marketing_consent"] = rng.random(n) < 0.72
    # Years at the current address on 1 January 2025 (assumption shaped on English Housing Survey lengths of residence).
    shape_scale = {"owned_outright": (2.2, 8.0), "mortgage": (1.8, 4.5), "social_rent": (1.6, 5.0), "private_rent": (1.3, 2.2)}
    years = np.zeros(n)
    for tenure_name, (shape, scale) in shape_scale.items():
        mask = frame["tenure"].values == tenure_name
        years[mask] = rng.gamma(shape, scale, mask.sum())
    frame["years_at_address"] = np.clip(years, 0.1, np.maximum(frame["age"].values - 17, 0.5)).round(1)
    return frame


def fit_employment(base: np.ndarray, frame: pd.DataFrame, rows: pd.DataFrame, ref: Reference, iterations: int = 6) -> np.ndarray:
    """Rescale age-based probabilities so every (region, tenure) cell matches Census RM133 shares.

    Regions rather than authorities: with authority-level cells most cells hold one or two customers and the fit
    is biased. Local variation still comes through the authority's age mix (RM201).
    """
    act = ref.tenure_activity.assign(group=ref.tenure_activity["hrp_activity"].map(ACTIVITY_GROUP),
                                     region=ref.tenure_activity["la_code"].map(ref.la.set_index("la_code")["region"]))
    act = act.groupby(["region", "tenure", "group"])["households"].sum().unstack(fill_value=0.0)
    act = act[["work", "retired", "student", "not_working"]]
    target = (act.div(act.sum(axis=1), axis=0)).reindex(
        pd.MultiIndex.from_arrays([frame["region"].values, frame["tenure"].values])).to_numpy()
    work = base[:, 0] + base[:, 1]
    g = np.column_stack([work, base[:, 2], base[:, 3], base[:, 4]])
    cells = pd.factorize(frame["region"].astype(str) + "|" + frame["tenure"].astype(str))[0]
    valid = ~np.isnan(target).any(axis=1)
    for _ in range(iterations):
        sums = np.zeros((cells.max() + 1, 4))
        np.add.at(sums, cells, g)
        counts = np.bincount(cells, minlength=cells.max() + 1)[:, None]
        mean = sums / np.maximum(counts, 1)
        factor = np.where(valid[:, None], target / np.maximum(mean[cells], 1e-9), 1.0)
        g = g * np.clip(factor, 0.05, 20.0)
        g = g / g.sum(axis=1, keepdims=True)
    se_share = base[:, 1] / np.maximum(work, 1e-9)
    local_se = rows["self_employed_share_of_workers"].values
    se_share = np.clip(se_share * local_se / local_se.mean(), 0.0, 0.9)
    return np.column_stack([g[:, 0] * (1 - se_share), g[:, 0] * se_share, g[:, 1], g[:, 2], g[:, 3]])


def household_composition(frame: pd.DataFrame, rows: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Household type and number of dependent children, consistent with household size (Census TS017/TS003)."""
    size = frame["household_size"].values
    age = frame["age"].values
    share_dep = rows["share_hh_dependent_children"].values
    share_lone = rows["share_hh_lone_parent_dependent"].values
    age_factor = np.select([age < 25, age < 55, age < 65], [0.5, 1.0, 0.35], default=0.03)
    base = np.where(size >= 3, 0.85, np.where(size == 2, 0.12, 0.0)) * age_factor * share_dep / share_dep.mean()
    has_dep = (size >= 2) & (rng.random(len(size)) < calibrate_scale(base, float(share_dep.mean())) * (size >= 2))
    lone = has_dep & ((size == 2) | (rng.random(len(size)) < share_lone / np.maximum(share_dep, 1e-6)))
    dependants = np.where(has_dep, np.maximum(np.where(lone, size - 1, size - 2), 1), 0)
    composition = np.select(
        [size == 1, has_dep & lone, has_dep, (size == 2) & (age >= 66), size == 2],
        ["one_person", "lone_parent", "couple_with_children", "older_couple", "couple_no_children"], default="other_adults")
    return composition, np.minimum(dependants, 6)
