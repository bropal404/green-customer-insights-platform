"""Reference tables the simulator reads, with lookups by date."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gcip.config import REFERENCE_DIR


@dataclass
class Reference:
    la: pd.DataFrame
    fuel_prices: pd.DataFrame
    price_cap: pd.DataFrame
    nts_fuel: pd.DataFrame
    nts_bands: pd.DataFrame
    consumption_dist: pd.DataFrame
    family_spending: pd.DataFrame
    pvgis: pd.DataFrame
    tenure_age: pd.DataFrame          # Census RM201: households by tenure and age of reference person, per LA
    tenure_activity: pd.DataFrame     # Census RM133: households by tenure and economic activity of reference person
    house_prices: pd.DataFrame        # UK HPI: monthly average prices by LA and property type
    rates: pd.DataFrame               # Bank of England quoted household rates, monthly (%)
    need: pd.DataFrame                # NEED 2025: 4 million real homes with metered gas and electricity
    mot: pd.DataFrame                 # DVSA MOT 2024: cars by fuel and age with lifetime-average annual mileage

    @classmethod
    def load(cls, directory: Path = REFERENCE_DIR) -> "Reference":
        fuel = pd.read_csv(directory / "fuel_prices_weekly.csv", parse_dates=["week_start"])
        cap = pd.read_csv(directory / "ofgem_price_cap.csv", parse_dates=["start", "end"])
        prices = pd.read_csv(directory / "house_prices_la_monthly.csv", parse_dates=["month"])
        rates = pd.read_csv(directory / "boe_quoted_rates_monthly.csv", parse_dates=["month"])
        return cls(
            la=pd.read_csv(directory / "la_profile.csv"),
            fuel_prices=fuel.sort_values("week_start").reset_index(drop=True),
            price_cap=cap.sort_values("start").reset_index(drop=True),
            nts_fuel=pd.read_csv(directory / "nts_mileage_by_fuel.csv"),
            nts_bands=pd.read_csv(directory / "nts_mileage_bands.csv"),
            consumption_dist=pd.read_csv(directory / "consumption_distribution_2024.csv"),
            family_spending=pd.read_csv(directory / "family_spending_a6_fye2025.csv"),
            pvgis=pd.read_csv(directory / "pvgis_region_yield.csv"),
            tenure_age=pd.read_csv(directory / "census_tenure_age_hrp.csv"),
            tenure_activity=pd.read_csv(directory / "census_tenure_activity_hrp.csv"),
            house_prices=prices.sort_values(["la_code", "month"]).reset_index(drop=True),
            rates=rates.sort_values("month").reset_index(drop=True),
            need=pd.read_parquet(directory / "need_homes.parquet"),
            mot=pd.read_csv(directory / "mot_car_profiles.csv"),
        )

    # ---------------------------------------------------------------- house prices and interest rates by date

    def house_price(self, la_codes: np.ndarray, dates: np.ndarray, prop_type: np.ndarray) -> np.ndarray:
        """UK HPI average price for the authority, property type and month (clamped to 2010-01 .. latest)."""
        hp = self.house_prices
        months = np.sort(hp["month"].unique()).astype("datetime64[M]")
        month = np.clip(np.asarray(dates).astype("datetime64[M]"), months[0], months[-1])
        column = {"detached": "price_detached", "semi": "price_semi", "terraced": "price_terraced", "flat": "price_flat"}
        table = hp.set_index(["la_code", hp["month"].values.astype("datetime64[M]")])
        out = np.empty(len(la_codes))
        keys = pd.MultiIndex.from_arrays([np.asarray(la_codes), month])
        for kind in ("detached", "semi", "terraced", "flat", "caravan"):
            mask = np.asarray(prop_type) == kind
            if not mask.any():
                continue
            values = table[column.get(kind, "price_flat")].reindex(keys[mask]).to_numpy(dtype=float)
            out[mask] = values * (0.4 if kind == "caravan" else 1.0)  # assumption: park homes sell well below flats
        return out

    def rate(self, series: str, dates: np.ndarray) -> np.ndarray:
        """Bank of England quoted rate (%) for the month of each date (clamped to the published range)."""
        months = self.rates["month"].values.astype("datetime64[M]")
        idx = np.searchsorted(months, np.asarray(dates).astype("datetime64[M]"), side="right") - 1
        return self.rates[series].values[np.clip(idx, 0, len(months) - 1)]

    # ---------------------------------------------------------------- prices by date

    def fuel_price_ppl(self, dates: np.ndarray, fuel: np.ndarray) -> np.ndarray:
        """Pump price (pence per litre) in force on each date for petrol or diesel."""
        weeks = self.fuel_prices["week_start"].values.astype("datetime64[D]")
        idx = np.searchsorted(weeks, dates.astype("datetime64[D]"), side="right") - 1
        idx = np.clip(idx, 0, len(weeks) - 1)
        petrol = self.fuel_prices["petrol_ppl"].values[idx]
        diesel = self.fuel_prices["diesel_ppl"].values[idx]
        return np.where(fuel == "diesel", diesel, petrol)

    def cap_rates(self, dates: np.ndarray) -> dict[str, np.ndarray]:
        """Ofgem cap unit rates (p/kWh) and standing charges (p/day) in force on each date."""
        starts = self.price_cap["start"].values.astype("datetime64[D]")
        idx = np.searchsorted(starts, dates.astype("datetime64[D]"), side="right") - 1
        idx = np.clip(idx, 0, len(starts) - 1)
        cap = self.price_cap
        return {
            "elec_unit": cap["elec_unit_p_per_kwh"].values[idx],
            "elec_sc": cap["elec_standing_p_per_day"].values[idx],
            "gas_unit": cap["gas_unit_p_per_kwh"].values[idx],
            "gas_sc": cap["gas_standing_p_per_day"].values[idx],
        }

    # ---------------------------------------------------------------- distributions

    def mileage_band_sampler(self, year: int = 2025) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bands = self.nts_bands[self.nts_bands["year"] == year].copy()
        bands["upper_miles"] = bands["upper_miles"].fillna(45000.0)
        probs = bands["share_pct"].values / bands["share_pct"].sum()
        return bands["lower_miles"].values, bands["upper_miles"].values, probs

    def nts_miles(self, fuel_label: str, year: int = 2025) -> float:
        rows = self.nts_fuel[(self.nts_fuel["year"] == year) & (self.nts_fuel["fuel"] == fuel_label)]
        value = rows["miles_per_year"].dropna()
        return float(value.iloc[0]) if len(value) else float("nan")

    def consumption_shape(self, fuel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Band edges and probabilities of national domestic consumption per meter."""
        rows = self.consumption_dist[self.consumption_dist["fuel"] == fuel].copy()
        rows["upper_kwh"] = rows["upper_kwh"].fillna(rows["lower_kwh"] * 1.5 + 1000.0)
        probs = rows["meters"].values / rows["meters"].sum()
        return rows["lower_kwh"].values, rows["upper_kwh"].values, probs

    def spend_weekly(self, code: str) -> np.ndarray:
        """Average weekly household spend (GBP) for a COICOP code by income decile 1-10."""
        row = self.family_spending[self.family_spending["code"].astype(str) == code]
        if row.empty:
            raise KeyError(f"COICOP code {code} not in ONS A6")
        values = row[[f"decile_{i}" for i in range(1, 11)]].values[0].astype(float)
        return np.nan_to_num(values, nan=0.0)

    def income_weekly_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        fs = self.family_spending.set_index("code")
        lower = fs.loc["income_lower_boundary_gbp_week", [f"decile_{i}" for i in range(1, 11)]].astype(float).values
        mean = fs.loc["income_mean_gbp_week", [f"decile_{i}" for i in range(1, 11)]].astype(float).values
        lower = np.nan_to_num(lower, nan=0.0)
        lower[0] = 60.0
        upper = np.append(lower[1:], mean[-1] * 2.2)
        return lower, upper
