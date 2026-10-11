"""Transactions: turn each household's physical world and finances into bank statement lines.

Generation is vectorised over (customer, month) grids inside a chunk of customers;
chunks run in parallel and are checkpointed by the orchestrator. Every row also
gets a truth record (true category, merchant, physical quantity) for the vault.

Seasonal shapes below are modelling assumptions (documented in the data card):
gas use peaks in winter, electricity varies mildly, driving rises in summer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.simulate.lexicon import FIRST_NAMES, SURNAMES, Merchants, pick_merchants
from gcip.simulate.reference import Reference

GAS_MONTH = np.array([0.155, 0.135, 0.125, 0.085, 0.055, 0.035, 0.030, 0.030, 0.040, 0.070, 0.110, 0.140])
ELEC_MONTH = np.array([0.100, 0.090, 0.088, 0.080, 0.075, 0.070, 0.072, 0.072, 0.074, 0.082, 0.093, 0.104])
KM_MONTH = np.array([0.080, 0.078, 0.083, 0.083, 0.086, 0.085, 0.090, 0.090, 0.084, 0.083, 0.080, 0.078])
GAS_MONTH, ELEC_MONTH, KM_MONTH = GAS_MONTH / GAS_MONTH.sum(), ELEC_MONTH / ELEC_MONTH.sum(), KM_MONTH / KM_MONTH.sum()

PTYPE_ISO = {
    "DEB": "CardPayment", "DD": "DirectDebit", "SO": "StandingOrder", "FPO": "FasterPaymentOut",
    "FPI": "FasterPaymentIn", "BGC": "BankGiroCredit", "CPT": "CashWithdrawal", "TFR": "InternalTransfer",
    "CHG": "Charge",
}
CHANNEL_PTYPE = {"card": "DEB", "online": "DEB", "dd": "DD", "so": "SO", "fp": "FPO", "fpo": "FPO", "fpi": "FPI",
                 "bgc": "BGC", "cpt": "CPT", "tfr": "TFR", "chg": "CHG"}


class Buffer:
    """Collects transaction rows as column arrays, then builds one frame."""

    def __init__(self) -> None:
        self.parts: list[dict[str, np.ndarray]] = []

    def add(self, cust: np.ndarray, date: np.ndarray, amount: np.ndarray, merchant: np.ndarray, *,
            credit: bool | np.ndarray = False, ptype: str | np.ndarray | None = None, stream: str,
            qty: np.ndarray | None = None, qty_unit: str = "", fuel: np.ndarray | str = "",
            on_card: np.ndarray | None = None) -> None:
        n = len(cust)
        if n == 0:
            return
        self.parts.append({
            "cust": np.asarray(cust, dtype=np.int32),
            "date": np.asarray(date).astype("datetime64[D]"),
            "amount": np.round(np.asarray(amount, dtype=float), 2),
            "merchant_id": np.asarray(merchant, dtype=np.int32),
            "credit": np.broadcast_to(np.asarray(credit, dtype=bool), n).copy(),
            "ptype": np.broadcast_to(np.asarray(ptype if ptype is not None else "", dtype=object), n).copy(),
            "stream": np.full(n, stream, dtype=object),
            "qty": np.asarray(qty, dtype=float) if qty is not None else np.full(n, np.nan),
            "qty_unit": np.full(n, qty_unit, dtype=object),
            "fuel": np.broadcast_to(np.asarray(fuel, dtype=object), n).copy(),
            "on_card": np.asarray(on_card, dtype=bool) if on_card is not None else np.zeros(n, dtype=bool),
        })

    def frame(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame()
        keys = self.parts[0].keys()
        return pd.DataFrame({k: np.concatenate([p[k] for p in self.parts]) for k in keys})


class ChunkGenerator:
    def __init__(self, state: pd.DataFrame, merchants: Merchants, ref: Reference, cfg: dict,
                 rng: np.random.Generator) -> None:
        self.s = state.reset_index(drop=True)
        self.m = merchants
        self.ref = ref
        self.cfg = cfg
        self.rng = rng
        self.months = np.arange(np.datetime64(cfg["transactions"]["start"], "M"),
                                np.datetime64(cfg["transactions"]["end"], "M") + 1)
        self.month_start = self.months.astype("datetime64[D]")
        self.month_days = ((self.months + 1).astype("datetime64[D]") - self.month_start).astype(int)
        self.cal_month = (self.months.astype(int) % 12)  # 0 = January
        self.C, self.M = len(self.s), len(self.months)
        self.buf = Buffer()
        self.regions = self.s["region"].values.astype(object)
        self.la = self.s["la_code"].values.astype(object)
        self.mid = {k: i for k, i in zip(merchants.table["merchant_key"], merchants.table["merchant_id"])}
        self.cutoff = np.datetime64(cfg["labels"]["cutoff"])
        self.first_variant = np.array([v[0] for v in merchants.table["variants"].values], dtype=object)

    # ------------------------------------------------------------------ helpers

    def expand(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = np.maximum(counts, 0).astype(int).ravel()
        idx = np.repeat(np.arange(flat.size), flat)
        return (idx // self.M).astype(np.int32), (idx % self.M).astype(np.int32)

    def random_dates(self, mi: np.ndarray) -> np.ndarray:
        days = (self.rng.random(len(mi)) * self.month_days[mi]).astype(int)
        return self.month_start[mi] + days.astype("timedelta64[D]")

    def fixed_dates(self, ci: np.ndarray, mi: np.ndarray, day: np.ndarray) -> np.ndarray:
        d = np.minimum(day[ci], self.month_days[mi]) - 1
        return self.month_start[mi] + d.astype("timedelta64[D]")

    def mask_grid(self, mask: np.ndarray) -> np.ndarray:
        return np.repeat(mask[:, None], self.M, axis=1)

    def after(self, dates: np.ndarray) -> np.ndarray:
        """(C, M) boolean: month starts on/after each customer's date (False for NaT)."""
        d = np.asarray(dates).astype("datetime64[M]")
        valid = ~np.isnat(d)
        return valid[:, None] & (self.months[None, :] >= np.where(valid, d, self.months[0])[:, None])

    def before(self, dates: np.ndarray) -> np.ndarray:
        d = np.asarray(dates).astype("datetime64[M]")
        valid = ~np.isnat(d)
        return ~valid[:, None] | (self.months[None, :] < np.where(valid, d, self.months[-1] + 1)[:, None])

    def in_month(self, dates: np.ndarray) -> np.ndarray:
        """(C, M) boolean: the month that contains each customer's date."""
        d = np.asarray(dates).astype("datetime64[M]")
        return ~np.isnat(d)[:, None] & (self.months[None, :] == d[:, None])

    def fav(self, sub: str, k: int, local_share: float = 0.0, green: np.ndarray | None = None) -> np.ndarray:
        """k favourite merchants per customer for one sub-category: (C, k) merchant ids."""
        cols = [pick_merchants(self.m, sub, self.regions, self.la, self.rng, local_share=local_share,
                               green_affinity=green) for _ in range(k)]
        return np.column_stack(cols)

    def choose(self, fav: np.ndarray, ci: np.ndarray, p_fav: float, sub: str, local_share: float = 0.0) -> np.ndarray:
        k = fav.shape[1]
        pick = fav[ci, self.rng.integers(0, k, len(ci))]
        other = self.rng.random(len(ci)) > p_fav
        if other.any():
            pick[other] = pick_merchants(self.m, sub, self.regions[ci[other]], self.la[ci[other]], self.rng,
                                         local_share=local_share)
        return pick

    def key(self, name: str) -> int:
        return self.mid[name]

    # ------------------------------------------------------------------ streams

    def income(self) -> None:
        s, rng = self.s, self.rng
        retired_after = self.after(s["retirement_date"].values)
        source = s["income_source"].values
        net = s["monthly_net_income"].values
        pay_day = s["pay_day"].values
        is_salary = self.mask_grid(source == "salary") & ~retired_after
        ci, mi = self.expand(is_salary.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, pay_day), net[ci] * rng.normal(1.0, 0.02, len(ci)),
                     np.full(len(ci), self.key("salary")), credit=True, stream="salary")
        selfemp = self.mask_grid(source == "self_employed") & ~retired_after
        counts = selfemp * rng.integers(1, 5, (self.C, self.M))
        ci, mi = self.expand(counts)
        self.buf.add(ci, self.random_dates(mi), net[ci] * rng.uniform(0.25, 0.6, len(ci)),
                     np.full(len(ci), self.key("client_payment")), credit=True, stream="self_employed")
        pension = self.mask_grid(source == "pension") | retired_after
        ci, mi = self.expand(pension.astype(int))
        state_pension = np.minimum(net * 0.45, 997.0)
        self.buf.add(ci, self.fixed_dates(ci, mi, (pay_day - 18)), state_pension[ci],
                     np.full(len(ci), self.key("pension_state")), credit=True, stream="pension")
        private = pension & self.mask_grid(net > 1100)
        ci, mi = self.expand(private.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, pay_day), (net - state_pension)[ci].clip(50),
                     np.full(len(ci), self.key("pension_private")), credit=True, stream="pension")
        benefits = self.mask_grid(source == "benefits")
        ci, mi = self.expand(benefits.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, pay_day - 10), net[ci],
                     np.full(len(ci), self.key("benefits")), credit=True, stream="benefits")
        student = self.mask_grid(source == "student") & np.isin(self.cal_month, [0, 3, 8])[None, :]
        ci, mi = self.expand(student.astype(int))
        self.buf.add(ci, self.month_start[mi] + 5, rng.uniform(1500, 3600, len(ci)),
                     np.full(len(ci), self.key("student_loan")), credit=True, stream="student")
        kids = self.mask_grid(s["household_size"].values >= 3)
        ci, mi = self.expand(kids.astype(int) * (rng.random((self.C, self.M)) < 0.85))
        self.buf.add(ci, self.fixed_dates(ci, mi, np.full(self.C, 14)), np.full(len(ci), 104.0),
                     np.full(len(ci), self.key("child_benefit")), credit=True, stream="benefits")

    def housing(self) -> None:
        s, rng, m = self.s, self.rng, self.m
        day = rng.integers(1, 29, self.C)
        # Rent.
        priv = (s["tenure"].values == "private_rent") & (s["rent_monthly"].values > 0)
        social = (s["tenure"].values == "social_rent") & (s["rent_monthly"].values > 0)
        priv_ids = self._sub_filter("housing", "openrent|foxtons|countrywide|local::housing_rent")
        social_ids = self._sub_filter("housing", "clarion|lq|peabody|sanctuary|places_for_people|council_rent")
        landlord = np.where(priv, self._pick_from(priv_ids, self.C), self._pick_from(social_ids, self.C))
        for mask, ptype in ((priv, "SO"), (social, "DD")):
            ci, mi = self.expand(self.mask_grid(mask).astype(int))
            self.buf.add(ci, self.fixed_dates(ci, mi, day), s["rent_monthly"].values[ci], landlord[ci],
                         ptype=ptype, stream="rent")
        # Mortgage.
        mort = s["mortgage"].values & ~s["mortgage_paid_elsewhere"].values
        with_us = s["mortgage_with_us"].values
        lender_ids = self._sub_filter("housing", "nationwide|santander_mortgage|barclays_mortgage|natwest_mortgage|"
                                                 "hsbc_mortgage|coventry_bs|yorkshire_bs|skipton_bs|virgin_money|tsb_mortgage")
        lender = np.where(with_us, self.key("mortgage_internal"), self._pick_from(lender_ids, self.C))
        ci, mi = self.expand(self.mask_grid(mort).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day), s["mortgage_monthly"].values[ci], lender[ci],
                     ptype=np.where(with_us[ci], "TFR", "DD"), stream="mortgage")
        # Council tax: ten instalments April to January.
        ct = self.mask_grid(~s["bills_paid_elsewhere"].values) & np.isin(self.cal_month, [3, 4, 5, 6, 7, 8, 9, 10, 11, 0])[None, :]
        ci, mi = self.expand(ct.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, rng.integers(1, 16, self.C)), s["council_tax_monthly"].values[ci],
                     np.full(len(ci), self.key("council_tax")), stream="council_tax")
        # Water.
        water_merchant = pick_merchants(m, "water", self.regions, self.la, rng)
        wt = self.mask_grid(~s["bills_paid_elsewhere"].values)
        ci, mi = self.expand(wt.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, rng.integers(1, 29, self.C)), s["water_monthly"].values[ci],
                     water_merchant[ci], stream="water")

    def _sub_filter(self, sub: str, pattern: str) -> np.ndarray:
        t = self.m.table.loc[self.m.ids(sub)]
        return t.loc[t["merchant_key"].str.contains(pattern, regex=True), "merchant_id"].values

    def _pick_from(self, ids: np.ndarray, n: int) -> np.ndarray:
        return ids[self.rng.integers(0, len(ids), n)]

    def energy(self) -> None:
        s, rng = self.s, self.rng
        visible = ~s["energy_paid_elsewhere"].values
        supplier = s["supplier_key"].values
        billing = s["billing_type"].values
        elec_year = s["elec_kwh_year_true"].values.astype(float)
        gas_year = s["gas_kwh_year_true"].values.astype(float)
        # GP02 heat-pump / insulation and GP03 solar take-ups change demand after installation.
        hp_after = self.after(s["gp02_install_date"].values)
        hp = s["gp02_measure"].values == "heat_pump"
        gas_factor = np.where(hp_after & hp[:, None], 0.0, np.where(hp_after, 0.80, 1.0))
        hp_extra = np.where(hp_after & hp[:, None], (gas_year * 0.84 / 2.93)[:, None], 0.0)
        pv_after = self.after(s["gp03_install_date"].values)
        pv_cut = np.where(pv_after, (s["gp03_kwp"].values * s["pv_kwh_per_kwp_region"].values * 0.9 * 0.45)[:, None], 0.0)
        ev_after = self.after(s["ev_from_date"].values)
        ev_extra = np.where(ev_after & ~s["has_ev_at_start"].values[:, None], s["ev_home_kwh_if_ev"].values[:, None], 0.0)

        elec_m = np.maximum((elec_year[:, None] + hp_extra + ev_extra - pv_cut) * ELEC_MONTH[self.cal_month][None, :], 20.0)
        gas_m = gas_year[:, None] * gas_factor * GAS_MONTH[self.cal_month][None, :]
        rates = self.ref.cap_rates(self.month_start)
        days = self.month_days[None, :]
        elec_cost = elec_m * rates["elec_unit"][None, :] / 100 + days * rates["elec_sc"][None, :] / 100
        gas_cost = np.where(gas_year[:, None] > 0, gas_m * rates["gas_unit"][None, :] / 100 + days * rates["gas_sc"][None, :] / 100, 0.0)
        # Direct debits are level payments re-set each quarter from the annual estimate.
        elec_annual = (elec_year[:, None] + hp_extra + ev_extra - pv_cut).clip(300) * rates["elec_unit"][None, :] / 100 + 365 * rates["elec_sc"][None, :] / 100
        gas_annual = np.where(gas_year[:, None] > 0, gas_year[:, None] * gas_factor * rates["gas_unit"][None, :] / 100 + 365 * rates["gas_sc"][None, :] / 100, 0.0)
        bias = rng.normal(0.04, 0.07, self.C)[:, None]
        elec_dd = np.round(elec_annual / 12 * (1 + bias))
        gas_dd = np.round(gas_annual / 12 * (1 + bias))
        day = rng.integers(1, 29, self.C)
        tag = rng.random(self.C) < self.cfg["observability"]["dd_reference_has_fuel_tag"]

        def mids(suffix: str) -> np.ndarray:
            return np.array([self.mid.get(f"{k}::{suffix}", self.mid[f"{k}::dual"]) for k in supplier])

        dual_id, etag, gtag, eplain, gplain = mids("dual"), mids("elec_tag"), mids("gas_tag"), mids("elec_plain"), mids("gas_plain")
        eid = np.where(tag, etag, eplain)
        gid = np.where(tag, gtag, gplain)
        dual = visible & (billing == "dual")
        ci, mi = self.expand(self.mask_grid(dual).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day), (elec_dd + gas_dd)[ci, mi], dual_id[ci], ptype="DD", stream="energy_dual")
        sep = visible & (billing == "separate")
        ci, mi = self.expand(self.mask_grid(sep).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day), elec_dd[ci, mi], eid[ci], ptype="DD", stream="energy_elec")
        self.buf.add(ci, self.fixed_dates(ci, mi, day + 2), gas_dd[ci, mi], gid[ci], ptype="DD", stream="energy_gas")
        eo = visible & (billing == "electric_only")
        ci, mi = self.expand(self.mask_grid(eo).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day), elec_dd[ci, mi], eid[ci], ptype="DD", stream="energy_elec")
        # Prepayment top-ups follow actual (seasonal) cost in round amounts.
        pre = visible & (billing == "prepayment")
        for fuel, cost, sub_ids in (("elec", elec_cost, (mids("prepay_elec"), "elec")), ("gas", gas_cost, (mids("prepay_gas"), "gas"))):
            amount_step = rng.choice([10.0, 15.0, 20.0, 25.0, 30.0], (self.C, self.M))
            counts = np.where(self.mask_grid(pre) & (cost > 0), np.maximum(np.round(cost / amount_step), 1), 0)
            ci, mi = self.expand(counts)
            network = self._sub_filter("electricity_prepayment" if fuel == "elec" else "gas_prepayment", "prepay_network")
            use_network = rng.random(len(ci)) < 0.55
            merchant = np.where(use_network, network[rng.integers(0, len(network), len(ci))], sub_ids[0][ci])
            self.buf.add(ci, self.random_dates(mi), amount_step[ci, mi], merchant, ptype="DEB", stream=f"prepay_{fuel}")
        # Annual reconciliation refunds (credits keep the supplier's category, rule B10).
        refund = visible & np.isin(billing, ["dual", "separate", "electric_only"]) & (bias[:, 0] > 0.02)
        refund_month = rng.integers(0, self.M, self.C)
        grid = self.mask_grid(refund) & (np.arange(self.M)[None, :] == refund_month[:, None])
        ci, mi = self.expand(grid.astype(int))
        refund_id = np.where(billing == "dual", dual_id, eid)
        self.buf.add(ci, self.random_dates(mi), (bias[ci, 0] * (elec_dd + gas_dd)[ci, mi] * 12 * rng.uniform(0.4, 0.9, len(ci))).clip(5),
                     refund_id[ci], credit=True, ptype="BGC", stream="energy_refund")
        # Smart Export Guarantee payments for solar owners (quarterly credits).
        seg = visible & (s["has_solar"].values | ~np.isnat(s["gp03_install_date"].values.astype("datetime64[D]")))
        seg = seg & (rng.random(self.C) < 0.7)
        export_q = (s["pv_export_kwh_year_true"].values + s["gp03_kwp"].values * s["pv_kwh_per_kwp_region"].values * 0.5) / 4
        grid = self.mask_grid(seg) & np.isin(self.cal_month, [0, 3, 6, 9])[None, :]
        grid = grid & (self.mask_grid(s["has_solar"].values) | pv_after)
        ci, mi = self.expand(grid.astype(int))
        self.buf.add(ci, self.random_dates(mi), export_q[ci] * 0.12 * rng.uniform(0.8, 1.2, len(ci)), mids("seg")[ci],
                     credit=True, ptype="BGC", stream="seg_export")
        # Heating oil and LPG deliveries (autumn and winter).
        for fuel, litres_col, price_key, lpg in (("oil", "oil_litres_year_true", "oil_price_gbp_per_litre", False),
                                                 ("lpg", "lpg_litres_year_true", "lpg_price_gbp_per_litre", True)):
            litres = s[litres_col].values.astype(float)
            has = visible & (litres > 0)
            n_del = rng.integers(2, 4, self.C)
            month_ok = np.isin(self.cal_month, [9, 10, 11, 0, 1])[None, :]
            p = np.where(month_ok, n_del[:, None] / 5.0, 0.03)
            grid = self.mask_grid(has) & (rng.random((self.C, self.M)) < p)
            ci, mi = self.expand(grid.astype(int))
            per = litres[ci] / n_del[ci]
            t = self.m.table
            pool = t[(t["sub_category"] == "heating_oil_lpg") & (t["lpg"] == lpg)]["merchant_id"].values
            merchant = pool[self.rng.integers(0, len(pool), len(ci))]
            self.buf.add(ci, self.random_dates(mi), per * self.cfg["homes"][price_key] * rng.uniform(0.9, 1.1, len(ci)),
                         merchant, ptype="DEB", stream=f"heating_{fuel}", qty=per, qty_unit="litre", fuel=fuel)

    def vehicles(self) -> None:
        s, rng = self.s, self.rng
        vcfg = self.cfg["vehicles"]
        ev_after = self.after(s["ev_from_date"].values)
        car_active = self.mask_grid(s["has_car"].values) | self.after(s["new_car_date"].values)
        ice_fuel_start = np.isin(s["car_fuel"].values, ["petrol", "diesel", "hev", "phev"])
        # Monthly ICE litres: primary car until it is replaced by an EV, plus extra cars.
        km_m = s["annual_km_true"].values[:, None] * KM_MONTH[self.cal_month][None, :]
        kml = s["km_per_litre_true"].fillna(13.0).values[:, None]
        ice_share = (1 - s["ev_km_share"].values)[:, None]
        primary_ice = car_active & self.mask_grid(ice_fuel_start) & ~ev_after
        litres = np.where(primary_ice, km_m * ice_share / kml, 0.0)
        new_ice = self.after(s["new_car_date"].values) & self.mask_grid(np.isin(s["new_car_fuel"].values, ["petrol", "hev"])) & ~ev_after
        litres = np.where(new_ice, km_m / np.where(s["new_car_fuel"].values == "hev", 16.0, 12.85)[:, None], litres)
        extra = s["extra_km_true"].values[:, None] * KM_MONTH[self.cal_month][None, :] / s["extra_km_per_litre_true"].fillna(13.0).values[:, None]
        litres_total = litres + np.where(s["extra_cars"].values[:, None] > 0, extra, 0.0)
        fuel_kind = np.where(s["car_fuel"].values == "diesel", "diesel", "petrol")

        visible = ~s["fuel_on_external_card"].values
        fills = np.where(litres_total > 1, np.maximum(rng.poisson(litres_total / 38.0), 1), 0)
        ci, mi = self.expand(fills * self.mask_grid(visible))
        per_fill = litres_total[ci, mi] / np.maximum(fills[ci, mi], 1) * np.exp(rng.normal(0, 0.15, len(ci)))
        dates = self.random_dates(mi)
        cash = rng.random(len(ci)) < s["cash_fuel_share"].values[ci]
        fav = self.fav("road_fuel", 2, local_share=0.06)
        merchant = self.choose(fav, ci, 0.8, "road_fuel", local_share=0.06)
        keys = self.m.table["merchant_key"].values[merchant]
        premium = np.where(np.char.startswith(keys.astype(str), "moto") | np.char.startswith(keys.astype(str), "welcome")
                           | np.char.startswith(keys.astype(str), "roadchef"), 18.0,
                           np.where(np.isin(keys, ["tesco_pfs", "asda_petrol", "sainsburys_petrol", "morrisons_fuel"]), -2.0, 2.0))
        price = (self.ref.fuel_price_ppl(dates, fuel_kind[ci]) + premium + rng.normal(0, 1.5, len(ci))) / 100.0
        round_fill = rng.random(len(ci)) < vcfg["round_amount_fill_share"]
        amount = np.where(round_fill, np.maximum(np.round(per_fill * price / 10) * 10, 10.0), per_fill * price)
        litres_bought = amount / price
        keep = ~cash
        self.buf.add(ci[keep], dates[keep], amount[keep], merchant[keep], ptype="DEB", stream="fuel",
                     qty=litres_bought[keep], qty_unit="litre", fuel=fuel_kind[ci][keep], on_card=self._on_card(ci[keep]))
        # Forecourt shop purchases (rule B3: Other).
        shop = keep & (rng.random(len(ci)) < vcfg["forecourt_shop_rate_per_fill"])
        shop_merchant = np.array([self.mid.get(f"{k}::shop", self.mid["shell::shop"]) for k in keys[shop]], dtype=int)
        self.buf.add(ci[shop], dates[shop], np.round(rng.gamma(2.0, 2.4, shop.sum()) + 0.9, 2), shop_merchant,
                     ptype="DEB", stream="forecourt_shop", on_card=self._on_card(ci[shop]))

        # Public EV charging (BEV and PHEV drivers).
        ev_start = np.isin(s["car_fuel"].values, ["bev", "phev"])
        ev_on = (self.mask_grid(ev_start & s["has_car"].values) & self.before(s["new_car_date"].values)) | ev_after
        kwh_km = s["ev_kwh_per_km_true"].fillna(0.205).values[:, None]
        share = np.where(ev_after & ~self.mask_grid(ev_start), 1.0, s["ev_km_share"].values[:, None])
        pub_kwh = np.where(ev_on, km_m * share * kwh_km * s["public_charging_share"].values[:, None], 0.0)
        sessions = np.where(pub_kwh > 4, np.maximum(rng.poisson(pub_kwh / 22.0), 1), 0)
        ci, mi = self.expand(sessions)
        kwh = pub_kwh[ci, mi] / np.maximum(sessions[ci, mi], 1) * np.exp(rng.normal(0, 0.3, len(ci)))
        rapid = rng.random(len(ci)) < 0.6
        price = np.where(rapid, vcfg["public_charge_price_gbp_per_kwh"]["rapid"], vcfg["public_charge_price_gbp_per_kwh"]["slow"])
        merchant = pick_merchants(self.m, "ev_public_charging", self.regions[ci], self.la[ci], rng)
        self.buf.add(ci, self.random_dates(mi), kwh * price, merchant, ptype="DEB", stream="ev_charging",
                     qty=kwh, qty_unit="kWh", fuel="electricity", on_card=self._on_card(ci))

        # Car finance: external direct debit or internal motor loan, until it ends or the car is replaced.
        fin = self.mask_grid(s["motor_finance"].values) & self.before(s["motor_finance_end"].values) & self.before(s["new_car_date"].values) & self.before(s["ev_from_date"].values)
        lender = pick_merchants(self.m, "vehicle_finance", self.regions, self.la, rng)
        lender = np.where(s["motor_finance_with_us"].values, self.key("motor_loan_internal"), lender)
        ci, mi = self.expand(fin.astype(int))
        day = rng.integers(1, 29, self.C)
        self.buf.add(ci, self.fixed_dates(ci, mi, day), s["motor_finance_monthly"].values[ci], lender[ci],
                     ptype=np.where(s["motor_finance_with_us"].values[ci], "TFR", "DD"), stream="car_finance")
        # Replacement cars bought outside our EV product: a dealer deposit, then new external finance.
        new_car = ~np.isnat(s["new_car_date"].values.astype("datetime64[D]"))
        ci = np.where(new_car)[0]
        dealer = pick_merchants(self.m, "vehicle_purchase", self.regions[ci], self.la[ci], rng, local_share=0.3)
        self.buf.add(ci, s["new_car_date"].values.astype("datetime64[D]")[ci], rng.uniform(800, 7000, len(ci)), dealer,
                     ptype="FPO", stream="car_purchase")
        financed_new = new_car & s["new_car_financed_elsewhere"].values
        new_fin = self.after(s["new_car_date"].values.astype("datetime64[M]") + 1) & self.mask_grid(financed_new)
        new_lender = pick_merchants(self.m, "vehicle_finance", self.regions, self.la, rng)
        ci, mi = self.expand(new_fin.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day), rng.uniform(220, 520, self.C)[ci], new_lender[ci], ptype="DD",
                     stream="car_finance")
        # Insurance, tax, breakdown, servicing, parking and clean-air charges for car owners.
        owner = car_active
        ins = pick_merchants(self.m, "motor_insurance", self.regions, self.la, rng)
        ci, mi = self.expand(owner.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, (day + 7) % 28 + 1), rng.uniform(38, 120, self.C)[ci], ins[ci],
                     ptype="DD", stream="motor_insurance")
        ved_monthly = self.mask_grid(rng.random(self.C) < 0.7) & owner
        ci, mi = self.expand(ved_monthly.astype(int))
        ved = np.where(s["car_fuel"].values == "diesel", 21.0, 17.06)
        self.buf.add(ci, self.fixed_dates(ci, mi, np.full(self.C, 1)), ved[ci], np.full(len(ci), self.key("dvla")),
                     ptype="DD", stream="vehicle_tax")
        breakdown = pick_merchants(self.m, "breakdown_cover", self.regions, self.la, rng)
        bd = owner & self.mask_grid(rng.random(self.C) < 0.42)
        ci, mi = self.expand(bd.astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day), rng.uniform(7.5, 15.0, self.C)[ci], breakdown[ci], ptype="DD", stream="breakdown")
        service = owner & (rng.random((self.C, self.M)) < 0.11)
        ci, mi = self.expand(service.astype(int))
        merchant = pick_merchants(self.m, "maintenance", self.regions[ci], self.la[ci], rng, local_share=0.45)
        self.buf.add(ci, self.random_dates(mi), np.round(rng.lognormal(5.0, 0.7, len(ci)), 2), merchant, ptype="DEB",
                     stream="car_maintenance", on_card=self._on_card(ci))
        urban = np.isin(self.regions, ["London", "West Midlands", "North West", "South East"])
        park_rate = np.where(urban, 1.5, 0.6)[:, None] * owner
        ci, mi = self.expand(rng.poisson(park_rate))
        merchant = pick_merchants(self.m, "parking_charges", self.regions[ci], self.la[ci], rng, local_share=0.2,
                                  keys_filter=None)
        self.buf.add(ci, self.random_dates(mi), np.round(rng.uniform(1.4, 14.0, len(ci)), 2), merchant, ptype="DEB",
                     stream="parking", on_card=self._on_card(ci))
        t = self.m.table
        ulez_ids = t[t["merchant_key"].isin(["tfl_ulez", "tfl_congestion"])]["merchant_id"].values
        caz_ids = t[t["merchant_key"] == "gov_caz"]["merchant_id"].values
        congestion_id = self.mid["tfl_congestion"]
        for payer, ids, fee in ((s["ulez_payer"].values, ulez_ids, 12.50), (s["caz_payer"].values, caz_ids, 8.0)):
            grid = self.mask_grid(payer) & owner & ~ev_after
            ci, mi = self.expand(grid * rng.integers(1, 7, (self.C, self.M)))
            dates = self.random_dates(mi)
            chosen = ids[rng.integers(0, len(ids), len(ci))]
            # Congestion charge: GBP 15 a day, GBP 18 from 2 January 2026; ULEZ GBP 12.50; clean-air zones GBP 8.
            amount = np.where(chosen == congestion_id, np.where(dates >= np.datetime64("2026-01-02"), 18.0, 15.0), fee)
            self.buf.add(ci, dates, amount, chosen, ptype="DEB", stream="clean_air_charge", on_card=self._on_card(ci))

    def _on_card(self, ci: np.ndarray) -> np.ndarray:
        return self.s["has_credit_card"].values[ci] & (self.rng.random(len(ci)) < 0.35)

    def public_transport(self) -> None:
        s, rng = self.s, self.rng
        retired_after = self.after(s["retirement_date"].values)
        mode = s["commute_mode"].values
        london = self.regions == "London"
        commute_days = np.where(np.isin(mode, ["train", "metro_tram", "bus"]), rng.integers(6, 16, self.C), 0)
        active = self.mask_grid(commute_days > 0) & ~retired_after
        # Rail: a monthly season ticket or per-day tickets.
        rail = active & self.mask_grid(mode == "train")
        season = rng.random(self.C) < 0.35
        ci, mi = self.expand(np.where(rail & season[:, None], 1, 0))
        rail_price = np.where(np.isin(self.regions, ["London", "South East", "East"]), rng.uniform(220, 520, self.C), rng.uniform(110, 280, self.C))
        merchant_rail = pick_merchants(self.m, "rail", self.regions, self.la, rng)
        self.buf.add(ci, self.month_start[mi] + 1, rail_price[ci], merchant_rail[ci], ptype="DEB", stream="rail",
                     on_card=self._on_card(ci))
        ci, mi = self.expand(np.where(rail & ~season[:, None], commute_days[:, None], 0))
        day_price = np.where(np.isin(self.regions[ci], ["London", "South East", "East"]), rng.uniform(14, 38, len(ci)), rng.uniform(7, 19, len(ci)))
        self.buf.add(ci, self.random_dates(mi), day_price, merchant_rail[ci], ptype="DEB", stream="rail",
                     on_card=self._on_card(ci))
        # Metro and tram: London pay-as-you-go daily caps, other cities day tickets.
        metro = active & self.mask_grid(mode == "metro_tram")
        ci, mi = self.expand(np.where(metro, commute_days[:, None], 0))
        merchant = pick_merchants(self.m, "metro_tram", self.regions[ci], self.la[ci], rng)
        price = np.where(london[ci], rng.uniform(5.25, 8.90, len(ci)), rng.uniform(4.2, 6.6, len(ci)))
        self.buf.add(ci, self.random_dates(mi), price, merchant, ptype="DEB", stream="metro_tram", on_card=self._on_card(ci))
        # Bus commuters and car-free households.
        bus = (active & self.mask_grid(mode == "bus"))
        car_free = self.mask_grid(s["n_cars"].values == 0)
        trips = np.where(bus, commute_days[:, None], 0) + rng.poisson(np.where(car_free, 2.5, 0.25))
        ci, mi = self.expand(trips)
        merchant = pick_merchants(self.m, "bus_coach", self.regions[ci], self.la[ci], rng, local_share=0.05)
        keys = self.m.table["merchant_key"].values[merchant]
        names = self.first_variant[merchant]
        coach = np.isin(keys, ["national_express", "megabus"]) | (np.char.find(names.astype(str), "COACHES") >= 0)
        fare = np.where(coach, rng.uniform(9, 48, len(ci)), np.where(rng.random(len(ci)) < 0.8, 3.00, rng.uniform(4.5, 7.5, len(ci))))
        self.buf.add(ci, self.random_dates(mi), fare, merchant, ptype="DEB", stream="bus", on_card=self._on_card(ci))
        # Leisure rail and micromobility.
        leisure = rng.poisson(np.where(s["z_green"].values > 0.5, 0.6, 0.25)[:, None] * np.ones((1, self.M)))
        ci, mi = self.expand(leisure)
        merchant = pick_merchants(self.m, "rail", self.regions[ci], self.la[ci], rng)
        self.buf.add(ci, self.random_dates(mi), rng.uniform(9, 95, len(ci)), merchant, ptype="DEB", stream="rail_leisure",
                     on_card=self._on_card(ci))
        micro_rate = np.where(s["cyclist"].values, 2.0, 0.08) * np.where(np.isin(self.regions, ["London", "North West", "West Midlands"]), 1.4, 0.6)
        ci, mi = self.expand(rng.poisson(micro_rate[:, None] * np.ones((1, self.M))))
        merchant = pick_merchants(self.m, "micromobility", self.regions[ci], self.la[ci], rng)
        self.buf.add(ci, self.random_dates(mi), rng.uniform(1.6, 6.5, len(ci)), merchant, ptype="DEB", stream="micromobility",
                     on_card=self._on_card(ci))

    def everyday(self) -> None:
        s, rng, ref = self.s, self.rng, self.ref
        tcfg = self.cfg["transactions"]
        decile = s["income_decile"].values - 1
        size_after_baby = s["household_size"].values[:, None] + self.after(s["baby_date"].values)
        size_factor = (size_after_baby / 2.4) ** tcfg["household_size_elasticity"]
        noise = np.exp(rng.normal(0, 0.35, self.C))[:, None]
        to_month = 52.0 / 12.0 * tcfg["other_spend_capture_share"]
        local = tcfg["local_business_share"]

        def budget(codes: list[str]) -> np.ndarray:
            weekly = sum(ref.spend_weekly(c) for c in codes)
            return weekly[decile][:, None] * to_month * size_factor * noise

        streams = [
            ("groceries", ["1", "2"], 46.0, 3, 0.80, local * 0.5),
            ("eating_out", ["11"], 24.0, 4, 0.55, local * 1.6),
            ("retail", ["3", "5", "9", "12"], 48.0, 3, 0.45, local * 0.6),
        ]
        green_aff = np.exp(0.9 * s["z_green"].values)
        for sub, codes, avg, k, p_fav, loc_share in streams:
            b = budget(codes)
            if sub == "retail":
                b = b * 0.45  # part of these COICOP lines is subscriptions, travel and services, generated below
            counts = rng.poisson(np.maximum(b / avg, 0.2))
            ci, mi = self.expand(counts)
            amount = b[ci, mi] / np.maximum(counts[ci, mi], 1) * np.exp(rng.normal(0, 0.45, len(ci)))
            fav = self.fav(sub, k, local_share=loc_share)
            merchant = self.choose(fav, ci, p_fav, sub, local_share=loc_share)
            self.buf.add(ci, self.random_dates(mi), np.maximum(amount, 0.8), merchant, ptype="DEB", stream=sub,
                         on_card=self._on_card(ci))
        # Health and beauty, charity and second-hand (green-leaning), taxis, travel.
        for sub, rate, lo, hi, loc_share, green in (("health_beauty", 1.0, 3, 60, 0.35, False),
                                                    ("charity_second_hand", 0.5, 3, 45, 0.25, True),
                                                    ("taxi_ridehail", 0.6, 7, 34, 0.35, False),
                                                    ("travel", 0.18, 40, 650, 0.0, False)):
            lam = np.full((self.C, self.M), rate)
            if green:
                lam = lam * green_aff[:, None]
            if sub == "taxi_ridehail":
                lam = lam * np.where(s["n_cars"].values == 0, 2.5, 0.7)[:, None] * np.where(self.regions == "London", 1.8, 1.0)[:, None]
            if sub == "travel":
                lam = lam * (0.4 + decile / 6.0)[:, None]
            ci, mi = self.expand(rng.poisson(lam))
            merchant = pick_merchants(self.m, sub, self.regions[ci], self.la[ci], rng, local_share=loc_share,
                                      green_affinity=green_aff[ci] if green else None)
            self.buf.add(ci, self.random_dates(mi), rng.uniform(lo, hi, len(ci)) * np.where(sub == "travel", 1 + decile[ci] / 10, 1.0),
                         merchant, ptype="DEB", stream=sub, on_card=self._on_card(ci))
        # Subscriptions: fixed monthly card payments.
        n_subs = rng.integers(0, 5, self.C)
        for j in range(4):
            has = n_subs > j
            merchant = pick_merchants(self.m, "subscriptions", self.regions, self.la, rng)
            price = rng.choice([4.99, 5.99, 7.99, 8.99, 10.99, 11.99, 12.99, 17.99, 24.99], self.C)
            day = rng.integers(1, 29, self.C)
            ci, mi = self.expand(self.mask_grid(has).astype(int))
            self.buf.add(ci, self.fixed_dates(ci, mi, day), price[ci], merchant[ci], ptype="DEB", stream="subscriptions",
                         on_card=self._on_card(ci))
        # Telecoms: broadband and mobile direct debits.
        for j, (lo, hi) in enumerate(((22, 65), (8, 38))):
            has = rng.random(self.C) < (0.85 if j == 0 else 0.92)
            merchant = pick_merchants(self.m, "telecoms", self.regions, self.la, rng)
            ci, mi = self.expand(self.mask_grid(has).astype(int))
            self.buf.add(ci, self.fixed_dates(ci, mi, rng.integers(1, 29, self.C)), rng.uniform(lo, hi, self.C)[ci],
                         merchant[ci], ptype="DD", stream="telecoms")
        # Home and life insurance.
        has = rng.random(self.C) < np.where(np.isin(s["tenure"].values, ["owned_outright", "mortgage"]), 0.85, 0.35)
        merchant = pick_merchants(self.m, "insurance_other", self.regions, self.la, rng)
        ci, mi = self.expand(self.mask_grid(has).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, rng.integers(1, 29, self.C)), rng.uniform(12, 48, self.C)[ci],
                     merchant[ci], ptype="DD", stream="insurance_other")
        # Cash machine withdrawals: more for older, less digital customers and cash fuel buyers.
        lam = 0.6 + 1.2 * (s["z_digital"].values < -0.8) + 1.5 * (s["cash_fuel_share"].values > 0)
        ci, mi = self.expand(rng.poisson(lam[:, None] * np.ones((1, self.M))))
        self.buf.add(ci, self.random_dates(mi), rng.choice([10, 20, 30, 40, 50, 60, 80, 100], len(ci)).astype(float),
                     np.full(len(ci), self.key("atm")), ptype="CPT", stream="cash")
        # Payments to and from people.
        ci, mi = self.expand(rng.poisson(np.full((self.C, self.M), 0.9)))
        self.buf.add(ci, self.random_dates(mi), np.round(rng.lognormal(3.4, 0.9, len(ci)), 2), np.full(len(ci), self.key("p2p_out")),
                     ptype="FPO", stream="p2p")
        ci, mi = self.expand(rng.poisson(np.full((self.C, self.M), 0.5)))
        self.buf.add(ci, self.random_dates(mi), np.round(rng.lognormal(3.2, 0.9, len(ci)), 2), np.full(len(ci), self.key("p2p_in")),
                     credit=True, ptype="FPI", stream="p2p")

    def home_and_green(self) -> None:
        s, rng = self.s, self.rng
        owner = np.isin(s["tenure"].values, ["owned_outright", "mortgage"])
        moved = self.after(s["move_date"].values) & self.before(s["move_date"].values.astype("datetime64[D]") + np.timedelta64(62, "D"))
        diy_rate = s["diy_intensity"].values[:, None] * 0.5 * (1 + 3.0 * moved)
        ci, mi = self.expand(rng.poisson(diy_rate))
        merchant = pick_merchants(self.m, "diy_materials", self.regions[ci], self.la[ci], rng, local_share=0.12)
        self.buf.add(ci, self.random_dates(mi), np.round(rng.lognormal(3.3, 0.9, len(ci)), 2), merchant, ptype="DEB",
                     stream="diy", on_card=self._on_card(ci))
        trades_rate = np.where(owner, 0.07, 0.01)[:, None] * (1 + 4.0 * moved)
        ci, mi = self.expand(rng.poisson(trades_rate))
        merchant = pick_merchants(self.m, "trades", self.regions[ci], self.la[ci], rng, local_share=0.9)
        self.buf.add(ci, self.random_dates(mi), np.round(rng.lognormal(6.0, 0.9, len(ci)), 2), merchant, ptype="FPO", stream="trades")
        appl_rate = np.full((self.C, self.M), 0.03) * (1 + 3 * moved)
        ci, mi = self.expand(rng.poisson(appl_rate))
        merchant = pick_merchants(self.m, "appliances", self.regions[ci], self.la[ci], rng, local_share=0.1)
        self.buf.add(ci, self.random_dates(mi), rng.uniform(180, 950, len(ci)), merchant, ptype="DEB", stream="appliances",
                     on_card=self._on_card(ci))
        ci, mi = self.expand(self.in_month(s["move_date"].values).astype(int))
        self.buf.add(ci, self.random_dates(mi), rng.uniform(350, 1600, len(ci)), np.full(len(ci), self.key("removals")),
                     ptype="FPO", stream="removals")
        # Pre-cutoff green purchase intent shows as small spend at bike and appliance shops.
        t = self.m.table
        intent_ids = t[t["intent"] == "green_purchase"]["merchant_id"].values
        lam = np.where(s["green_purchase_intent"].values, 0.35, 0.02)[:, None] * np.ones((1, self.M))
        ci, mi = self.expand(rng.poisson(lam))
        self.buf.add(ci, self.random_dates(mi), rng.uniform(15, 260, len(ci)), intent_ids[rng.integers(0, len(intent_ids), len(ci))],
                     ptype="DEB", stream="green_intent", on_card=self._on_card(ci))

    def finance(self) -> None:
        s, rng = self.s, self.rng
        day = rng.integers(1, 29, self.C)
        rows = (
            ("has_savings", "savings_internal", lambda: rng.uniform(25, 450, self.C), 0.9, "TFR", "savings"),
            ("has_credit_card", "card_repayment_internal", lambda: (s["card_balance"].values * rng.uniform(0.05, 0.6, self.C)).clip(25), 1.0, "TFR", "card_repayment"),
            ("has_personal_loan", "personal_loan_internal", lambda: s["personal_loan_monthly"].values, 1.0, "TFR", "loan_repayment"),
        )
        for flag, key, amount_fn, prob, ptype, stream in rows:
            has = s[flag].values & (rng.random(self.C) < prob)
            amount = amount_fn()
            ci, mi = self.expand(self.mask_grid(has).astype(int))
            self.buf.add(ci, self.fixed_dates(ci, mi, day), amount[ci], np.full(len(ci), self.key(key)), ptype=ptype, stream=stream)
        ext = s["external_card_repayment"].values > 0
        issuer = pick_merchants(self.m, "finance", self.regions, self.la, rng)
        ci, mi = self.expand(self.mask_grid(ext).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day + 3), s["external_card_repayment"].values[ci], issuer[ci], ptype="DD", stream="external_credit")
        # Investing: with us by internal transfer, elsewhere to a platform.
        reg = s["regular_investor"].values
        platform = pick_merchants(self.m, "investments", self.regions, self.la, rng)
        target = np.where(s["investments_with_us"].values, self.key("investment_internal"), platform)
        ci, mi = self.expand(self.mask_grid(reg).astype(int))
        self.buf.add(ci, self.fixed_dates(ci, mi, day + 1), s["monthly_investment"].values[ci], target[ci],
                     ptype=np.where(s["investments_with_us"].values[ci], "TFR", "FPO"), stream="investing")
        lump = self.mask_grid(s["investor"].values) & (rng.random((self.C, self.M)) < 0.04)
        ci, mi = self.expand(lump.astype(int))
        self.buf.add(ci, self.random_dates(mi), rng.uniform(500, 8000, len(ci)), target[ci],
                     ptype=np.where(s["investments_with_us"].values[ci], "TFR", "FPO"), stream="investing")
        # Signs of financial difficulty.
        od = self.mask_grid(s["overdraft_heavy"].values) & (rng.random((self.C, self.M)) < 0.85)
        ci, mi = self.expand(od.astype(int))
        self.buf.add(ci, self.month_start[mi] + 27, rng.uniform(6, 45, len(ci)), np.full(len(ci), self.key("overdraft_interest")),
                     ptype="CHG", stream="overdraft")
        arrears_recent = self.mask_grid(s["in_arrears"].values) & (np.arange(self.M)[None, :] >= self.M - 12) & (rng.random((self.C, self.M)) < 0.35)
        ci, mi = self.expand(arrears_recent.astype(int))
        self.buf.add(ci, self.random_dates(mi), rng.uniform(12, 25, len(ci)), np.full(len(ci), self.key("unpaid_dd")),
                     ptype="CHG", stream="unpaid_dd")

    def take_up_effects(self) -> None:
        """Payments created by green products taken up in the label window (post-cutoff only)."""
        s, rng = self.s, self.rng
        specs = (
            ("gp01_date", "ev_loan_internal", lambda: s["gp01_monthly"].values, "vehicle_purchase", lambda n: rng.uniform(1500, 6000, n)),
            ("gp02_date", "green_home_internal", lambda: s["gp02_monthly"].values, "energy_installers", lambda n: rng.uniform(2500, 14000, n)),
            ("gp03_date", "solar_loan_internal", lambda: s["gp03_monthly"].values, "energy_installers", lambda n: rng.uniform(900, 2500, n)),
            ("gp06_date", "green_loan_internal", lambda: s["gp06_monthly"].values, None, None),
        )
        for date_col, loan_key, monthly_fn, purchase_sub, purchase_fn in specs:
            dates = s[date_col].values.astype("datetime64[D]")
            has = ~np.isnat(dates)
            if not has.any():
                continue
            start = self.after(dates.astype("datetime64[M]") + 1)
            ci, mi = self.expand((start & self.mask_grid(has)).astype(int))
            self.buf.add(ci, self.fixed_dates(ci, mi, np.full(self.C, 12)), monthly_fn()[ci], np.full(len(ci), self.key(loan_key)),
                         ptype="TFR", stream=f"{date_col[:4]}_repayment")
            if purchase_sub:
                ci = np.where(has)[0]
                merchant = pick_merchants(self.m, purchase_sub, self.regions[ci], self.la[ci], rng, local_share=0.3)
                self.buf.add(ci, dates[ci] + 3, purchase_fn(len(ci)), merchant, ptype="FPO", stream=f"{date_col[:4]}_purchase")
        gp06 = s["gp06_date"].values.astype("datetime64[D]")
        ci = np.where(~np.isnat(gp06))[0]
        if len(ci):
            t = self.m.table
            pool = t[(t["intent"] == "green_purchase") | (t["sub_category"].isin(["appliances", "energy_installers"]) & (t["local_la"] == ""))]["merchant_id"].values
            self.buf.add(ci, gp06[ci] + 2, rng.uniform(600, 3500, len(ci)), pool[rng.integers(0, len(pool), len(ci))],
                         ptype="DEB", stream="gp06_purchase")
        gp05 = s["gp05_date"].values.astype("datetime64[D]")
        ci = np.where(~np.isnat(gp05))[0]
        self.buf.add(ci, gp05[ci], rng.uniform(1000, 15000, len(ci)), np.full(len(ci), self.key("investment_internal")),
                     ptype="TFR", stream="gp05_switch")

    # ------------------------------------------------------------------ assemble

    def run(self) -> pd.DataFrame:
        self.income()
        self.housing()
        self.energy()
        self.vehicles()
        self.public_transport()
        self.everyday()
        self.home_and_green()
        self.finance()
        self.take_up_effects()
        frame = self.buf.frame()
        end = np.datetime64(self.cfg["transactions"]["end"]) + 1
        frame = frame[(frame["date"] >= self.month_start[0]) & (frame["date"] < end)]
        return frame.sort_values(["cust", "date"], kind="stable").reset_index(drop=True)
