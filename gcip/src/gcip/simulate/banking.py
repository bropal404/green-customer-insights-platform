"""Bank relationship at the start of the simulation: products, housing costs, bills,
income, digital behaviour, financial health and behavioural intensities.

Writes customer-level facts used later by events, labels and transactions. The
system-of-record tables (accounts, product_holdings, balances) are assembled in warehouse.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.simulate.population import categorical
from gcip.simulate.reference import Reference


def _annuity(principal: np.ndarray, annual_rate: np.ndarray, months: np.ndarray) -> np.ndarray:
    r = annual_rate / 12.0
    return principal * r / (1 - (1 + r) ** (-months))


TYPE_MEDIAN_AREA = {"detached": 140.0, "semi": 92.0, "terraced": 82.0, "flat": 61.0, "caravan": 45.0}
START = np.datetime64("2025-01-01")


def _months_between(earlier: np.ndarray, later: np.datetime64) -> np.ndarray:
    return (later.astype("datetime64[M]") - earlier.astype("datetime64[M]")).astype(int)


def _balance_after(principal: np.ndarray, annual_rate: np.ndarray, months: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    """Outstanding principal of a repayment loan after ``elapsed`` monthly payments."""
    r = annual_rate / 12.0
    payment = _annuity(principal, annual_rate, months)
    grow = (1 + r) ** elapsed
    return np.maximum(principal * grow - payment * (grow - 1) / r, 0.0)


def build_banking(pop: pd.DataFrame, veh: pd.DataFrame, homes: pd.DataFrame, cfg: dict,
                  rng: np.random.Generator, ref: Reference) -> pd.DataFrame:
    bcfg, hcfg = cfg["banking"], cfg["homes"]
    n = len(pop)
    tenure = pop["tenure"].values
    region = pop["region"].values
    decile = pop["income_decile"].values
    decile_scaled = (decile - 5.5) / 2.87
    age = pop["age"].values
    london = region == "London"

    out = pd.DataFrame({"customer_id": pop["customer_id"].values})

    # ---------------------------------------------------------------- income
    weekly = pop["gross_income_week"].values
    net_ratio = np.clip(0.86 - 0.025 * decile, 0.58, 0.84)
    monthly_net = weekly * 52 / 12 * net_ratio
    out["monthly_net_income"] = monthly_net.round(2)
    out["pay_day"] = rng.integers(24, 29, n)
    out["income_source"] = np.select(
        [np.isin(pop["employment"].values, ["employed"]), pop["employment"].values == "self_employed",
         pop["employment"].values == "retired", pop["employment"].values == "student"],
        ["salary", "self_employed", "pension", "student"], default="benefits",
    )

    # ---------------------------------------------------------------- housing costs
    # Mortgages: the property is valued at the UK HPI price for its authority, type and origination month, scaled
    # by floor area; the loan is a loan-to-value share of that value (assumption on the LTV mix), repaid monthly.
    # The rate is the Bank of England quoted rate for the customer's current fixed deal, at the month it started.
    mcfg = hcfg["mortgage"]
    mortgage = tenure == "mortgage"
    years_since = np.minimum(rng.uniform(0.5, 12, n), np.maximum(pop["years_at_address"].values, 0.5))
    start = (START - (years_since * 365.25).astype("timedelta64[D]")).astype("datetime64[D]")
    acc = pop["accommodation"].values
    price = ref.house_price(pop["la_code"].values, start, acc)
    type_area = np.array([TYPE_MEDIAN_AREA[a] for a in acc])
    property_value = price * (homes["floor_area_m2"].values / type_area) ** 0.5 * np.exp(rng.normal(0, 0.18, n))
    ltv0 = np.clip(rng.beta(5.0, 2.2, n) * 0.95, 0.35, 0.95)
    original = property_value * ltv0
    age_at_start = age - years_since
    term_total = np.minimum(rng.integers(mcfg["term_years"][0] + 10, mcfg["term_years"][1] + 6, n) * 12,
                            np.maximum((75 - age_at_start) * 12, 60).astype(int))
    elapsed = np.minimum(_months_between(start, START), term_total - 12)
    fix_years = np.where(rng.random(n) < mcfg["two_year_fix_share"], 2, 5)
    fix_start = np.maximum(start, (START - (rng.random(n) * fix_years * 365.25).astype("timedelta64[D]")).astype("datetime64[D]"))
    ltv_at_fix = ltv0 * np.clip(1 - _months_between(start, fix_start) / np.maximum(term_total, 1) * 0.6, 0.3, 1.0)
    band = np.select([ltv_at_fix <= 0.60, ltv_at_fix <= 0.75, ltv_at_fix <= 0.85, ltv_at_fix <= 0.90], ["60", "75", "85", "90"], "95")
    rate = np.zeros(n)
    for b in ("60", "75", "85", "90", "95"):
        two = (fix_years == 2) & (band == b)
        rate[two] = ref.rate(f"mortgage_2y_fixed_{b}ltv", fix_start[two])
    five = fix_years == 5
    rate[five] = ref.rate("mortgage_5y_fixed_75ltv", fix_start[five]) + np.where(ltv_at_fix[five] > 0.85, 0.25, 0.0)
    rate = np.clip(rate / 100.0 + rng.normal(0, 0.002, n), 0.008, 0.09)
    balance = _balance_after(original, rate, term_total, elapsed)
    term_left = np.maximum(term_total - elapsed, 12)
    mortgage_payment = _annuity(balance, rate, term_left)
    with_us = mortgage & (rng.random(n) < mcfg["with_us_share"])
    paid_elsewhere = mortgage & ~with_us & (rng.random(n) < mcfg["paid_from_other_bank_share"] / (1 - mcfg["with_us_share"]))
    out["mortgage"] = mortgage
    out["mortgage_with_us"] = with_us
    out["mortgage_paid_elsewhere"] = paid_elsewhere
    out["mortgage_balance"] = np.where(mortgage, balance.round(0), 0.0)
    out["mortgage_original"] = np.where(mortgage, original.round(0), 0.0)
    out["mortgage_rate"] = np.where(mortgage, rate.round(4), np.nan)
    out["mortgage_fix_years"] = np.where(mortgage, fix_years, 0)
    out["mortgage_fix_start"] = np.where(mortgage, fix_start, np.datetime64("NaT")).astype("datetime64[D]")
    out["mortgage_monthly"] = np.where(mortgage, mortgage_payment.round(2), 0.0)
    out["mortgage_term_left_months"] = np.where(mortgage, term_left, 0)
    out["mortgage_term_months"] = np.where(mortgage, term_total, 0)
    out["property_value_at_origination"] = np.where(mortgage, np.round(property_value, -3), np.nan)
    out["mortgage_start"] = np.where(mortgage, start, np.datetime64("NaT")).astype("datetime64[D]")
    # The lender matches the property to the EPC register; homes without a certificate cannot be matched.
    out["epc_known_to_bank"] = with_us & homes["epc_registered"].values & (rng.random(n) < mcfg["epc_match_rate"])

    rcfg = hcfg["rent_gbp_month"]
    priv_lo = np.where(london, rcfg["private"]["London"][0], rcfg["private"]["default"][0])
    priv_hi = np.where(london, rcfg["private"]["London"][1], rcfg["private"]["default"][1])
    rent = np.select(
        [tenure == "private_rent", tenure == "social_rent"],
        [priv_lo + rng.random(n) * (priv_hi - priv_lo), rng.uniform(*rcfg["social"], n)], default=0.0,
    )
    out["rent_monthly"] = rent.round(0)

    bills_elsewhere = homes["energy_paid_elsewhere"].values | (rng.random(n) < 0.04)
    out["bills_paid_elsewhere"] = bills_elsewhere
    area_factor = (homes["floor_area_m2"].values / 90.0) ** 0.5
    out["council_tax_monthly"] = (rng.uniform(*hcfg["council_tax_gbp_month"], n) * area_factor).clip(70, 520).round(2)
    out["water_monthly"] = (rng.uniform(*hcfg["water_bill_gbp_month"], n) * (pop["household_size"].values / 2.4) ** 0.3).round(2)

    # ---------------------------------------------------------------- everyday banking
    out["has_savings"] = rng.random(n) < np.clip(bcfg["savings_account_share"] * (1 + 0.5 * decile_scaled), 0.1, 0.95)
    savings = np.exp(rng.normal(8.0 + 0.8 * decile_scaled + 0.7 * pop["z_liquidity"].values, 1.1))
    out["savings_balance"] = np.where(out["has_savings"], savings.round(0), 0.0)
    out["savings_rate"] = np.where(out["has_savings"], np.clip(ref.rate("cash_isa_variable_excl_bonus", np.full(n, START)) / 100.0
                                                               + rng.normal(0, 0.003, n), 0.001, 0.06).round(4), np.nan)
    out["has_credit_card"] = rng.random(n) < bcfg["credit_card_share"]
    out["card_apr"] = np.where(out["has_credit_card"], np.clip(ref.rate("credit_card", np.full(n, START)) / 100.0
                                                              + rng.normal(0, 0.03, n), 0.189, 0.399).round(4), np.nan)
    limit = np.exp(rng.normal(8.0 + 0.45 * decile_scaled, 0.5))
    util = np.clip(rng.beta(1.2, 3.0, n) * np.exp(-0.5 * pop["z_liquidity"].values), 0, 1)
    out["card_limit"] = np.where(out["has_credit_card"], np.round(limit, -2), 0.0)
    out["card_balance"] = np.where(out["has_credit_card"], (limit * util).round(0), 0.0)

    out["has_personal_loan"] = rng.random(n) < bcfg["personal_loan_share"] * np.exp(-0.4 * pop["z_liquidity"].values)
    purposes = np.array(["car", "home_improvement", "debt_consolidation", "wedding_or_holiday", "other"])
    out["personal_loan_purpose"] = np.where(out["has_personal_loan"], purposes[categorical(rng, np.tile([0.24, 0.30, 0.26, 0.10, 0.10], (n, 1)))], "")
    # Personal loans priced at the Bank of England quoted rate for the loan size in the month it started, with
    # spread for risk-based pricing (assumption).
    loan_amount = np.round(rng.uniform(3000, 25000, n), -2)
    loan_term = rng.choice([24, 36, 48, 60], n)
    loan_elapsed = (rng.random(n) * (loan_term - 1)).astype(int) + 1
    loan_start = (START.astype("datetime64[M]") - loan_elapsed.astype("timedelta64[M]")).astype("datetime64[D]")
    loan_rate = np.where(loan_amount >= 7500, ref.rate("personal_loan_10k", loan_start), ref.rate("personal_loan_5k", loan_start))
    loan_rate = np.clip(loan_rate / 100.0 + rng.normal(0, 0.015, n), 0.029, 0.299)
    has_pl = out["has_personal_loan"].values
    out["personal_loan_amount"] = np.where(has_pl, loan_amount, 0.0)
    out["personal_loan_rate"] = np.where(has_pl, loan_rate.round(4), np.nan)
    out["personal_loan_term_months"] = np.where(has_pl, loan_term, 0)
    out["personal_loan_start"] = np.where(has_pl, loan_start, np.datetime64("NaT")).astype("datetime64[D]")
    out["personal_loan_monthly"] = np.where(has_pl, _annuity(loan_amount, loan_rate, loan_term).round(2), 0.0)
    out["personal_loan_balance"] = np.where(has_pl, _balance_after(loan_amount, loan_rate, loan_term, loan_elapsed).round(0), 0.0)
    out["external_card_repayment"] = np.where(rng.random(n) < 0.22, rng.uniform(40, 350, n).round(2), 0.0)

    # Investments (FCA Financial Lives 2024: 35% of adults held investments).
    p_inv = bcfg["investments_share"] * np.exp(0.55 * decile_scaled + 0.4 * pop["z_risk"].values)
    p_inv = p_inv * bcfg["investments_share"] / np.mean(p_inv)
    investor = rng.random(n) < np.clip(p_inv, 0, 0.95)
    inv_with_us = investor & (rng.random(n) < bcfg["investments_with_us_share"])
    out["investor"] = investor
    out["investments_with_us"] = inv_with_us
    out["investment_balance"] = np.where(investor, np.exp(rng.normal(9.3 + 0.6 * decile_scaled, 1.1)).round(0), 0.0)
    regular = investor & (rng.random(n) < 0.55)
    out["regular_investor"] = regular
    out["monthly_investment"] = np.where(regular, rng.uniform(50, 600, n).round(0), 0.0)
    pref_known = inv_with_us & (rng.random(n) < bcfg["stated_sustainability_pref_known_share"] / bcfg["investments_with_us_share"])
    pref_value = (pop["z_green"].values + rng.normal(0, 0.6, n)) > 0.45
    out["stated_sustainability_pref"] = np.where(pref_known, pref_value.astype(float), np.nan)

    # ---------------------------------------------------------------- digital behaviour and statements
    z_dig = pop["z_digital"].values
    rate = np.exp(1.9 + 0.9 * z_dig)
    inactive_cut = np.quantile(z_dig, bcfg["digitally_inactive_target_share"])
    inactive = z_dig <= inactive_cut
    out["logins_per_month_rate"] = np.where(inactive, 0.0, rate.round(2))
    out["app_user"] = ~inactive & (rng.random(n) < 1 / (1 + np.exp(-(1.2 + 1.3 * z_dig))))
    paper_base = np.array([bcfg["paper_statement_share_by_age"][b] for b in pop["age_band"].values])
    p_paper = np.clip(paper_base * np.exp(-0.8 * z_dig), 0, 0.97)
    out["paper_statements"] = inactive | (rng.random(n) < p_paper)

    # ---------------------------------------------------------------- financial health
    out["in_arrears"] = rng.random(n) < bcfg["arrears_rate"] * np.exp(-0.9 * pop["z_liquidity"].values)
    out["overdraft_heavy"] = rng.random(n) < bcfg["overdraft_heavy_rate"] * np.exp(-0.8 * pop["z_liquidity"].values)
    unsecured = out["personal_loan_monthly"].values + out["external_card_repayment"].values + 0.03 * out["card_balance"].values
    out["debt_to_income"] = (unsecured / np.maximum(monthly_net, 300)).round(3)

    # ---------------------------------------------------------------- behavioural intensities (latent, observable through spend)
    owner = np.isin(tenure, ["owned_outright", "mortgage"])
    house = pop["accommodation"].values != "flat"
    out["diy_intensity"] = np.exp(rng.normal(-0.2 + 0.6 * owner + 0.3 * house + 0.3 * pop["z_stay"].values
                                             + 0.2 * decile_scaled, 0.7)).round(3)
    cyclist = (pop["commute_mode"].values == "bicycle") | ((pop["z_green"].values > 0.8) & (rng.random(n) < 0.35))
    out["cyclist"] = cyclist
    p_intent = 1 / (1 + np.exp(-(-2.4 + 0.7 * pop["z_green"].values + 0.9 * cyclist)))
    out["green_purchase_intent"] = rng.random(n) < p_intent
    green_supplier_odds = np.exp(hcfg["billing"]["green_supplier_boost"] * pop["z_green"].values)
    out["green_supplier_affinity"] = green_supplier_odds.round(3)
    return out
