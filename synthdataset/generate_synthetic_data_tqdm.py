"""
Green Customer Insights Platform - Synthetic Data Generator
Megathon X: Singularity - Problem Statement 1 (Lloyds)
"""

import math
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Set fixed seeds for deterministic reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_AS_OF = "2024-03-31"
DATA_AS_OF_DT = datetime.strptime(DATA_AS_OF, "%Y-%m-%d")

REGIONS = [
    "North East",
    "North West",
    "Yorkshire and The Humber",
    "East Midlands",
    "West Midlands",
    "East of England",
    "London",
    "South East",
    "South West",
    "Wales",
    "Scotland",
    "Northern Ireland",
]

PROPERTY_TYPES = ["DETACHED", "SEMI_DETACHED", "TERRACED", "FLAT_APARTMENT"]
EPC_RATINGS = ["A", "B", "C", "D", "E", "F", "G"]
HEATING_TYPES = ["GAS_BOILER", "ELECTRIC", "OIL", "HEAT_PUMP", "OTHER"]
VEHICLE_TYPES = ["PETROL", "DIESEL", "HYBRID", "BEV", "NONE"]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ==============================================================================
# 1. GENERATE CUSTOMERS
# ==============================================================================
def generate_customers(num_customers=50):
    customers = []

    for i in tqdm(range(1, num_customers + 1), total=num_customers, desc="Generating customers", unit="customer"):
        cid = f"C{i:05d}"
        age = int(np.random.choice(range(21, 80), p=np.exp(-np.linspace(0, 2, 59)) / np.exp(-np.linspace(0, 2, 59)).sum()))
        region = random.choice(REGIONS)
        tenure_months = int(random.randint(6, min(240, (age - 18) * 12)))

        # Income logic conditioned on age & region
        base_income = 24000 if region not in ["London", "South East"] else 32000
        income_annual = round(float(np.random.lognormal(mean=np.log(base_income + (age * 300)), sigma=0.45)), 2)

        # Home ownership logic
        if age < 28:
            home_ownership = np.random.choice(["RENTER", "OWNER_MORTGAGED"], p=[0.75, 0.25])
        elif age > 58:
            home_ownership = np.random.choice(["OWNER_OUTRIGHT", "OWNER_MORTGAGED", "RENTER"], p=[0.60, 0.30, 0.10])
        else:
            home_ownership = np.random.choice(["OWNER_MORTGAGED", "OWNER_OUTRIGHT", "RENTER"], p=[0.65, 0.15, 0.20])

        has_mortgage = home_ownership == "OWNER_MORTGAGED"
        if has_mortgage:
            mortgage_balance = round(float(income_annual * random.uniform(1.8, 4.2)), 2)
            prop_type = np.random.choice(PROPERTY_TYPES, p=[0.25, 0.40, 0.25, 0.10])
        elif home_ownership == "OWNER_OUTRIGHT":
            mortgage_balance = None
            prop_type = np.random.choice(PROPERTY_TYPES, p=[0.35, 0.35, 0.20, 0.10])
        else:  # RENTER
            mortgage_balance = None
            prop_type = np.random.choice(PROPERTY_TYPES, p=[0.05, 0.20, 0.35, 0.40])

        floor_area = round(float(random.uniform(45.0, 85.0) if prop_type == "FLAT_APARTMENT" else random.uniform(75.0, 185.0)), 1)
        epc = np.random.choice(EPC_RATINGS, p=[0.03, 0.12, 0.35, 0.30, 0.12, 0.06, 0.02])

        if home_ownership == "RENTER":
            solar_installed = False
            has_off_street = random.choice([True, False]) if prop_type != "FLAT_APARTMENT" else False
        else:
            solar_installed = bool(np.random.choice([True, False], p=[0.14, 0.86]))
            has_off_street = bool(np.random.choice([True, False], p=[0.80, 0.20]))

        heating = np.random.choice(HEATING_TYPES, p=[0.74, 0.12, 0.05, 0.07, 0.02])

        # Vehicle logic
        if region == "London" and age < 35:
            veh_type = np.random.choice(VEHICLE_TYPES, p=[0.25, 0.10, 0.15, 0.10, 0.40])
        else:
            veh_type = np.random.choice(VEHICLE_TYPES, p=[0.48, 0.22, 0.15, 0.10, 0.05])

        if veh_type == "NONE":
            mileage = 0
            veh_age = None
            has_auto_loan = False
            auto_months = None
            auto_balance = None
            fuel_spend = 0.0
            auto_spend = 0.0
        else:
            mileage = int(random.randint(4000, 18000))
            veh_age = int(random.randint(1, 14))
            has_auto_loan = bool(np.random.choice([True, False], p=[0.35, 0.65]))
            if has_auto_loan:
                auto_months = int(random.randint(3, 48))
                auto_balance = round(float(random.uniform(2500, 18000)), 2)
            else:
                auto_months = None
                auto_balance = None

            if veh_type in ["PETROL", "DIESEL"]:
                fuel_spend = round(float((mileage / 12.0) * random.uniform(0.12, 0.18)), 2)
            elif veh_type == "HYBRID":
                fuel_spend = round(float((mileage / 12.0) * random.uniform(0.06, 0.09)), 2)
            else:  # BEV
                fuel_spend = 0.0
            auto_spend = round(float(random.uniform(15.0, 75.0)), 2)

        # Utilities & Commute spend
        elec_spend = round(float(random.uniform(65.0, 140.0) + (40.0 if veh_type == "BEV" else 0.0)), 2)
        gas_spend = round(float(random.uniform(60.0, 160.0) if heating == "GAS_BOILER" else random.uniform(0.0, 25.0)), 2)
        pt_spend = round(float(random.uniform(60.0, 220.0) if (veh_type == "NONE" or region == "London") else random.uniform(0.0, 45.0)), 2)
        home_imp_spend = round(float(random.uniform(30.0, 250.0) if home_ownership != "RENTER" else random.uniform(0.0, 30.0)), 2)

        # Balances & logins
        current_acc_bal = round(float(max(100.0, np.random.normal(income_annual * 0.08, 1200))), 2)
        has_investment = bool(np.random.choice([True, False], p=[0.25, 0.75]))
        inv_bal = round(float(random.uniform(2000, 65000)), 2) if has_investment else None

        # Digital behaviour
        paper_opt_in = bool(np.random.choice([True, False], p=[0.35, 0.65] if age > 50 else [0.15, 0.85]))
        statements_yr = int(random.choice([12, 4, 1])) if paper_opt_in else 0
        mobile_logins = int(max(0, np.random.poisson(18 if age < 40 else 8)))
        web_logins = int(max(0, np.random.poisson(5 if age >= 40 else 2)))
        sust_pref = bool(np.random.choice([True, False, None], p=[0.30, 0.20, 0.50]))

        # Split assignment (70% train, 15% validation, 15% test)
        split_val = np.random.choice(["train", "validation", "test"], p=[0.70, 0.15, 0.15])

        customers.append({
            "customer_id": cid,
            "age": age,
            "income_annual_estimated": income_annual,
            "region": region,
            "tenure_months": tenure_months,
            "paper_statement_opt_in": paper_opt_in,
            "has_current_account": True,
            "has_mortgage": has_mortgage,
            "has_auto_loan": has_auto_loan,
            "has_investment_account": has_investment,
            "home_ownership_status": home_ownership,
            "property_type": prop_type,
            "epc_rating": epc,
            "solar_installed": solar_installed,
            "current_vehicle_type": veh_type,
            "estimated_annual_mileage": mileage,
            "auto_loan_months_remaining": auto_months,
            "current_account_balance": current_acc_bal,
            "investment_account_balance": inv_bal,
            "mortgage_balance_remaining": mortgage_balance,
            "auto_loan_balance_remaining": auto_balance,
            "mobile_app_logins_monthly": mobile_logins,
            "web_portal_logins_monthly": web_logins,
            "avg_monthly_fuel_spend": fuel_spend,
            "avg_monthly_electricity_spend": elec_spend,
            "avg_monthly_gas_spend": gas_spend,
            "avg_monthly_public_transport_spend": pt_spend,
            "avg_monthly_home_improvement_spend": home_imp_spend,
            "avg_monthly_automotive_spend": auto_spend,
            "heating_type": heating,
            "floor_area_sqm": floor_area,
            "has_off_street_parking": has_off_street,
            "vehicle_age_years": veh_age,
            "statements_per_year": statements_yr,
            "sustainability_preference_flag": sust_pref,
            "data_as_of": DATA_AS_OF,
            "split": split_val
        })

    return pd.DataFrame(customers)


# ==============================================================================
# 2. GENERATE TRANSACTIONS CONDITIONED ON CUSTOMERS
# ==============================================================================
def generate_transactions(df_customers, days_history=90):
    transactions = []
    tx_counter = 1

    start_date = DATA_AS_OF_DT - timedelta(days=days_history)

    for _, cust in tqdm(df_customers.iterrows(), total=len(df_customers), desc="Generating transactions", unit="customer"):
        cid = cust["customer_id"]

        # A. Monthly Salary Credit (positive)
        net_monthly_salary = round((cust["income_annual_estimated"] * 0.76) / 12.0, 2)
        for month_offset in [15, 45, 75]:
            ts = (start_date + timedelta(days=month_offset, hours=9, minutes=15)).strftime("%Y-%m-%dT%H:%M:%S")
            transactions.append({
                "transaction_id": f"TX{tx_counter:07d}",
                "customer_id": cid,
                "timestamp": ts,
                "amount": net_monthly_salary,
                "merchant_name": "EMPLOYER PAYROLL BACS",
                "mcc": None,
                "transaction_type": "CREDIT",
                "category_ground_truth": "Income"
            })
            tx_counter += 1

        # B. Housing / Mortgage DD (negative)
        if cust["has_mortgage"]:
            mortgage_monthly = round(cust["mortgage_balance_remaining"] * 0.0055, 2)
            for month_offset in [2, 32, 62]:
                ts = (start_date + timedelta(days=month_offset, hours=4, minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": ts,
                    "amount": -mortgage_monthly,
                    "merchant_name": "LLOYDS BANK MORTGAGE DD",
                    "mcc": None,
                    "transaction_type": "DIRECT_DEBIT",
                    "category_ground_truth": "Housing"
                })
                tx_counter += 1
        elif cust["home_ownership_status"] == "RENTER":
            rent_monthly = round(random.uniform(650.0, 1600.0), 2)
            for month_offset in [1, 31, 61]:
                ts = (start_date + timedelta(days=month_offset, hours=6, minutes=0)).strftime("%Y-%m-%dT%H:%M:%S")
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": ts,
                    "amount": -rent_monthly,
                    "merchant_name": "RESIDENTIAL LETTINGS DIRECT DEBIT",
                    "mcc": None,
                    "transaction_type": "DIRECT_DEBIT",
                    "category_ground_truth": "Housing"
                })
                tx_counter += 1

        # C. Auto Loan DD (if applicable)
        if cust["has_auto_loan"]:
            auto_pmt = round(cust["auto_loan_balance_remaining"] / max(1, cust["auto_loan_months_remaining"]), 2)
            for month_offset in [5, 35, 65]:
                ts = (start_date + timedelta(days=month_offset, hours=5, minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": ts,
                    "amount": -auto_pmt,
                    "merchant_name": "BLACK HORSE MOTOR FINANCE DD",
                    "mcc": None,
                    "transaction_type": "DIRECT_DEBIT",
                    "category_ground_truth": "Debt"
                })
                tx_counter += 1

        # D. Electricity & Gas Utility DD
        for month_offset in [10, 40, 70]:
            if cust["avg_monthly_electricity_spend"] > 0:
                e_amt = round(np.random.normal(cust["avg_monthly_electricity_spend"], 8.0), 2)
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": (start_date + timedelta(days=month_offset, hours=7)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "amount": -abs(e_amt),
                    "merchant_name": "OCTOPUS ENERGY ELECTRICITY DD",
                    "mcc": None,
                    "transaction_type": "DIRECT_DEBIT",
                    "category_ground_truth": "Electricity"
                })
                tx_counter += 1

            if cust["avg_monthly_gas_spend"] > 10.0:
                g_amt = round(np.random.normal(cust["avg_monthly_gas_spend"], 10.0), 2)
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": (start_date + timedelta(days=month_offset + 1, hours=7)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "amount": -abs(g_amt),
                    "merchant_name": "BRITISH GAS DUAL FUEL DD",
                    "mcc": None,
                    "transaction_type": "DIRECT_DEBIT",
                    "category_ground_truth": "Gas"
                })
                tx_counter += 1

        # E. Fuel Card Payments (Petrol/Diesel/Hybrid)
        if cust["avg_monthly_fuel_spend"] > 15.0:
            fuel_tx_count = max(2, int(cust["avg_monthly_fuel_spend"] / 45.0) * 3)
            for _ in range(fuel_tx_count):
                day_rnd = random.randint(1, days_history - 1)
                f_amt = round(random.uniform(35.0, 75.0), 2)
                merch = random.choice(["SHELL SERVICE STATION", "BP EXPRESS OIL", "ESSO PETROL", "TEXACO FORECOURT"])
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": (start_date + timedelta(days=day_rnd, hours=random.randint(7, 21), minutes=random.randint(0, 59))).strftime("%Y-%m-%dT%H:%M:%S"),
                    "amount": -abs(f_amt),
                    "merchant_name": merch,
                    "mcc": "5541",
                    "transaction_type": "CARD_PAYMENT",
                    "category_ground_truth": "Fuel"
                })
                tx_counter += 1

        # F. Public Transport
        if cust["avg_monthly_public_transport_spend"] > 20.0:
            pt_count = max(3, int(cust["avg_monthly_public_transport_spend"] / 15.0) * 2)
            for _ in range(pt_count):
                day_rnd = random.randint(1, days_history - 1)
                pt_amt = round(random.uniform(2.80, 18.50), 2)
                merch = random.choice(["TFL TRAVEL CHARGE", "NATIONAL RAIL TICKET", "TRAINLINE.COM", "STAGECOACH BUS"])
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": (start_date + timedelta(days=day_rnd, hours=random.randint(6, 22), minutes=random.randint(0, 59))).strftime("%Y-%m-%dT%H:%M:%S"),
                    "amount": -abs(pt_amt),
                    "merchant_name": merch,
                    "mcc": "4111",
                    "transaction_type": "CARD_PAYMENT",
                    "category_ground_truth": "Public Transport"
                })
                tx_counter += 1

        # G. Home Improvement
        if cust["avg_monthly_home_improvement_spend"] > 25.0:
            hi_count = random.randint(2, 5)
            for _ in range(hi_count):
                day_rnd = random.randint(1, days_history - 1)
                hi_amt = round(random.uniform(25.0, 180.0), 2)
                merch = random.choice(["B&Q SUPERSTORE", "WICKES BUILDING SUPPLIES", "SCREWFIX DIRECT", "TOOLSTATION LTD"])
                transactions.append({
                    "transaction_id": f"TX{tx_counter:07d}",
                    "customer_id": cid,
                    "timestamp": (start_date + timedelta(days=day_rnd, hours=random.randint(10, 19), minutes=random.randint(0, 59))).strftime("%Y-%m-%dT%H:%M:%S"),
                    "amount": -abs(hi_amt),
                    "merchant_name": merch,
                    "mcc": "5200",
                    "transaction_type": "CARD_PAYMENT",
                    "category_ground_truth": "Home Improvement"
                })
                tx_counter += 1

        # H. Automotive / Servicing
        if cust["avg_monthly_automotive_spend"] > 15.0:
            day_rnd = random.randint(10, days_history - 5)
            auto_s_amt = round(random.uniform(50.0, 220.0), 2)
            transactions.append({
                "transaction_id": f"TX{tx_counter:07d}",
                "customer_id": cid,
                "timestamp": (start_date + timedelta(days=day_rnd, hours=14, minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
                "amount": -abs(auto_s_amt),
                "merchant_name": random.choice(["HALFORDS AUTOCENTRE", "KWIK FIT TYRES", "NATIONAL TYRES"]),
                "mcc": "7538",
                "transaction_type": "CARD_PAYMENT",
                "category_ground_truth": "Automotive"
            })
            tx_counter += 1

        # I. General Everyday "Other" Spend
        for _ in range(random.randint(4, 8)):
            day_rnd = random.randint(1, days_history - 1)
            other_amt = round(random.uniform(4.50, 85.00), 2)
            transactions.append({
                "transaction_id": f"TX{tx_counter:07d}",
                "customer_id": cid,
                "timestamp": (start_date + timedelta(days=day_rnd, hours=random.randint(8, 21), minutes=random.randint(0, 59))).strftime("%Y-%m-%dT%H:%M:%S"),
                "amount": -abs(other_amt),
                "merchant_name": random.choice(["SAINSBURYS SUPERMARKET", "TESCO STORES", "AMAZON UK", "BOOTS CHEMIST", "COSTA COFFEE"]),
                "mcc": "5411",
                "transaction_type": "CARD_PAYMENT",
                "category_ground_truth": "Other"
            })
            tx_counter += 1

    df_tx = pd.DataFrame(transactions)
    return df_tx.sort_values(by="timestamp").reset_index(drop=True)


# ==============================================================================
# 3. GENERATE PRODUCT LABELS & REASON CODES
# ==============================================================================
def generate_product_labels(df_customers):
    labels = []

    products = [
        {"id": "GP01", "name": "EV Finance"},
        {"id": "GP02", "name": "Green Home Loan"},
        {"id": "GP03", "name": "Solar Finance"},
        {"id": "GP04", "name": "Paperless Banking"},
    ]

    for _, cust in tqdm(df_customers.iterrows(), total=len(df_customers), desc="Generating product labels", unit="customer"):
        cid = cust["customer_id"]

        for prod in products:
            pid = prod["id"]
            pname = prod["name"]

            eligible = True
            ineligibility_reason = None
            reasons = []
            logit = 0.0

            # ------------------------------------------------------------------
            # GP01: EV Finance
            # ------------------------------------------------------------------
            if pid == "GP01":
                if cust["current_vehicle_type"] == "BEV":
                    eligible = False
                    ineligibility_reason = "ALREADY_BEV"
                    propensity = 0.0
                elif cust["current_vehicle_type"] == "NONE":
                    eligible = False
                    ineligibility_reason = "NO_DATA"
                    propensity = 0.0
                else:
                    # Signals
                    if cust["avg_monthly_fuel_spend"] > 110:
                        reasons.append("HIGH_FUEL_SPEND")
                        logit += 0.95
                    if cust["estimated_annual_mileage"] > 10000:
                        reasons.append("HIGH_ANNUAL_MILEAGE")
                        logit += 0.80
                    if cust["has_off_street_parking"]:
                        reasons.append("OFF_STREET_PARKING")
                        logit += 0.65
                    if cust["has_auto_loan"] and (cust["auto_loan_months_remaining"] or 99) <= 12:
                        reasons.append("EXPIRING_AUTO_FINANCE")
                        logit += 0.75
                    if cust["vehicle_age_years"] and cust["vehicle_age_years"] >= 6:
                        reasons.append("AGING_ICE_VEHICLE")
                        logit += 0.50

                    propensity = round(sigmoid(logit - 1.2), 4)

            # ------------------------------------------------------------------
            # GP02: Green Home Loan
            # ------------------------------------------------------------------
            elif pid == "GP02":
                if cust["home_ownership_status"] == "RENTER":
                    eligible = False
                    ineligibility_reason = "NOT_HOME_OWNER"
                    propensity = 0.0
                else:
                    if cust["epc_rating"] in ["D", "E", "F", "G"]:
                        reasons.append("SUBOPTIMAL_EPC_RATING")
                        logit += 1.10
                    if cust["heating_type"] in ["GAS_BOILER", "OIL"]:
                        reasons.append("FOSSIL_FUEL_HEATING")
                        logit += 0.85
                    if cust["avg_monthly_gas_spend"] > 95:
                        reasons.append("HIGH_GAS_BILLS")
                        logit += 0.70
                    if cust["avg_monthly_home_improvement_spend"] > 80:
                        reasons.append("ACTIVE_HOME_IMPROVER")
                        logit += 0.60
                    if cust["has_mortgage"]:
                        reasons.append("EXISTING_MORTGAGE_RELATIONSHIP")
                        logit += 0.45

                    propensity = round(sigmoid(logit - 1.5), 4)

            # ------------------------------------------------------------------
            # GP03: Solar Finance
            # ------------------------------------------------------------------
            elif pid == "GP03":
                if cust["home_ownership_status"] == "RENTER":
                    eligible = False
                    ineligibility_reason = "NOT_HOME_OWNER"
                    propensity = 0.0
                elif cust["solar_installed"] is True:
                    eligible = False
                    ineligibility_reason = "SOLAR_ALREADY_INSTALLED"
                    propensity = 0.0
                else:
                    if cust["avg_monthly_electricity_spend"] > 100:
                        reasons.append("HIGH_ELECTRICITY_CONSUMPTION")
                        logit += 1.15
                    if cust["property_type"] in ["DETACHED", "SEMI_DETACHED"]:
                        reasons.append("SUITABLE_ROOF_PROPERTY")
                        logit += 0.90
                    if (cust["floor_area_sqm"] or 0) >= 100:
                        reasons.append("LARGE_PROPERTY_FOOTPRINT")
                        logit += 0.55
                    if cust["current_vehicle_type"] == "BEV":
                        reasons.append("HOME_EV_CHARGING_DEMAND")
                        logit += 0.70

                    propensity = round(sigmoid(logit - 1.6), 4)

            # ------------------------------------------------------------------
            # GP04: Paperless Banking
            # ------------------------------------------------------------------
            elif pid == "GP04":
                if not cust["paper_statement_opt_in"]:
                    eligible = False
                    ineligibility_reason = "ALREADY_DIGITAL"
                    propensity = 0.0
                else:
                    if cust["mobile_app_logins_monthly"] >= 6:
                        reasons.append("HIGH_MOBILE_APP_ENGAGEMENT")
                        logit += 1.25
                    if cust["web_portal_logins_monthly"] >= 2:
                        reasons.append("ACTIVE_ONLINE_BANKING")
                        logit += 0.75
                    if cust["statements_per_year"] >= 12:
                        reasons.append("HIGH_PAPER_VOLUME")
                        logit += 0.65
                    if cust["age"] < 45:
                        reasons.append("DIGITALLY_NATIVE_DEMOGRAPHIC")
                        logit += 0.50

                    propensity = round(sigmoid(logit - 0.9), 4)

            # Fallback if no specific reason code triggered
            if eligible and not reasons:
                reasons.append("GENERAL_PORTFOLIO_CANDIDATE")

            # Simulated Take-up (Bernoulli trial; strictly 0 if ineligible)
            if eligible:
                takeup = int(np.random.binomial(1, propensity))
            else:
                takeup = 0

            reason_str = "|".join(reasons) if reasons else "INELIGIBLE"

            labels.append({
                "customer_id": cid,
                "product_id": pid,
                "product_name": pname,
                "eligible_for_recommendation": eligible,
                "ineligibility_reason": ineligibility_reason,
                "propensity_label": propensity,
                "simulated_takeup": takeup,
                "reason_codes": reason_str,
                "label_version": "v1.0",
                "data_as_of": DATA_AS_OF
            })

    return pd.DataFrame(labels)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Starting customer generation...", flush=True)
    df_customers = generate_customers(num_customers=50)

    print("Starting transaction generation...", flush=True)
    df_tx = generate_transactions(df_customers, days_history=90)

    print("Starting product-label generation...", flush=True)
    df_labels = generate_product_labels(df_customers)

    # Save to CSV
    df_customers.to_csv("customers.csv", index=False)
    df_tx.to_csv("transactions.csv", index=False)
    df_labels.to_csv("product_labels.csv", index=False)

    print(f"Generated {len(df_customers)} customers -> customers.csv")
    print(f"Generated {len(df_tx)} transactions -> transactions.csv")
    print(f"Generated {len(df_labels)} product label rows -> product_labels.csv")