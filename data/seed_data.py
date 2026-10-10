"""
Synthetic UK retail banking dataset generator.
Fixed random seed for reproducibility.
All data is synthetic. Labels are SIMULATED, not observed outcomes.

Assumptions documented inline. Run once to populate the database.
"""
import hashlib
import json
import os
import random
import sys
import string
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure backend is importable ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

RANDOM_SEED = 42
NUM_CUSTOMERS = 10_000
START_DATE = date(2023, 1, 1)
END_DATE = date(2023, 12, 31)

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Reference data ────────────────────────────────────────────────────────────
UK_REGIONS = [
    'London', 'South East', 'South West', 'East of England',
    'East Midlands', 'West Midlands', 'Yorkshire and The Humber',
    'North West', 'North East', 'Wales', 'Scotland', 'Northern Ireland',
]
REGION_WEIGHTS = [0.15, 0.14, 0.08, 0.09, 0.07, 0.09, 0.08, 0.10, 0.04, 0.05, 0.08, 0.03]

FIRST_NAMES_M = ['James', 'Oliver', 'Harry', 'Jack', 'George', 'Noah', 'Charlie', 'Jacob', 'Alfie', 'Freddie',
                  'Muhammad', 'Thomas', 'Oscar', 'William', 'Henry', 'Archie', 'Joshua', 'Samuel', 'Lucas', 'Ethan']
FIRST_NAMES_F = ['Olivia', 'Amelia', 'Isla', 'Ava', 'Emily', 'Isabella', 'Mia', 'Poppy', 'Ella', 'Lily',
                  'Sophie', 'Grace', 'Freya', 'Charlotte', 'Evie', 'Jessica', 'Daisy', 'Alice', 'Florence', 'Sienna']
LAST_NAMES = ['Smith', 'Jones', 'Williams', 'Taylor', 'Brown', 'Davies', 'Evans', 'Wilson', 'Thomas', 'Roberts',
               'Johnson', 'Walker', 'Wright', 'Robinson', 'Thompson', 'White', 'Hughes', 'Edwards', 'Green', 'Hall',
               'Lewis', 'Harris', 'Clarke', 'Patel', 'Jackson', 'Wood', 'Turner', 'Martin', 'Cooper', 'Hill']

HOUSING_TYPES = ['Detached', 'Semi-detached', 'Terraced', 'Flat', 'Bungalow']
VEHICLE_TYPES = ['Petrol', 'Diesel', 'Electric', 'Hybrid', 'None']
HOUSEHOLD_TYPES = ['Single', 'Couple', 'Family', 'Shared']

MERCHANTS = {
    'Fuel': [
        ('BP Garage', 1.55), ('Shell Forecourt', 1.58), ('Esso Service Station', 1.52),
        ('Texaco Petrol', 1.56), ('Jet Fuel', 1.50), ('Moto Services', 1.60),
        ('Total Petrol', 1.54), ('Gulf Garage', 1.51), ('BP Fill Up', 1.55),
    ],
    'Electricity': [
        ('Octopus Energy', None), ('British Gas Electric', None), ('EDF Energy', None),
        ('OVO Energy', None), ('E.ON Electric', None), ('Scottish Power Electric', None),
        ('Bulb Energy', None), ('Utility Warehouse Electric', None),
    ],
    'Gas': [
        ('British Gas', None), ('Centrica Gas', None), ('Scottish Power Gas', None),
        ('Npower Gas', None), ('EDF Gas', None), ('Utilita Gas', None),
    ],
    'Public Transport': [
        ('TfL Oyster', None), ('National Rail', None), ('GWR Train', None),
        ('Avanti West Coast', None), ('CrossCountry Rail', None), ('LNER', None),
        ('Stagecoach Bus', None), ('First Bus', None), ('Arriva Trains', None),
        ('Trainline', None), ('Southern Rail', None),
    ],
    'Home Improvement': [
        ('B&Q', None), ('Screwfix', None), ('Wickes', None), ('Homebase', None),
        ('Toolstation', None), ('Travis Perkins', None), ('Tile Giant', None),
        ('Victorian Plumbing', None), ('Loft Insulation Co', None),
    ],
    'Automotive': [
        ('Halfords', None), ('Kwik Fit', None), ('National Tyres', None),
        ('Eurocarparts', None), ('DVLA', None), ('RAC Membership', None),
        ('Arnold Clark', None), ('GSF Car Parts', None),
    ],
    'Utilities': [
        ('Thames Water', None), ('Virgin Media', None), ('Sky Broadband', None),
        ('BT Broadband', None), ('Severn Trent Water', None), ('Council Tax',  None),
        ('TV Licence', None), ('United Utilities', None),
    ],
    'Other': [
        ('Tesco Express', None), ('Sainsburys', None), ('Marks & Spencer', None),
        ('Amazon UK', None), ('Next', None), ('ASOS', None), ('Boots', None),
        ('Argos', None), ('Waterstones', None), ('Pizza Hut', None),
        ('Costa Coffee', None), ('Greggs', None), ('McDonald\'s', None),
        ('Just Eat', None), ('Deliveroo', None), ('Uber Eats', None),
        ('Netflix UK', None), ('Spotify', None), ('Apple Store', None),
        ('JD Sports', None), ('Sports Direct', None),
    ],
}

# Monthly transaction count distribution per category (mean, std)
CATEGORY_MONTHLY = {
    'Fuel':             {'weight': 0.08, 'mean_gbp': 60,  'std_gbp': 25,  'freq_mean': 3, 'freq_std': 1},
    'Electricity':      {'weight': 0.04, 'mean_gbp': 85,  'std_gbp': 30,  'freq_mean': 1, 'freq_std': 0},
    'Gas':              {'weight': 0.03, 'mean_gbp': 55,  'std_gbp': 20,  'freq_mean': 1, 'freq_std': 0},
    'Public Transport': {'weight': 0.06, 'mean_gbp': 35,  'std_gbp': 20,  'freq_mean': 4, 'freq_std': 2},
    'Home Improvement': {'weight': 0.03, 'mean_gbp': 80,  'std_gbp': 50,  'freq_mean': 1, 'freq_std': 1},
    'Automotive':       {'weight': 0.04, 'mean_gbp': 90,  'std_gbp': 40,  'freq_mean': 1, 'freq_std': 1},
    'Utilities':        {'weight': 0.07, 'mean_gbp': 45,  'std_gbp': 20,  'freq_mean': 2, 'freq_std': 1},
    'Other':            {'weight': 0.65, 'mean_gbp': 35,  'std_gbp': 30,  'freq_mean': 8, 'freq_std': 3},
}


def customer_id(idx: int) -> str:
    return f'CUST{idx:06d}'


def account_id(cust_idx: int, acc_idx: int) -> str:
    return f'ACC{cust_idx:06d}{acc_idx:02d}'


def transaction_id(cust_idx: int, txn_idx: int) -> str:
    return f'TXN{cust_idx:06d}{txn_idx:05d}'


def age_band(age: int) -> str:
    if age < 25: return '18-24'
    if age < 35: return '25-34'
    if age < 45: return '35-44'
    if age < 55: return '45-54'
    if age < 65: return '55-64'
    return '65+'


def postcode_district(region: str) -> str:
    mapping = {
        'London': ['EC1A', 'WC2N', 'E1', 'N1', 'NW3', 'SW1A', 'SE1', 'W1A'],
        'South East': ['RH1', 'GU1', 'CT1', 'BN1', 'SO14', 'PO1', 'ME1'],
        'South West': ['BS1', 'EX1', 'PL4', 'TQ1', 'GL1', 'BA1', 'SP1'],
        'East of England': ['CB1', 'NR1', 'IP1', 'CO1', 'SG1', 'AL1', 'LU1'],
        'East Midlands': ['NG1', 'LE1', 'DE1', 'LN1', 'PE1', 'NN1', 'MK1'],
        'West Midlands': ['B1', 'CV1', 'WV1', 'DY1', 'WS1', 'ST1', 'TF1'],
        'Yorkshire and The Humber': ['LS1', 'BD1', 'HX1', 'HG1', 'YO1', 'DN1', 'S1', 'HU1'],
        'North West': ['M1', 'L1', 'PR1', 'BB1', 'LA1', 'WN1', 'SK1', 'CH1'],
        'North East': ['NE1', 'SR1', 'DH1', 'TS1', 'DL1'],
        'Wales': ['CF10', 'SA1', 'NP1', 'LD1', 'SY1', 'LL30'],
        'Scotland': ['EH1', 'G1', 'AB10', 'DD1', 'FK1', 'KA1', 'PA1'],
        'Northern Ireland': ['BT1', 'BT2', 'BT3', 'BT4', 'BT5'],
    }
    choices = mapping.get(region, ['XX1'])
    return random.choice(choices)


def income_for_band(band: str, rng_local) -> float:
    if band == 'Low':
        return round(float(rng_local.uniform(15000, 28000)), -2)
    elif band == 'Medium':
        return round(float(rng_local.uniform(28000, 60000)), -2)
    else:
        return round(float(rng_local.uniform(60000, 200000)), -2)


def split_set(idx: int) -> str:
    r = idx % 10
    if r < 7: return 'train'
    if r < 9: return 'val'
    return 'test'


def generate_customers(n: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        idx = i
        gender = rng.choice(['Male', 'Female'], p=[0.50, 0.50])
        first = random.choice(FIRST_NAMES_M if gender == 'Male' else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        age = int(rng.integers(18, 80))
        region = rng.choice(UK_REGIONS, p=REGION_WEIGHTS)
        income_band = rng.choice(['Low', 'Medium', 'High'], p=[0.30, 0.50, 0.20])
        annual_income = income_for_band(income_band, rng)
        household_size = int(rng.choice([1, 2, 3, 4, 5], p=[0.20, 0.30, 0.25, 0.18, 0.07]))
        household_type = 'Single' if household_size == 1 else ('Couple' if household_size == 2 else ('Family' if household_size >= 3 else 'Shared'))
        # Homeownership correlated with age and income
        homeowner_prob = 0.1 + (age - 18) / 200 + (0.2 if income_band == 'High' else 0.05 if income_band == 'Medium' else 0)
        is_homeowner = bool(rng.random() < homeowner_prob)
        housing_type = rng.choice(HOUSING_TYPES, p=[0.25, 0.28, 0.22, 0.18, 0.07]) if is_homeowner else rng.choice(['Flat', 'Terraced', 'Shared'], p=[0.50, 0.30, 0.20])
        has_mortgage = bool(is_homeowner and rng.random() < 0.70)
        prop_band = None
        if is_homeowner:
            if income_band == 'High': prop_band = '£500k+'
            elif income_band == 'Medium': prop_band = '£200k-£500k'
            else: prop_band = '<£200k'
        # Vehicle correlated with age and region
        rural_bonus = 0.1 if region not in ('London',) else -0.15
        vehicle_prob = 0.30 + (age - 18) / 250 + rural_bonus + (0.15 if income_band == 'High' else 0)
        has_vehicle = bool(rng.random() < vehicle_prob)
        if has_vehicle:
            if income_band == 'High': vehicle_type = rng.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'], p=[0.30, 0.15, 0.30, 0.25])
            else: vehicle_type = rng.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'], p=[0.50, 0.30, 0.05, 0.15])
            est_mileage = float(rng.normal(10000, 4000))
            est_mileage = max(1000, est_mileage)
        else:
            vehicle_type = 'None'
            est_mileage = None
        digital_score = float(rng.beta(3 if age < 50 else 1.5, 1.5 if age < 50 else 3))
        paper_stmts = bool(rng.random() < (0.6 - digital_score * 0.5))
        mobile_active = bool(rng.random() < digital_score)
        since_years = int(rng.integers(1, 20))
        customer_since_date = date(2024, 1, 1) - timedelta(days=since_years * 365)
        segment = 'HNW' if income_band == 'High' else ('Affluent' if income_band == 'Medium' else 'Mass')
        rows.append({
            'customer_id': customer_id(idx),
            'first_name': first,
            'last_name': last,
            'age': age,
            'age_band': age_band(age),
            'gender': gender,
            'region': region,
            'postcode_district': postcode_district(region),
            'income_band': income_band,
            'annual_income_gbp': annual_income,
            'household_size': household_size,
            'household_type': household_type,
            'is_homeowner': is_homeowner,
            'housing_type': housing_type,
            'has_mortgage': has_mortgage,
            'property_value_band': prop_band,
            'has_vehicle': has_vehicle,
            'vehicle_type': vehicle_type,
            'estimated_annual_mileage': est_mileage,
            'digital_activity_score': round(digital_score, 4),
            'paper_statements': paper_stmts,
            'mobile_banking_active': mobile_active,
            'customer_since': customer_since_date,
            'customer_segment': segment,
            'split_set': split_set(i),
        })
    return pd.DataFrame(rows)


def generate_accounts(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in customers.iterrows():
        cust_idx = int(c['customer_id'].replace('CUST', ''))
        n_accounts = rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])
        types = ['Current']
        if n_accounts >= 2: types.append(rng.choice(['Savings', 'ISA']))
        if n_accounts >= 3: types.append(rng.choice(['Savings', 'ISA', 'Instant Access']))
        for j, atype in enumerate(types):
            balance = float(rng.exponential(2000) if atype == 'Current' else rng.exponential(8000))
            rows.append({
                'account_id': account_id(cust_idx, j + 1),
                'customer_id': c['customer_id'],
                'account_type': atype,
                'balance_gbp': round(balance, 2),
                'overdraft_limit_gbp': 500.0 if atype == 'Current' else 0.0,
                'is_primary': j == 0,
                'opened_date': c['customer_since'],
            })
    return pd.DataFrame(rows)


def generate_transactions(customers: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Map customer -> primary account
    primary = accounts[accounts['is_primary']].set_index('customer_id')['account_id'].to_dict()

    for _, c in customers.iterrows():
        cust_id = c['customer_id']
        cust_idx = int(cust_id.replace('CUST', ''))
        acct_id = primary.get(cust_id)
        if not acct_id: continue

        has_vehicle = c['has_vehicle']
        income_band = c['income_band']
        is_homeowner = c['is_homeowner']

        txn_counter = 1
        current_date = START_DATE
        while current_date <= END_DATE:
            for category, params in CATEGORY_MONTHLY.items():
                # Skip vehicle-related categories for non-vehicle owners
                if category == 'Fuel' and not has_vehicle:
                    current_date_check = current_date
                    continue
                # Scale frequency by income
                scale = 1.2 if income_band == 'High' else (1.0 if income_band == 'Medium' else 0.8)
                n_txns = max(0, int(rng.normal(params['freq_mean'] * scale, params['freq_std'])))
                for _ in range(n_txns):
                    merchant_list = MERCHANTS.get(category, MERCHANTS['Other'])
                    merchant_entry = random.choice(merchant_list)
                    merchant_name = merchant_entry[0]
                    # Amount
                    amt = float(rng.normal(params['mean_gbp'], params['std_gbp']))
                    amt = max(0.50, round(amt, 2))
                    # Date within month
                    days_in_month = 28 if current_date.month == 2 else (30 if current_date.month in [4,6,9,11] else 31)
                    day_offset = int(rng.integers(0, days_in_month))
                    txn_date = date(current_date.year, current_date.month, min(day_offset + 1, days_in_month))
                    if txn_date > END_DATE: txn_date = END_DATE
                    rows.append({
                        'transaction_id': transaction_id(cust_idx, txn_counter),
                        'customer_id': cust_id,
                        'account_id': acct_id,
                        'transaction_date': txn_date,
                        'merchant_description': merchant_name,
                        'amount_gbp': amt,
                        'transaction_type': 'Debit',
                        'mcc_code': None,
                        '_category': category,
                    })
                    txn_counter += 1
            # Advance to next month
            if current_date.month == 12:
                current_date = date(current_date.year + 1, 1, 1)
            else:
                current_date = date(current_date.year, current_date.month + 1, 1)

    return pd.DataFrame(rows)


def compute_features(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    # Aggregate monthly spending by customer and category
    if transactions.empty:
        return pd.DataFrame()
    agg = transactions.groupby(['customer_id', '_category'])['amount_gbp'].sum().unstack(fill_value=0)
    # Normalize to monthly (12 months of data)
    agg = agg / 12.0
    agg.columns = [f'monthly_{c.lower().replace(" ", "_")}_spend_gbp' for c in agg.columns]

    feat_rows = []
    for _, c in customers.iterrows():
        cid = c['customer_id']
        row = agg.loc[cid] if cid in agg.index else pd.Series(dtype=float)

        def g(col): return float(row.get(col, 0.0))

        monthly_fuel = g('monthly_fuel_spend_gbp')
        monthly_elec = g('monthly_electricity_spend_gbp')
        monthly_gas = g('monthly_gas_spend_gbp')
        monthly_transport = g('monthly_public_transport_spend_gbp')
        monthly_home = g('monthly_home_improvement_spend_gbp')
        monthly_auto = g('monthly_automotive_spend_gbp')
        monthly_util = g('monthly_utilities_spend_gbp')
        monthly_other = g('monthly_other_spend_gbp')
        monthly_total = monthly_fuel + monthly_elec + monthly_gas + monthly_transport + monthly_home + monthly_auto + monthly_util + monthly_other

        # Annual fuel txn count
        fuel_txns = int(transactions[(transactions['customer_id'] == cid) & (transactions['_category'] == 'Fuel')].shape[0])
        transport_txns = int(transactions[(transactions['customer_id'] == cid) & (transactions['_category'] == 'Public Transport')].shape[0])
        total_txns = int(transactions[transactions['customer_id'] == cid].shape[0])

        est_mileage = c.get('estimated_annual_mileage') or (monthly_fuel * 12 / 1.55 / 0.08 * 100 if monthly_fuel > 0 else 0)
        est_kwh_elec = (monthly_elec * 12) / 0.28 if monthly_elec > 0 else 0
        est_kwh_gas = (monthly_gas * 12) / 0.065 if monthly_gas > 0 else 0
        digital_logins = float(c.get('digital_activity_score', 0.5) * 20)

        # Synthetic labels (simulated - not observed outcomes)
        label_ev = bool(
            c['has_vehicle'] and
            monthly_fuel > 100 and
            c['income_band'] in ('Medium', 'High') and
            rng.random() < 0.35
        )
        label_green_home = bool(
            c['is_homeowner'] and
            (monthly_elec > 60 or monthly_home > 30) and
            rng.random() < 0.30
        )
        label_solar = bool(
            c['is_homeowner'] and
            c['housing_type'] not in ('Flat',) and
            monthly_elec > 70 and
            rng.random() < 0.25
        )
        label_paperless = bool(
            c['paper_statements'] and
            c['digital_activity_score'] > 0.3 and
            rng.random() < 0.55
        )

        feat_rows.append({
            'customer_id': cid,
            'monthly_fuel_spend_gbp': round(monthly_fuel, 2),
            'monthly_electricity_spend_gbp': round(monthly_elec, 2),
            'monthly_gas_spend_gbp': round(monthly_gas, 2),
            'monthly_transport_spend_gbp': round(monthly_transport, 2),
            'monthly_home_improvement_spend_gbp': round(monthly_home, 2),
            'monthly_utilities_spend_gbp': round(monthly_util, 2),
            'monthly_total_spend_gbp': round(monthly_total, 2),
            'annual_fuel_transactions': fuel_txns,
            'annual_transport_transactions': transport_txns,
            'estimated_annual_mileage': round(float(est_mileage), 1) if est_mileage else None,
            'estimated_annual_kwh_electricity': round(est_kwh_elec, 1) if est_kwh_elec else None,
            'estimated_annual_kwh_gas': round(est_kwh_gas, 1) if est_kwh_gas else None,
            'avg_transactions_per_month': round(total_txns / 12, 1),
            'digital_logins_per_month': round(digital_logins, 1),
            'label_ev_taker': label_ev,
            'label_green_home_taker': label_green_home,
            'label_solar_taker': label_solar,
            'label_paperless': label_paperless,
        })
    return pd.DataFrame(feat_rows)


def generate_financial_products(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in customers.iterrows():
        if c['has_mortgage']:
            rows.append({
                'customer_id': c['customer_id'],
                'product_code': 'MORTGAGE',
                'product_name': 'Residential Mortgage',
                'balance_gbp': round(float(rng.uniform(80000, 500000)), -2),
                'monthly_payment_gbp': round(float(rng.uniform(600, 2500)), 0),
                'interest_rate': round(float(rng.uniform(2.5, 5.5)), 2),
                'start_date': c['customer_since'],
                'end_date': None,
                'status': 'Active',
            })
        if rng.random() < 0.15:
            rows.append({
                'customer_id': c['customer_id'],
                'product_code': 'PERSONAL_LOAN',
                'product_name': 'Personal Loan',
                'balance_gbp': round(float(rng.uniform(2000, 25000)), -2),
                'monthly_payment_gbp': round(float(rng.uniform(50, 500)), 0),
                'interest_rate': round(float(rng.uniform(4.0, 18.0)), 2),
                'start_date': c['customer_since'],
                'end_date': None,
                'status': 'Active',
            })
        if rng.random() < 0.40:
            rows.append({
                'customer_id': c['customer_id'],
                'product_code': 'CREDIT_CARD',
                'product_name': 'Credit Card',
                'balance_gbp': round(float(rng.uniform(0, 5000)), 2),
                'monthly_payment_gbp': round(float(rng.uniform(20, 200)), 0),
                'interest_rate': round(float(rng.uniform(15.0, 34.0)), 2),
                'start_date': c['customer_since'],
                'end_date': None,
                'status': 'Active',
            })
    return pd.DataFrame(rows)


def seed_database():
    from flask import Flask
    from backend.models import db, Customer, Account, FinancialProduct, Transaction
    from backend.models import TransactionClassification, CustomerFeatures, ProductCatalogue
    from backend.models import EmissionsFactor, DataGenerationRun
    from backend.config import Config
    from backend.services.classification_service import TransactionClassificationService

    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()

        # Skip if already seeded
        if DataGenerationRun.query.count() > 0:
            print('Database already seeded. Skipping.')
            return

        print('Generating synthetic data (seed=42)...')
        print('  Customers...')
        customers_df = generate_customers(NUM_CUSTOMERS)
        print(f'  Generated {len(customers_df)} customers')

        print('  Accounts...')
        accounts_df = generate_accounts(customers_df)
        print(f'  Generated {len(accounts_df)} accounts')

        print('  Transactions (this may take a moment)...')
        transactions_df = generate_transactions(customers_df, accounts_df)
        print(f'  Generated {len(transactions_df)} transactions')

        print('  Financial products...')
        products_df = generate_financial_products(customers_df)
        print(f'  Generated {len(products_df)} financial products')

        print('  Customer features...')
        features_df = compute_features(customers_df, transactions_df)
        print(f'  Computed features for {len(features_df)} customers')

        # Bulk insert customers
        print('  Inserting customers to DB...')
        for _, row in customers_df.iterrows():
            db.session.add(Customer(**row.to_dict()))
        db.session.commit()

        print('  Inserting accounts...')
        for _, row in accounts_df.iterrows():
            db.session.add(Account(**row.to_dict()))
        db.session.commit()

        print('  Inserting financial products...')
        for _, row in products_df.iterrows():
            db.session.add(FinancialProduct(**row.to_dict()))
        db.session.commit()

        print('  Inserting transactions...')
        classifier = TransactionClassificationService()
        batch = []
        cls_batch = []
        for _, row in transactions_df.iterrows():
            txn_dict = row.to_dict()
            cat = txn_dict.pop('_category')
            batch.append(Transaction(**txn_dict))
            result = classifier.classify(txn_dict['merchant_description'])
            # Use ground truth category with slight noise for realism
            if rng.random() < 0.9:
                use_cat = cat
                conf = 0.90 + float(rng.random() * 0.09)
            else:
                use_cat = result['category']
                conf = result['confidence']
            cls_batch.append(TransactionClassification(
                transaction_id=txn_dict['transaction_id'],
                category=use_cat,
                confidence=round(conf, 4),
                status='auto',
                is_manual=False,
                model_version='RULES-v1.0',
            ))
            if len(batch) >= 1000:
                db.session.bulk_save_objects(batch)
                db.session.bulk_save_objects(cls_batch)
                db.session.commit()
                batch = []
                cls_batch = []
                print(f'  Committed batch...', end='\r')
        if batch:
            db.session.bulk_save_objects(batch)
            db.session.bulk_save_objects(cls_batch)
            db.session.commit()

        print('  Inserting customer features...')
        for _, row in features_df.iterrows():
            db.session.add(CustomerFeatures(**row.to_dict()))
        db.session.commit()

        # Product catalogue
        print('  Seeding product catalogue...')
        products = [
            ProductCatalogue(product_code='GP01', product_name='EV Finance', category='Electric Vehicle', description='Finance for electric vehicle purchase or lease', min_propensity_threshold=0.3),
            ProductCatalogue(product_code='GP02', product_name='Green Home Loan', category='Home Improvement', description='Loan for energy efficiency home improvements', min_propensity_threshold=0.35),
            ProductCatalogue(product_code='GP03', product_name='Solar Finance', category='Renewable Energy', description='Finance for solar panel installation', min_propensity_threshold=0.30),
            ProductCatalogue(product_code='GP04', product_name='Paperless Banking', category='Digital', description='Switch to fully paperless banking', min_propensity_threshold=0.40),
        ]
        for p in products:
            db.session.add(p)
        db.session.commit()

        # Emissions factors
        print('  Seeding emissions factors...')
        factors = [
            EmissionsFactor(activity_type='petrol_car', factor_value=2.31, unit='kg_co2e_per_litre', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative petrol combustion factor'),
            EmissionsFactor(activity_type='diesel_car', factor_value=2.68, unit='kg_co2e_per_litre', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative diesel combustion factor'),
            EmissionsFactor(activity_type='electricity', factor_value=0.233, unit='kg_co2e_per_kwh', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative UK grid electricity factor'),
            EmissionsFactor(activity_type='natural_gas', factor_value=0.183, unit='kg_co2e_per_kwh', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative natural gas factor'),
            EmissionsFactor(activity_type='bus', factor_value=0.089, unit='kg_co2e_per_km', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative average bus factor'),
            EmissionsFactor(activity_type='rail', factor_value=0.035, unit='kg_co2e_per_km', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative average rail factor'),
            EmissionsFactor(activity_type='ev_car', factor_value=0.053, unit='kg_co2e_per_km', source='Illustrative - not official DEFRA/DESNZ', year=2023, version='ILLUSTRATIVE-v1.0', is_illustrative=True, notes='Illustrative EV well-to-wheel factor'),
        ]
        for f in factors:
            db.session.add(f)
        db.session.commit()

        # Log data generation run
        run = DataGenerationRun(
            random_seed=RANDOM_SEED,
            num_customers=len(customers_df),
            num_transactions=len(transactions_df),
            num_accounts=len(accounts_df),
            status='complete',
            notes='Initial synthetic dataset. All data is simulated. Labels are not observed outcomes.',
        )
        db.session.add(run)
        db.session.commit()

        print(f'\nSeed complete!')
        print(f'  Customers: {len(customers_df):,}')
        print(f'  Accounts:  {len(accounts_df):,}')
        print(f'  Transactions: {len(transactions_df):,}')
        print(f'  Financial products: {len(products_df):,}')


if __name__ == '__main__':
    # Ensure data directory exists
    data_dir = BASE_DIR / 'data'
    data_dir.mkdir(exist_ok=True)
    seed_database()
